import argparse
import csv
import inspect
import math
import os
import pathlib
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch
import yaml
from mup import MuAdamW, set_base_shapes

from model.transformer import GPT, GPTConfig

BASE_CONFIG = pathlib.Path(__file__).parent / "configs" / "base.yaml"


# Width of the proxy base model used by set_base_shapes().
# Must be the same across all model sizes in a µP sweep so that
# LR scaling is consistent.
MUP_BASE_WIDTH = 128


@dataclass
class TrainingConfig:
    batch_tokens: int
    micro_batch_size: int
    lr: float
    min_lr: float
    warmup_frac: float
    grad_clip: float
    weight_decay: float
    eval_interval: int
    save_interval: int
    checkpoint_dir: str
    log_backend: str
    max_steps: Optional[int] = None


@dataclass
class DeviceConfig:
    device: str
    dtype: str


def _deep_merge(base, override):
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path, override_path):
    with open(BASE_CONFIG) as fp:
        cfg = yaml.safe_load(fp)
    with open(config_path) as fp:
        cfg = _deep_merge(cfg, yaml.safe_load(fp))
    if override_path:
        with open(override_path) as fp:
            cfg = _deep_merge(cfg, yaml.safe_load(fp))

    model_dict = dict(cfg["model"])
    model_dict["vocab_size"] = cfg["tokenizer"]["vocab_size"]
    model_dict["mup"] = True                        # always True in this script
    gpt_cfg = GPTConfig(**model_dict)

    train_cfg  = TrainingConfig(**cfg["training"])
    device_cfg = DeviceConfig(**cfg["device"])
    data_dir   = cfg["data"]["output_dir"]

    return gpt_cfg, train_cfg, device_cfg, data_dir


def configure_mup_model(gpt_cfg, device):
    """Instantiate model and call set_base_shapes() before any optimizer is built."""
    base_cfg = replace(gpt_cfg, n_embd=MUP_BASE_WIDTH)
    base_model = GPT(base_cfg)

    model = GPT(gpt_cfg)
    set_base_shapes(model, base_model)   # MUST come before optimizer construction
    del base_model

    return model.to(device)


def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i     : i + seq_len    ].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + seq_len + 1].astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)


def get_lr(step, train_cfg):
    warmup_steps = int(train_cfg.warmup_frac * train_cfg.max_steps)
    if step < warmup_steps:
        return train_cfg.lr * step / max(1, warmup_steps)
    if step >= train_cfg.max_steps:
        return train_cfg.min_lr
    progress = (step - warmup_steps) / (train_cfg.max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return train_cfg.min_lr + coeff * (train_cfg.lr - train_cfg.min_lr)


def configure_optimizer(model, train_cfg, device_cfg):
    """MuAdamW applies per-parameter LR scaling set by set_base_shapes()."""
    param_dict    = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params  = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups  = [
        {"params": decay_params,   "weight_decay": train_cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return MuAdamW(optim_groups, lr=train_cfg.lr, betas=(0.9, 0.95))


@torch.no_grad()
def evaluate(model, val_data, seq_len, batch_size, device, ctx, eval_batches=20):
    model.eval()
    losses = [
        model.loss(*get_batch(val_data, seq_len, batch_size, device)).item()
        for _ in range(eval_batches)
    ]
    model.train()
    return sum(losses) / len(losses)


def train(model, train_data, val_data, train_cfg, device_cfg, gpt_cfg, checkpoint_dir):
    device = torch.device(device_cfg.device)
    dtype  = {"float32": torch.float32, "bfloat16": torch.bfloat16}[device_cfg.dtype]
    ctx    = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if device_cfg.device == "cuda" else nullcontext()
    )

    if train_cfg.max_steps is None:
        steps = max(1, math.ceil(len(train_data) / train_cfg.batch_tokens))
        print(f"max_steps unset — auto = 1 epoch = {steps} steps")
        train_cfg = replace(train_cfg, max_steps=steps)

    seq_len          = gpt_cfg.max_seq_len
    micro_batch_size = train_cfg.micro_batch_size
    grad_accum_steps = train_cfg.batch_tokens // (seq_len * micro_batch_size)
    assert grad_accum_steps >= 1

    # Optimizer MUST be built after configure_mup_model() has called set_base_shapes()
    optimizer = configure_optimizer(model, train_cfg, device_cfg)
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_path = os.path.join(checkpoint_dir, "log.csv")
    log_file = open(log_path, "w", newline="")
    logger   = csv.writer(log_file)
    logger.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])
    log_file.flush()

    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(train_cfg.max_steps):
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = get_batch(train_data, seq_len, micro_batch_size, device)
            with ctx:
                loss = model.loss(x, y) / grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if step % train_cfg.eval_interval == 0:
            val_loss = evaluate(model, val_data, seq_len, micro_batch_size, device, ctx)
            elapsed  = time.time() - t0
            print(f"step {step:5d} | train {accum_loss:.4f} | val {val_loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            logger.writerow([step, round(accum_loss, 6), round(val_loss, 6), lr, round(elapsed, 1)])
            log_file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "mup_base_width": MUP_BASE_WIDTH,
                    },
                    os.path.join(checkpoint_dir, "best.pt"),
                )

        if step % train_cfg.save_interval == 0 and step > 0:
            torch.save(
                {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                os.path.join(checkpoint_dir, f"ckpt_{step:06d}.pt"),
            )

    log_file.close()
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(prog="SVG LLM — µP train")
    parser.add_argument("-c", "--config",   required=True)
    parser.add_argument("-o", "--override", default=None)
    args = parser.parse_args()

    gpt_cfg, train_cfg, device_cfg, data_dir = load_config(args.config, args.override)

    device     = torch.device(device_cfg.device)
    train_data = np.load(os.path.join(data_dir, "train.npy"))
    val_data   = np.load(os.path.join(data_dir, "val.npy"))

    # set_base_shapes() is called inside configure_mup_model — before optimizer
    model = configure_mup_model(gpt_cfg, device)

    config_name    = os.path.splitext(os.path.basename(args.config))[0]
    checkpoint_dir = os.path.join(train_cfg.checkpoint_dir, f"{config_name}_mup")

    train(model, train_data, val_data, train_cfg, device_cfg, gpt_cfg, checkpoint_dir)


if __name__ == "__main__":
    main()
