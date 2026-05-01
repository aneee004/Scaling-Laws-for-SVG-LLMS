import argparse
import os
import pathlib
from dataclasses import replace

import torch
import torch.nn.functional as F
import yaml
from tokenizers import ByteLevelBPETokenizer

from model.transformer import GPT, GPTConfig

BASE_CONFIG = pathlib.Path(__file__).parent / "configs" / "base.yaml"


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
    gpt_cfg = GPTConfig(**model_dict)

    device   = cfg["device"]["device"]
    tok_path = cfg["tokenizer"]["save_path"]
    return gpt_cfg, device, tok_path


def load_model(checkpoint_path, gpt_cfg, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # µP checkpoints carry mup_base_width; SP checkpoints don't.
    if "mup_base_width" in ckpt:
        gpt_cfg = replace(gpt_cfg, mup=True)
        from mup import set_base_shapes
        base_cfg   = replace(gpt_cfg, n_embd=ckpt["mup_base_width"])
        base_model = GPT(base_cfg)
        model      = GPT(gpt_cfg)
        set_base_shapes(model, base_model)
        del base_model
    else:
        model = GPT(gpt_cfg)

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def _apply_top_k(logits, top_k):
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = float("-inf")
    return logits


def _apply_top_p(logits, top_p):
    """Nucleus filter: keep the smallest set of tokens whose cumulative probability >= top_p."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Mask tokens beyond the nucleus, but always keep at least the top-1 token.
    mask = cum_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0]  = False
    sorted_logits[mask] = float("-inf")
    # Scatter back into original index order
    return torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0,
             top_k=None, top_p=None, eot_id=None):
    """Sample tokens autoregressively. idx: [B, T] of token ids on the model's device."""
    max_seq_len = model.config.max_seq_len
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= max_seq_len else idx[:, -max_seq_len:]
        logits   = model(idx_cond)[:, -1, :] / temperature

        if top_k is not None:
            logits = _apply_top_k(logits, top_k)
        if top_p is not None and top_p < 1.0:
            logits = _apply_top_p(logits, top_p)

        probs      = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx        = torch.cat([idx, next_token], dim=1)

        if eot_id is not None and (next_token == eot_id).all():
            break
    return idx


def main():
    parser = argparse.ArgumentParser(prog="SVG LLM — generate")
    parser.add_argument("-c", "--config",     required=True)
    parser.add_argument("-o", "--override",   default=None)
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--prompt",           default="<svg",
                        help="Prefix to condition on. Use '' for unconditional (starts from EOT).")
    parser.add_argument("--num_samples",      type=int,   default=1)
    parser.add_argument("--max_new_tokens",   type=int,   default=512)
    parser.add_argument("--temperature",      type=float, default=0.8)
    parser.add_argument("--top_k",            type=int,   default=200)
    parser.add_argument("--top_p",            type=float, default=None,
                        help="Nucleus sampling threshold. Set <1.0 to enable; combines with top_k.")
    parser.add_argument("--output_dir",       default=None,
                        help="Save samples here (one file per sample). If omitted, prints to stdout.")
    parser.add_argument("--seed",             type=int,   default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    gpt_cfg, device_str, tok_path = load_config(args.config, args.override)
    device = torch.device(device_str)

    tokenizer = ByteLevelBPETokenizer.from_file(
        os.path.join(tok_path, "vocab.json"),
        os.path.join(tok_path, "merges.txt"),
    )
    eot_id = tokenizer.token_to_id("<|endoftext|>")

    model = load_model(args.checkpoint, gpt_cfg, device)

    if args.prompt:
        prompt_ids = tokenizer.encode(args.prompt).ids
    else:
        if eot_id is None:
            raise ValueError("Unconditional generation requires an <|endoftext|> token in the tokenizer.")
        prompt_ids = [eot_id]

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num_samples):
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = generate(
            model, x, args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            eot_id=eot_id,
        )
        out_ids = out[0].tolist()

        # Trim at first EOT after the prompt so output doesn't include <|endoftext|>.
        if eot_id is not None and eot_id in out_ids[len(prompt_ids):]:
            cut = out_ids.index(eot_id, len(prompt_ids))
            out_ids = out_ids[:cut]

        text = tokenizer.decode(out_ids)

        if args.output_dir:
            path = os.path.join(args.output_dir, f"sample_{i:04d}.svg")
            with open(path, "w") as f:
                f.write(text)
            print(f"sample {i}: {path}")
        else:
            print(f"--- sample {i} ---")
            print(text)
            print()


if __name__ == "__main__":
    main()
