import argparse
import json
import math
import os
import pathlib
from contextlib import nullcontext

import numpy as np
import torch
import yaml
from tokenizers import ByteLevelBPETokenizer

from generate import _deep_merge, generate, load_model
from model.transformer import GPTConfig

BASE_CONFIG = pathlib.Path(__file__).parent / "configs" / "base.yaml"

try:
    from lxml import etree
    HAS_LXML = True
except ImportError:
    HAS_LXML = False

try:
    import cairosvg
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False


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

    return (
        gpt_cfg,
        cfg["device"]["device"],
        cfg["device"]["dtype"],
        cfg["tokenizer"]["save_path"],
        cfg["data"]["output_dir"],
    )


@torch.no_grad()
def compute_perplexity(model, test_data, seq_len, batch_size, device, ctx, max_windows=None):
    """Token-level perplexity over non-overlapping windows of test_data."""
    model.eval()
    n_windows = (len(test_data) - 1) // seq_len
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    if n_windows == 0:
        raise ValueError(f"Test data too short for seq_len={seq_len}")

    total_loss_weighted = 0.0
    total_tokens        = 0

    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)
        x_list, y_list = [], []
        for w in range(batch_start, batch_end):
            off = w * seq_len
            x_list.append(torch.from_numpy(test_data[off     : off + seq_len    ].astype(np.int64)))
            y_list.append(torch.from_numpy(test_data[off + 1 : off + seq_len + 1].astype(np.int64)))
        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)

        with ctx:
            loss = model.loss(x, y)

        n = x.numel()
        total_loss_weighted += loss.item() * n
        total_tokens        += n

    avg_loss = total_loss_weighted / total_tokens
    return avg_loss, math.exp(avg_loss)


def is_valid_xml(svg_str):
    try:
        etree.fromstring(svg_str.encode("utf-8"))
        return True
    except (etree.XMLSyntaxError, ValueError):
        return False


def is_structurally_valid(svg_str):
    """Stronger than well-formed XML: must have an <svg> root and required attrs.

    Per the PDF: 'correct <svg> root element, properly closed tags, valid
    attribute values'. Properly-closed-tags is implied by lxml parsing
    successfully; the additional checks live here.
    """
    try:
        root = etree.fromstring(svg_str.encode("utf-8"))
    except (etree.XMLSyntaxError, ValueError):
        return False
    # Strip namespace if present, e.g. {http://www.w3.org/2000/svg}svg
    tag = root.tag.split("}")[-1]
    if tag != "svg":
        return False
    # Need at least one of viewBox / (width AND height) — these are what cairosvg uses to render
    has_viewbox = root.get("viewBox") is not None
    has_size    = root.get("width") is not None and root.get("height") is not None
    return has_viewbox or has_size


def can_render(svg_str):
    try:
        cairosvg.svg2png(bytestring=svg_str.encode("utf-8"))
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(prog="SVG LLM — evaluate")
    parser.add_argument("-c", "--config",    required=True)
    parser.add_argument("-o", "--override",  default=None)
    parser.add_argument("--checkpoint",      required=True)
    parser.add_argument("--num_samples",     type=int,   default=100)
    parser.add_argument("--max_new_tokens",  type=int,   default=1024)
    parser.add_argument("--temperature",     type=float, default=0.8)
    parser.add_argument("--top_k",           type=int,   default=200)
    parser.add_argument("--top_p",           type=float, default=None,
                        help="Nucleus sampling threshold (composes with top_k).")
    parser.add_argument("--temperature_sweep", type=float, nargs="+", default=None,
                        help="Sweep multiple temperatures (e.g. 0.5 0.8 1.0). "
                             "If set, generates --num_samples per temperature.")
    parser.add_argument("--batch_size",      type=int,   default=8,
                        help="batch size for perplexity (independent of training)")
    parser.add_argument("--max_perplexity_windows", type=int, default=None,
                        help="cap windows used for perplexity (default: all)")
    parser.add_argument("--skip_generation", action="store_true",
                        help="compute only perplexity, skip XML/render checks")
    parser.add_argument("--output_json",     default=None)
    parser.add_argument("--seed",            type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    gpt_cfg, device_str, dtype_str, tok_path, data_dir = load_config(args.config, args.override)
    device = torch.device(device_str)

    tokenizer = ByteLevelBPETokenizer.from_file(
        os.path.join(tok_path, "vocab.json"),
        os.path.join(tok_path, "merges.txt"),
    )
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    if eot_id is None:
        raise ValueError("Tokenizer has no <|endoftext|> token.")

    model = load_model(args.checkpoint, gpt_cfg, device)
    test_data = np.load(os.path.join(data_dir, "test.npy"))
    print(f"Test data: {len(test_data):,} tokens")

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_str]
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if device_str == "cuda" else nullcontext()
    )

    # ----- Perplexity -----
    print("\nComputing perplexity...")
    avg_loss, ppl = compute_perplexity(
        model, test_data, gpt_cfg.max_seq_len, args.batch_size, device, ctx,
        max_windows=args.max_perplexity_windows,
    )
    print(f"  avg cross-entropy: {avg_loss:.4f}")
    print(f"  perplexity:        {ppl:.4f}")

    results = {
        "perplexity":  ppl,
        "avg_loss":    avg_loss,
        "test_tokens": len(test_data),
    }

    # ----- Sample-based metrics -----
    def _sample_batch(temperature):
        """Generate args.num_samples unconditional samples at the given temperature."""
        out_samples = []
        for i in range(args.num_samples):
            x = torch.tensor([[eot_id]], dtype=torch.long, device=device)
            out = generate(
                model, x, args.max_new_tokens,
                temperature=temperature, top_k=args.top_k, top_p=args.top_p, eot_id=eot_id,
            )
            out_ids = out[0].tolist()
            if eot_id in out_ids[1:]:
                cut = out_ids.index(eot_id, 1)
                out_ids = out_ids[:cut]
            out_samples.append(tokenizer.decode(out_ids))
            if (i + 1) % 10 == 0:
                print(f"  T={temperature}  {i + 1}/{args.num_samples}")
        return out_samples

    def _score_samples(samples_list):
        out = {}
        if HAS_LXML:
            xml_ok = [is_valid_xml(s) for s in samples_list]
            struct_ok = [is_structurally_valid(s) for s in samples_list]
            out["xml_rate"]    = sum(xml_ok)    / len(samples_list)
            out["struct_rate"] = sum(struct_ok) / len(samples_list)
            print(f"  XML well-formed:     {sum(xml_ok)}/{len(samples_list)} = {out['xml_rate']:.2%}")
            print(f"  Structurally valid:  {sum(struct_ok)}/{len(samples_list)} = {out['struct_rate']:.2%}")
            if HAS_CAIROSVG:
                valid = [s for s, ok in zip(samples_list, xml_ok) if ok]
                rendered = sum(can_render(s) for s in valid)
                out["render_rate_overall"]  = rendered / len(samples_list)
                out["render_rate_in_valid"] = rendered / max(1, len(valid))
                print(f"  Render rate (overall):    {rendered}/{len(samples_list)} = {out['render_rate_overall']:.2%}")
        return out

    if not args.skip_generation:
        if args.temperature_sweep:
            results["per_temperature"] = {}
            for T in args.temperature_sweep:
                print(f"\n=== Generating {args.num_samples} samples @ T={T} ===")
                samples = _sample_batch(T)
                results["per_temperature"][str(T)] = _score_samples(samples)
        else:
            print(f"\nGenerating {args.num_samples} unconditional samples @ T={args.temperature}...")
            samples = _sample_batch(args.temperature)
            results.update(_score_samples(samples))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
