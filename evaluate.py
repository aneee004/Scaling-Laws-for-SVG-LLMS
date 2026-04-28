import argparse
import json
import math
import os
from contextlib import nullcontext

import numpy as np
import torch
import yaml
from tokenizers import ByteLevelBPETokenizer

from generate import _deep_merge, generate, load_model
from model.transformer import GPTConfig

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
    with open(config_path) as fp:
        cfg = yaml.safe_load(fp)
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
    if not args.skip_generation:
        print(f"\nGenerating {args.num_samples} unconditional samples...")
        samples = []
        for i in range(args.num_samples):
            x = torch.tensor([[eot_id]], dtype=torch.long, device=device)
            out = generate(
                model, x, args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k, eot_id=eot_id,
            )
            out_ids = out[0].tolist()
            if eot_id in out_ids[1:]:
                cut = out_ids.index(eot_id, 1)
                out_ids = out_ids[:cut]
            samples.append(tokenizer.decode(out_ids))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{args.num_samples}")

        if HAS_LXML:
            xml_ok    = [is_valid_xml(s) for s in samples]
            xml_count = sum(xml_ok)
            xml_rate  = xml_count / len(samples)
            print(f"\nXML well-formedness: {xml_count}/{len(samples)} = {xml_rate:.2%}")
            results["xml_valid"] = xml_count
            results["xml_rate"]  = xml_rate

            if HAS_CAIROSVG:
                valid_samples = [s for s, ok in zip(samples, xml_ok) if ok]
                rendered      = sum(can_render(s) for s in valid_samples)
                in_valid_rate = rendered / max(1, len(valid_samples))
                overall_rate  = rendered / len(samples)
                print(f"Render rate (of XML-valid): {rendered}/{len(valid_samples)} = {in_valid_rate:.2%}")
                print(f"Render rate (overall):     {rendered}/{len(samples)} = {overall_rate:.2%}")
                results["rendered"]            = rendered
                results["render_rate_in_valid"] = in_valid_rate
                results["render_rate_overall"]  = overall_rate
            else:
                print("cairosvg not installed — skipping render check.")
        else:
            print("\nlxml not installed — skipping XML/render checks.")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
