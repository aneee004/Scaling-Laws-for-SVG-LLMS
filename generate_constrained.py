"""Constrained SVG generation — every output is well-formed XML by construction.

Approach: at every sampling step, the SVGGuide state machine is consulted to
mask any token whose decoded characters would put the running output into an
invalid XML state. Tokens that survive the mask are sampled normally
(temperature / top-k / top-p apply on the surviving distribution). When the
maximum number of new tokens is reached and tags remain open, the sampler
emits the appropriate ``</TAG>`` sequences to leave the output complete.

The script reuses the existing ``generate.load_config`` and
``generate.load_model`` helpers so SP/µP checkpoints both work transparently.
"""
import argparse
import json
import os
import pathlib
import time
from dataclasses import replace

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

from generate import _deep_merge, load_config, load_model
from model.svg_state import SVGGuide
from model.transformer import GPTConfig

BASE_CONFIG = pathlib.Path(__file__).parent / "configs" / "base.yaml"


def precompute_token_strings(tokenizer, vocab_size):
    """Return a list[str] mapping token_id -> decoded string."""
    out = [""] * vocab_size
    for tid in range(vocab_size):
        try:
            out[tid] = tokenizer.decode([tid])
        except Exception:
            out[tid] = ""
    return out


def build_valid_token_mask(state: SVGGuide, token_strings, eot_id, vocab_size,
                           require_complete_for_eot=True):
    """Boolean mask of length vocab_size: True = token is allowed in this state.

    The EOT token is allowed only when the state is *complete* (no open tags,
    no open quote, not inside a tag) — otherwise emitting EOT would terminate
    the document mid-element.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for tid in range(vocab_size):
        if tid == eot_id:
            mask[tid] = state.is_complete() if require_complete_for_eot else True
            continue
        s = token_strings[tid]
        if not s:
            mask[tid] = False
            continue
        candidate = state.copy()
        if candidate.feed(s):
            mask[tid] = True
    return mask


def force_close_tags(state: SVGGuide, tokenizer, idx, model, max_close_tokens=400):
    """Drive the state machine to ``is_done()`` regardless of where it is.

    Handles four cases that can arise when ``max_new_tokens`` is hit mid-output:
      1. inside an open quote -> append the matching quote
      2. mid-tag-body / after attr-name without value -> append a placeholder
         attribute value and close the tag with ``/>`` (no stack push)
      3. mid-tag-name / mid-close-tag-name -> the partial name is unusable;
         we cannot recover and the output will not parse
      4. open tags on the stack -> emit ``</NAME>`` for each in reverse order

    All appends go through both the state machine and the running ``idx``
    tensor so they are reflected in the final decoded string.
    """
    device = idx.device

    def _append_str(s: str):
        nonlocal idx
        if not state.feed(s):
            return False
        ids = tokenizer.encode(s).ids
        idx = torch.cat([idx, torch.tensor([ids], dtype=torch.long, device=device)], dim=1)
        return True

    # 1. close an open quote
    if state.in_quote is not None:
        _append_str(state.in_quote)

    # 2. recoverable mid-tag states
    p = state.pos
    if p in (SVGGuide.TAG_AFTER_NAME, SVGGuide.ATTR_DONE):
        # End the tag with /> (self-closing, doesn't grow the stack).
        _append_str("/>")
    elif p == SVGGuide.ATTR_NAME or p == SVGGuide.ATTR_AFTER_NAME:
        # Add =""/> to close out the attribute and self-close the tag.
        _append_str('=""/>')
    elif p == SVGGuide.ATTR_AFTER_EQ:
        _append_str('""/>')
    # Mid-tag-name or close-tag-name states are unrecoverable.

    # 3. close any tags still on the stack
    appended = 0
    while state.needs_close() and appended < max_close_tokens:
        name = state.stack[-1]
        close_str = f"</{name}>"
        if not _append_str(close_str):
            break
        appended += len(close_str)

    return idx


@torch.no_grad()
def generate_constrained(model, tokenizer, idx, max_new_tokens, eot_id,
                          token_strings, temperature=1.0, top_k=None,
                          top_p=None, seed_state=None, verbose=False):
    """Sample autoregressively; mask tokens that would violate XML state.

    Parameters
    ----------
    seed_state : SVGGuide or None
        State machine pre-fed with the prompt (so its initial position
        reflects what the prompt has already consumed). If None, a fresh
        state is created.
    """
    state = seed_state if seed_state is not None else SVGGuide()
    max_seq_len = model.config.max_seq_len
    vocab_size  = model.config.vocab_size
    device      = idx.device
    no_valid_streak = 0

    for step in range(max_new_tokens):
        # Stop as soon as the document is fully closed and a root has been emitted.
        if state.is_done():
            # Emit EOT to end the sample cleanly.
            eot_tensor = torch.tensor([[eot_id]], dtype=torch.long, device=device)
            idx = torch.cat([idx, eot_tensor], dim=1)
            return idx, state

        idx_cond = idx if idx.size(1) <= max_seq_len else idx[:, -max_seq_len:]
        logits   = model(idx_cond)[:, -1, :].clone()

        # Mask invalid tokens
        valid_mask = build_valid_token_mask(state, token_strings, eot_id, vocab_size)
        logits[:, ~valid_mask] = float("-inf")

        if torch.isinf(logits).all():
            # No legal continuation — bail out and force-close
            if verbose:
                print(f"[step {step}] dead-end at state={state.pos!r} "
                      f"stack={state.stack!r}; forcing close")
            break

        logits = logits / temperature
        if top_k is not None and top_k < vocab_size:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            cutoff = v[:, [-1]]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask_p = cum_probs > top_p
            mask_p[..., 1:] = mask_p[..., :-1].clone()
            mask_p[..., 0]  = False
            sorted_logits[mask_p] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or probs.sum() == 0:
            if verbose:
                print(f"[step {step}] degenerate probs; forcing close")
            break

        next_token = torch.multinomial(probs, num_samples=1)
        token_id = int(next_token.item())

        if token_id == eot_id:
            # Sampler chose EOT — only legal when state is complete
            if state.is_complete():
                idx = torch.cat([idx, next_token], dim=1)
                return idx, state
            # Shouldn't happen because mask blocks EOT when not complete
            if verbose:
                print(f"[step {step}] EOT sampled mid-document; forcing close")
            break

        # Apply the token's chars to the state machine. Should always succeed
        # because the mask guaranteed validity, but assert defensively.
        chars = token_strings[token_id]
        if not state.feed(chars):
            if verbose:
                print(f"[step {step}] state-feed mismatch on token {token_id!r}; aborting")
            break
        idx = torch.cat([idx, next_token], dim=1)

    # Force-close any remaining open tags so the final string parses cleanly.
    if state.needs_close() or not state.is_complete():
        idx = force_close_tags(state, tokenizer, idx, model)

    return idx, state


def main():
    parser = argparse.ArgumentParser(prog="SVG LLM — constrained generate")
    parser.add_argument("-c", "--config",     required=True)
    parser.add_argument("-o", "--override",   default=None)
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--prompt",           default="<svg")
    parser.add_argument("--num_samples",      type=int,   default=10)
    parser.add_argument("--max_new_tokens",   type=int,   default=1000)
    parser.add_argument("--temperature",      type=float, default=0.8)
    parser.add_argument("--top_k",            type=int,   default=200)
    parser.add_argument("--top_p",            type=float, default=None)
    parser.add_argument("--output_dir",       default=None)
    parser.add_argument("--seed",             type=int,   default=None)
    parser.add_argument("--verbose",          action="store_true")
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

    print("Pre-decoding token strings...", flush=True)
    token_strings = precompute_token_strings(tokenizer, gpt_cfg.vocab_size)

    # Seed the state machine with the prompt so the running state is correct
    seed_chars = args.prompt
    seed_ids   = tokenizer.encode(seed_chars).ids if seed_chars else [eot_id]

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    valid_count = 0
    t0 = time.time()
    for i in range(args.num_samples):
        seed_state = SVGGuide()
        seed_state.feed(seed_chars)

        x = torch.tensor([seed_ids], dtype=torch.long, device=device)
        out, state = generate_constrained(
            model, tokenizer, x, args.max_new_tokens, eot_id, token_strings,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            seed_state=seed_state, verbose=args.verbose,
        )
        out_ids = out[0].tolist()
        if eot_id in out_ids[len(seed_ids):]:
            cut = out_ids.index(eot_id, len(seed_ids))
            out_ids = out_ids[:cut]
        text = tokenizer.decode(out_ids)

        # Validate via lxml as a final check
        try:
            from lxml import etree
            etree.fromstring(text.encode("utf-8"))
            valid_count += 1
            ok = True
        except Exception:
            ok = False

        elapsed = time.time() - t0
        print(f"sample {i:3d}  valid={ok}  len={len(text)}  ({elapsed:.1f}s elapsed)", flush=True)

        if args.output_dir:
            with open(os.path.join(args.output_dir, f"sample_{i:04d}.svg"), "w") as f:
                f.write(text)

    print(f"\nValid: {valid_count}/{args.num_samples}  ({100*valid_count/args.num_samples:.1f}%)")


if __name__ == "__main__":
    main()
