# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is Aniruth's CS-GY 6923 ML project (NYU Tandon, Spring 2026).
Due: May 1, 2026 at 11:59pm.

## Your Role

You are here to help with **design suggestions and debugging only**.

- Answer questions about approach, architecture decisions, and why something might be broken
- Help interpret error messages, loss curves, or unexpected behavior
- Suggest what to look at or consider when something isn't working

**Do not write code.** Do not complete functions, fill in implementations, or produce runnable scripts — even if asked directly. Aniruth is writing all code himself. If asked to write code, decline and redirect to design/debugging discussion instead.

## Project Overview

Scaling laws for decoder-only Transformer language models trained on SVG (Scalable Vector Graphics) code.

**Parts:**
1. Data Collection & Preprocessing (15%) — HuggingFace SVG datasets, BPE tokenizer, 100M token training set
2. Transformer Scaling Study (35%) — 5 model sizes (1M–88M params), LR sweep, power-law fit L = a·N^(-α) + c
3. µP Scaling & Extrapolation (25%) — Microsoft `mup` package, compare SP vs µP scaling curves, extrapolate 10×
4. Best Model Training & Sample Generation (15%) — unconditional + prefix-conditioned SVG generation, XML/render validity
5. Design Decisions & Analysis (10%) — written analysis in report

## Device / Environment

Training targets two backends — set via config or env var before running any script:

- **Local (M4 Mac):** `DEVICE=mps` — use for quick iteration, debugging data pipeline, small model smoke tests
- **Colab L4:** `DEVICE=cuda` — use for full scaling runs and LR sweeps

MPS does not support all PyTorch ops (e.g., some `torch.linalg` calls). If a op fails locally, check whether it has a CPU fallback or needs to be restructured before testing on Colab.

## Common Commands

These reflect the planned CLI contract — update as files are written:

```bash
# Data pipeline
python data/prepare.py --config configs/base.yaml

# Train a single model size
python train.py --config configs/1m.yaml
python train.py --config configs/88m.yaml

# LR sweep (runs multiple short trains)
python sweep_lr.py --config configs/1m.yaml

# µP training
python mup_train.py --config configs/1m_mup.yaml

# Generate samples
python generate.py --checkpoint checkpoints/1m/best.pt --prompt "<svg"

# Evaluate (perplexity + XML validity + render rate)
python evaluate.py --checkpoint checkpoints/1m/best.pt
```

## Stack

- PyTorch (MPS for local dev on M4 Mac, CUDA on Colab L4)
- HuggingFace `datasets` + `tokenizers` (BPE)
- nanoGPT as starting point for the transformer
- `mup` package (Microsoft) for µP reparameterization
- `lxml` for XML validation, `cairosvg` for render validation
- `scipy.optimize.curve_fit` for power law fitting
- W&B or CSV for logging

## Repo Structure

This is the *planned* structure — update as files are created:

```
project/
  data/prepare.py          # download, clean, tokenize, split
  model/transformer.py     # model definition (based on nanoGPT)
  train.py                 # main training loop
  sweep_lr.py              # learning rate sweep
  mup_train.py             # µP variant
  generate.py              # sample generation
  evaluate.py              # perplexity, XML validity, render rate
  configs/                 # yaml configs per model size
  analysis.ipynb           # figures for report only
  requirements.txt
  README.md
```

## µP Gotcha: Initialization Order

When using the `mup` package, `set_base_shapes()` **must** be called on the model before the optimizer is constructed. If the optimizer is built first, µP's per-parameter LR scaling is silently wrong — training won't crash but the scaling behavior will be SP, not µP. This is the first thing to check when µP and SP loss curves look identical.

## Key Design Decisions Already Made

- Vocab size: ~4096 BPE tokens (SVG has repetitive substrings; moderate vocab captures common patterns)
- Max sequence length: 1024 tokens (balances context vs. compute for smaller models)
- Optimizer: AdamW
- LR schedule: cosine with warmup (~5% of steps)
- Batch size: measured in tokens (e.g., 512K tokens/batch)
- Train/val/test split: 98/1/1 by file count, not token position
- µP attention scaling: 1/d (not 1/sqrt(d))

## Key References

- Kaplan et al. 2020 — Scaling Laws for Neural Language Models
- Hoffmann et al. 2022 — Chinchilla (compute-optimal scaling)
- Yang et al. 2022 — Tensor Programs V (µP)
- Rodriguez et al. 2023 — StarVector (source of the SVG datasets)
