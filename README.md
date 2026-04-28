# Scaling Laws for SVG Transformers

CS-GY 6923 ML Final Project (NYU Tandon, Spring 2026) — Aniruth R.

Empirical scaling-law study for decoder-only Transformer language models trained
on SVG (Scalable Vector Graphics) source code. Replicates the Kaplan-style power
law `L = a · N^(-α) + c` over five model sizes from ~1M to ~88M non-embedding
parameters, then re-runs the sweep under µP (Yang et al. 2022) and extrapolates
the best LR ~10× past the largest fitted size.

## Project Parts

1. **Data Collection & Preprocessing** — `starvector/svg-stack`, byte-level BPE
   (vocab 4096), 100M-token train budget, 98/1/1 split.
2. **Transformer Scaling Study** — five SP configs (`1m` … `88m`), LR sweep per
   size, fit power law on val loss.
3. **µP Scaling & Extrapolation** — same configs under Microsoft `mup`,
   compare SP vs µP curves, extrapolate the optimal LR.
4. **Best Model Training & Sample Generation** — train compute-optimal config to
   completion; unconditional and prefix-conditioned samples; XML/render
   validity metrics.
5. **Design Decisions & Analysis** — written discussion in `report/report.tex`.

## Stack

- PyTorch (MPS for local M4 dev, CUDA for Colab L4 runs)
- HuggingFace `datasets` + `tokenizers`
- nanoGPT as the SP transformer baseline
- Microsoft `mup` for µP reparameterization
- `lxml` (XML validity), `cairosvg` (render validity)
- `scipy.optimize.curve_fit` for the power-law fit

## Repo Layout

```
project/
  data/
    prepare.py           # download → filter → split → tokenize → save
    processed/           # train.npy, val.npy, test.npy (uint16)
  tokenizers/
    bpe_4096/            # vocab.json, merges.txt
  configs/
    base.yaml            # shared defaults
    {1m,3m,12m,34m,88m}.yaml   # per-size overrides (planned)
  model/
    transformer.py       # nanoGPT-style decoder (planned)
  train.py               # token-budget driven training loop (planned)
  sweep_lr.py            # short LR sweep per size (planned)
  mup_train.py           # µP variant (planned)
  generate.py            # sampling, prefix conditioning (planned)
  evaluate.py            # perplexity, XML validity, render rate (planned)
  report/report.tex      # final writeup
  requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`DEVICE=mps` for local iteration on the M4 Mac; `DEVICE=cuda` for the Colab L4
runs that produce the actual scaling-law data points.

## Usage

```bash
# Stage 1 — data pipeline (dev mode = 1000 samples for smoke testing)
python data/prepare.py --config configs/base.yaml --dev
python data/prepare.py --config configs/base.yaml         # full 100M-token run

# Stage 2 — single-size train + LR sweep (planned)
python train.py     --config configs/1m.yaml
python sweep_lr.py  --config configs/1m.yaml

# Stage 3 — µP variant (planned)
python mup_train.py --config configs/1m_mup.yaml

# Stage 4 — sample + evaluate the best model (planned)
python generate.py  --checkpoint checkpoints/best/best.pt --prompt "<svg"
python evaluate.py  --checkpoint checkpoints/best/best.pt
```

## Key Design Decisions

- Vocab 4096 BPE — SVG has heavy substring repetition; a moderate vocab captures
  common tag/attribute fragments without over-fragmenting the long tail.
- Max sequence length 1024 — balances context against compute on the 1M model.
- Train/val/test split is by file count (98/1/1), not by token position, to keep
  whole SVGs intact across splits.
- µP attention scale `1/d` (not `1/√d`) per Tensor Programs V.
- Token budget per run rather than fixed steps, so different sizes are
  comparable on the loss-vs-compute axis.
- LR schedule: cosine with 5% warmup; AdamW with weight decay 0.1.

## Status

- Stage 1 (data pipeline) — implemented; dev-mode verified (1000 samples →
  ~645K train tokens, decoded sample is valid SVG). Full 100M-token run
  pending.
- Stages 2–5 — in progress.

## References

- Kaplan et al. 2020. *Scaling Laws for Neural Language Models.*
- Hoffmann et al. 2022. *Training Compute-Optimal Large Language Models* (Chinchilla).
- Yang et al. 2022. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.*
- Rodriguez et al. 2023. *StarVector: Generating Scalable Vector Graphics Code from Images.*
