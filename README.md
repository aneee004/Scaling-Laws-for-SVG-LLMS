# Scaling Laws for SVG Transformers

CS-GY 6923 ML Final Project (NYU Tandon, Spring 2026) — Aniruth R.

Empirical scaling-law study for decoder-only Transformer language models trained
on SVG (Scalable Vector Graphics) source code. Replicates the Kaplan-style power
law `L = a · N^(-α) + c` over five model sizes from ~1M to ~88M non-embedding
parameters, then re-runs the sweep under µP (Yang et al. 2022) and extrapolates
the best LR ~10× past the largest fitted size.

## Project Parts

1. **Data Collection & Preprocessing** — `starvector/svg-stack-simple` (chosen over the
   PDF-recommended `svg-icons-simple` to hit the 100M-token target without
   supplementing from multiple sources; see report §2 for justification).
   Pipeline: filter by length → strip XML comments and collapse whitespace →
   round numerics to 1 decimal → validate XML → tokenise (byte-level BPE,
   vocab 4096) → 98/1/1 split.
2. **Transformer Scaling Study (SP)** — five sizes (`1m` … `88m`), LR sweep on
   the smallest, single LR re-used across sizes, 1-epoch training,
   power-law fit on val loss.
3. **µP Scaling & Extrapolation** — same five sizes under Microsoft `mup`,
   independent µP LR sweep on the smallest, transferred to all larger sizes.
   Power-law fit + extrapolation to ~10× the largest fitted scale.
4. **Best Model Training & Sample Generation** — trains the largest model to
   convergence; unconditional and prefix-conditioned generation with
   temperature, top-k, and top-p sampling; XML / structural / render validity
   metrics.
5. **Design Decisions & Analysis** — written discussion in `report/report.tex`,
   figures in `analysis.ipynb`.

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
    prepare.py                 # download → filter → split → tokenize → save
    processed/                 # train.npy, val.npy, test.npy (uint16, gitignored)
  tokenizers/
    bpe_4096/                  # vocab.json, merges.txt
  configs/
    base.yaml                  # shared defaults
    colab.yaml                 # Drive-path + cuda overrides for Colab
    {1m,3m,12m,34m,88m}.yaml   # per-size overrides (n_layer/n_head/n_embd)
  model/
    transformer.py             # nanoGPT-style decoder, mup flag for µP runs
  train.py                     # SP training loop
  sweep_lr.py                  # short LR sweep per size
  mup_train.py                 # µP variant — set_base_shapes + MuAdamW
  generate.py                  # sampling (temperature/top-k, prefix conditioning)
  evaluate.py                  # perplexity, XML validity, render rate
  analysis.ipynb               # power-law fits, training curves, scaling figures
  colab/                       # local-only Colab notebooks (gitignored)
    smoke.ipynb                #   end-to-end smoke test on Colab L4
    pipeline.ipynb             #   full pipeline (data → train → eval → analysis)
  report/
    report.tex                 # final writeup
    figures/                   # PDFs produced by analysis.ipynb
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
python data/prepare.py -c configs/base.yaml --dev
python data/prepare.py -c configs/base.yaml -o configs/colab.yaml   # full run on Colab

# Stage 2 — SP training (1 epoch, max_steps auto-computed if null)
python sweep_lr.py -c configs/1m.yaml --lrs 1e-4 3e-4 1e-3 3e-3 1e-2   # SP sweep on smallest only
python train.py    -c configs/1m.yaml -o configs/colab.yaml            # full SP run, repeat per size

# Stage 3 — µP variant. Independent LR sweep on 1m, transfer to all sizes.
python sweep_lr.py -c configs/1m.yaml --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 --mup
python mup_train.py -c configs/1m.yaml -o configs/colab.yaml
python mup_train.py -c configs/88m.yaml -o configs/colab.yaml

# Stage 4 — sample + evaluate the best model
# Generation supports temperature, top-k, AND top-p (nucleus) sampling.
python generate.py -c configs/88m.yaml --checkpoint checkpoints/88m/best.pt \
    --prompt "<svg" --temperature 0.8 --top_k 200 --top_p 0.9 --num_samples 10 \
    --output_dir samples/best/

# Evaluate: perplexity + XML validity + structural validity + render rate.
# Use --temperature_sweep to gather metrics at multiple temperatures.
python evaluate.py -c configs/88m.yaml --checkpoint checkpoints/88m/best.pt \
    --num_samples 100 --temperature_sweep 0.5 0.8 1.0 \
    --output_json results/88m_eval.json

# Perplexity only (skip slow generation)
python evaluate.py -c configs/88m.yaml --checkpoint checkpoints/88m/best.pt --skip_generation

# Stage 5 — analysis (after all training/eval runs are in)
jupyter notebook analysis.ipynb

# Optional — Extended training (for portfolio figure, not the scaling-law experiment).
# Trains the 3m sweet-spot model for 50 epochs (~2.5h on Colab L4).
python train.py -c configs/3m_extended.yaml -o configs/colab.yaml
python generate_constrained.py -c configs/3m_extended.yaml -o configs/colab.yaml \
    --checkpoint checkpoints/3m_extended/best.pt --num_samples 30 \
    --prompt '<svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">' \
    --output_dir samples/extended_constrained
```

`analysis.ipynb` reads `checkpoints/{size}/log.csv`, `checkpoints/{size}/sweep/sweep_results.csv`, and `results/{size}_eval.json`, then produces:

- training curves (SP vs µP, per size)
- LR sweep visualisation (val loss vs LR per size)
- power-law fit `L = a · N^(-α) + c` with 95% CIs (SP and µP)
- µP extrapolation to ~10× the largest fitted size
- sample quality vs model size (XML well-formedness, render rate, perplexity)
- a summary table for inclusion in the report

Figures are written to `report/figures/` for the LaTeX report.

## Key Design Decisions

- **Dataset deviation from spec** — `starvector/svg-stack-simple` instead of
  `svg-icons-simple`. Reason: hits the 100M-token target without
  supplementation, and provides a stricter validity test. See report §2.
- **Vocab 4096 BPE** — SVG has heavy substring repetition; a moderate vocab
  captures common tag/attribute fragments without over-fragmenting the long tail.
- **Coordinate-precision rounding (1 decimal place)** before tokenisation, so
  the BPE doesn't waste merges on near-duplicate numerics.
- **Max sequence length 1024** — balances context against compute. SVGs that
  exceed this are dropped entirely (not truncated) to avoid biasing the
  training distribution.
- **Train/val/test split by file (98/1/1)** — keeps whole SVGs intact across
  splits, required for the XML-validity metric.
- **µP attention scale `1/d`** (not `1/√d`) per Tensor Programs V. Implemented
  by pre-dividing q by `√d` before Flash Attention's internal `1/√d`.
- **1-epoch training** matched across sizes per the project spec — `max_steps`
  is auto-computed at runtime when unset.
- **AdamW** ($\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay 0.1 on 2D params
  only), cosine LR with 5% warmup, gradient clip 1.0.

## Status

- Stage 1 (data pipeline) — implemented; dev-mode verified (1000 samples →
  ~645K train tokens, decoded sample is valid SVG). Full 100M-token run on
  Colab pending.
- Stage 2 (SP scaling) — code complete (`model/transformer.py`, `train.py`,
  `sweep_lr.py`, five per-size configs). Runs and power-law fit pending.
- Stage 3 (µP scaling) — code complete (`mup_train.py`, `mup` flag in
  `GPTConfig`, `set_base_shapes()` ordering enforced). Runs pending.
- Stage 4 — `generate.py` and `evaluate.py` complete. Sample-based runs
  pending (require a trained checkpoint).
- Stage 5 (analysis) — `analysis.ipynb` scaffolded with all plotting and
  power-law fit code. Cells gracefully skip sections whose input files
  don't exist yet, so it can be run incrementally as results land.

## References

- Kaplan et al. 2020. *Scaling Laws for Neural Language Models.*
- Hoffmann et al. 2022. *Training Compute-Optimal Large Language Models* (Chinchilla).
- Yang et al. 2022. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.*
- Rodriguez et al. 2023. *StarVector: Generating Scalable Vector Graphics Code from Images.*
