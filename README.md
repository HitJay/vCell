# vCell — Virtual Cell Perturbation Model

`vCell` is a PyTorch project for **virtual-cell modelling**: learning a latent
representation of single-cell state and predicting how cells respond to
**perturbations** (gene knockouts/activations, compounds, cytokines, …).

The core model is a **latent-additive conditional VAE** (in the spirit of
scGEN / CPA): an encoder maps an expression profile to a *basal* latent state,
each perturbation is a learned vector added in latent space, and a decoder
reconstructs the perturbed profile. Once trained, the model can answer
counterfactual questions — *"what would these control cells look like under
perturbation k?"* — which is exactly the **virtual cell** use case.

```
control cell ──encode──► z_basal ──(+ perturbation vector p_k·dose)──► z ──decode──► predicted perturbed profile
```

## Features

- Latent-additive conditional VAE (`vcell.models.VirtualCell`).
- Self-contained **synthetic perturbation dataset** generator, so the whole
  pipeline is runnable end-to-end with zero external data.
- Optional `.npz` / `AnnData` (`.h5ad`) loading for real single-cell data.
- Trainer with early stopping, checkpointing and counterfactual evaluation
  (per-perturbation Δ-expression Pearson / R²).
- Typed, YAML-backed config system and a small CLI (`vcell gen-data|train|eval`).
- Pytest suite covering data, model, metrics and a training smoke test.

## Project layout

```
vCell/
├── configs/default.yaml        # default experiment config
├── src/vcell/
│   ├── data/                   # synthetic generator + PerturbationDataset
│   ├── models/                 # encoder / decoder / perturbation / VirtualCell
│   ├── train/                  # losses + Trainer
│   ├── utils/                  # config, seeding, metrics
│   └── cli.py                  # `vcell` command-line entry point
├── scripts/                    # thin wrappers (generate data / train)
├── tests/                      # pytest suite
├── requirements.txt
└── pyproject.toml
```

## Install

```bash
# from the vCell/ directory
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + dev tools (pytest)
# optional single-cell IO:
pip install -e ".[scrna]"        # anndata / scanpy
```

> Requires Python ≥ 3.10 and PyTorch ≥ 2.0 (CPU is fine for the synthetic demo).

## Quickstart (synthetic, ~1 min on CPU)

```bash
# 1) generate a synthetic perturbation dataset
vcell gen-data --out data/synthetic.npz --n-genes 200 --num-perturbations 16

# 2) train (writes checkpoints + config to runs/demo/)
vcell train --config configs/default.yaml \
    --set data.data_path=data/synthetic.npz train.out_dir=runs/demo train.epochs=30

# 3) evaluate counterfactual predictions on the held-out split
vcell eval --run runs/demo
```

Equivalent script form:

```bash
python scripts/generate_synthetic_data.py --out data/synthetic.npz
python scripts/train.py --config configs/default.yaml
```

## Using your own data

Provide an `.npz` with arrays `X` (cells × genes, float32), `pert`
(int64 perturbation id, `0` = control) and optional `dose` (float32):

```python
import numpy as np
np.savez("data/mine.npz", X=X, pert=pert, dose=dose)
```

…then point `data.data_path` at it. `.h5ad` files are supported when `anndata`
is installed (perturbation column configurable via `data.pert_key`).

## Tests

```bash
pytest -q
```

## Notes & roadmap

The default model uses a Gaussian (MSE) reconstruction suitable for
log-normalised expression. Natural extensions: negative-binomial decoder for
raw counts, adversarial basal disentanglement (full CPA), gene-program
perturbation embeddings for zero-shot generalisation to unseen perturbations,
and a transformer cell encoder. These are intentionally left as clean
extension points rather than baked in.
