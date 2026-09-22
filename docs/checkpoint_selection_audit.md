# Checkpoint selection audit

**Date**: 2026-09-21. **Trigger**: an HPC extraction of the classifier heads
reported that two of the thirteen ladder checkpoints named in
`summary.json["best_checkpoint"]` differ from the ones the cached embeddings
were built from.

## The selection rule, recovered

Neither repository documents which validation metric selects `best-*.ckpt`.
It was recovered empirically from `results/csg_lite/<run>/csv_logs/metrics.csv`:
the rule is **argmax of `val/acc`**, and it reproduces
`summary.json["best_checkpoint"]` for **11 of 13** runs exactly. `val/loss`
minimum matches none of them, so the rule is not loss-based.

## The two discrepancies

| run | embeddings built from | val/acc | rank | `summary.json` names | val/acc | rank |
|---|---|---|---|---|---|---|
| `runB_s42` | `best-31.ckpt` | 0.6722 | 6 / 40 | `best-39.ckpt` | 0.6873 | 1 / 40 |
| `runB_s62` | `best-28.ckpt` | 0.6342 | **18 / 40** | `best-39.ckpt` | 0.6823 | 1 / 40 |

Both are on the `runB` rung (λ_orth = 5), which has only three seeds, so two of
three points on the ladder's top rung come from a non-optimal epoch.

**E1, E2, E2.6 and E2.7 are internally consistent**: all four read the same
cache, so no single data point mixes two checkpoints. They are consistently
wrong together for these two seeds, not inconsistently wrong.

## Why the published results were kept

Re-extraction was attempted and abandoned. The August environment
(`torch 2.11.0+cu130`) could not be rebuilt on the current HPC host, and a
control re-run of `runB_s52` — a seed whose checkpoint is *correct* and needs
no change — failed to reproduce its own published numbers under
`torch 2.1.1`. Mixing two seeds extracted in a new environment with eleven from
the old one would have introduced a second, less tractable inconsistency to fix
the first. The authors chose to report the results as computed and disclose the
discrepancy in the Limitations section.

The control failure is itself unexplained and is *not* numerical noise:
condition number amplifies embedding perturbation by only 3–4× (measured), and
an independent reimplementation of the metric on the same cached embeddings
agrees with the published value to 1.25e-5. The leading hypothesis is a
`torchvision` preprocessing change (`Resize` antialias default), unverified.

## Sensitivity of the headline result

The only significant association in the paper is condition number vs.
λ_orth, τ = 0.8397 — the maximum the 5/5/3 design permits.

- Largest condition number on the two lower rungs: **738.8** (`runB_orth1`).
- The two affected checkpoints currently sit at **8256.8** (11.2×) and
  **3020.3** (4.1×) above it.
- While both stay above 738.8, τ and its exact *p* are unchanged.
- Exact null enumeration over all 72,072 label arrangements: one inverted pair
  gives τ = 0.8092, *p* = 8.3e-5; one checkpoint collapsing below all ten
  lower-rung points gives τ = 0.5344, *p* = 0.022 — still significant. Only if
  **both** collapse entirely (τ = 0.2290, *p* = 0.37) is the finding lost.

Substituting epoch 31 → 39 and 28 → 39 within the same training run does not
plausibly produce a 4–11× drop in condition number.

## Reproducing this audit

```bash
python3 - <<'PY'
import pandas as pd
d = pd.read_csv("<path>/results/csg_lite/runB_s42/csv_logs/metrics.csv")
v = d.dropna(subset=["val/acc"])[["epoch","val/acc"]].groupby("epoch").last()
print(v["val/acc"].idxmax(), v["val/acc"].max())
PY
```

The checkpoint each cached result was built from is recorded in the
`checkpoint_path` field of every `results/e2_distances/*.npz` and in the
`checkpoint_path` column of `e1_geometry_metrics.csv`, `e2_auroc.csv` and
`e2_6_scorer_comparison.csv`.
