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

## How it was resolved

Both discrepancies were corrected on 2026-09-22 by re-extracting **all fifteen**
ladder checkpoints in a single software environment, using the checkpoint each
run's `summary.json` names. `runB_s42` moved from `best-31` to `best-39` and
`runB_s62` from `best-28` to `best-39`; the other thirteen kept their file and
changed only by the environment.

Re-extraction had first been judged infeasible, and that judgement was wrong in
two ways worth recording. It needs no write access to the frozen repository —
only read access to the checkpoint — and no training, only a forward pass. What
had actually blocked it was an over-strict acceptance gate: a control re-run of
`runB_s52` was required to reproduce its published numbers to 1e-4 and did not,
so the work stopped. That threshold answered "is this the same environment",
which was not the question. The deviation it detected is about 1.5 units of
condition number against a between-seed standard deviation of 2672 units on the
same rung, three orders of magnitude apart.

On the eleven checkpoints whose file did not change, the largest relative
deviation between the two environments is 9.3e-4 — within the accepted drift
and far from the 1 % that would indicate something structural. The environment
difference itself is unexplained; the leading hypothesis is Pillow's resampling
implementation (9.3.0 here against a much later version in August), since
`Resize` in this pipeline runs on PIL images, where torchvision's `antialias`
argument has no effect.

`baseline_soft` was deliberately **not** re-extracted. It is a single
descriptive reference checkpoint, carries no statistical test, and its 2048-d
embeddings are a 193 MB file; it therefore remains from the August environment,
and that is stated in the manuscript's Limitations.

## Sensitivity of the headline result — as assessed before the correction

*Kept as the record of what informed the decision at the time. It describes the
5/5/3 design and the pre-correction values; for the current state see below.*

The only significant association in the paper was condition number vs.
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

**What the correction actually did.** `runB_s42` went from 8256.8 to 12719.6
and `runB_s62` from 3020.3 to 5905.8 — both up, both still far above 738.8, so
the ladder stayed perfectly monotone. On the 5/5/5 design the headline is now
τ = 0.8452 with exact *p* = 2.6e-6, and Jonckheere–Terpstra reaches J = 75 of
75 cross-rung pairs. The prediction above held.

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
