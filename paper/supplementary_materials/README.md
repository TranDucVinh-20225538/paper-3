# Supplementary materials

*Decodable but Directionally Misaligned: Auditing Distance-Based OOD Ranking on
Domain-Adversarial Skin-Lesion Representations*

This package contains the experiment code, the analysis code, the figure code,
and the derived results, in that order of importance.

```
scripts/     experiment pipeline: checkpoint -> embeddings -> geometry, scores
analysis/    statistics reported in the paper, over scripts/ output
figures/     plotting and table generation only
results/     derived CSVs produced by scripts/ and analysis/
```

## The pipeline, stage by stage

Each stage names the paper section it implements and the file it writes. Every
script takes an **explicit checkpoint file path**; none resolves a checkpoint by
directory search, which is a deliberate design constraint (Section 3.1).

| # | Stage | File | Paper section | Output |
|---|---|---|---|---|
| 1 | Extract $z_{\text{lesion}}$ embeddings from one checkpoint and compute the three geometry metrics on the fitted Mahalanobis parameters | `scripts/extract_embeddings_e1.py` | 3.1, 3.2 | `results/e1_geometry_metrics.csv` |
| 1a | Geometry metric library used by stage 1 (condition number, Fisher ratio, Mardia's kurtosis) — pure numpy over already-extracted arrays | `scripts/geometry_diagnostics.py` | 3.2, App. A.1 | — |
| 2 | Fit the Mahalanobis estimator on ISIC-train and score ISIC-test vs. PAD-UFES; AUROC and FPR@95%TPR for one checkpoint | `scripts/extract_auroc_e2.py` | 3.3, 4.2 | `results/e2_auroc.csv`, cached embeddings |
| 3 | Cosine-to-centroid and pooled *k*-NN (*k* = 1, 10, 50) scorers over the cached embeddings | `scripts/extract_e2_8_extra_scorers.py` | 3.3, 4.4 | `results/e2_6_scorer_comparison.csv` |
| 4 | Ladder trend tests for the geometry metrics (exact-permutation Kendall's τ and Jonckheere–Terpstra) | `analysis/analyze_e1.py` | 3.5, 4.1 | `results/e1_kendall_tau.csv`, `e1_jonckheere_terpstra.csv` |
| 5 | Geometry vs. Mahalanobis AUROC association | `analysis/analyze_e2.py` | 4.2 | `results/e2_kendall_tau.csv`, `e2_merged.csv` |
| 6 | Five-scorer comparison and its ladder trend tests | `analysis/analyze_e2_6.py` | 4.4 | `results/e2_6_kendall_tau.csv`, `table_e2_6_scorer_summary.csv` |
| 7 | Domain probes (logistic regression, linear SVM, random forest), 5-fold stratified CV on out-of-fold predictions | `analysis/analyze_e2_7_domain_probe.py` | 3.4, 4.5 | `results/e2_7_domain_probe.csv`, `e2_7_kendall_tau.csv` |
| 8 | Distance distributions, feature norms, and majority-class attraction | `analysis/analyze_e2_distances.py`, `analyze_norm.py`, `analyze_nv_attractor.py`, `analyze_predicted_class.py` | 4.3, App. B.1 | `results/distance_summary.csv`, `norm_summary.csv`, `nv_attractor_headline.csv`, `predicted_class_summary.csv` |
| 9 | Detectability, power, and BCa bootstrap intervals | `analysis/analyze_power_and_ci.py` | 3.6, App. C | `results/power_design_summary.csv`, `power_curve.csv`, `kendall_tau_ci.csv`, `jt_pvalue_conventions.csv` |

| 10 | Per-image scores for all eight scorers, rebuilt from the cached embeddings and each one validated against the aggregated AUROC/FPR95 above before being written | `analysis/dump_raw_scores.py` | 3.3, 4.4 | `results/raw_scores/*.npz`, `validation_report.csv` |

`results/raw_scores/` holds one `.npz` per checkpoint with `<scorer>__id` and
`<scorer>__ood` arrays (5,067 and 2,298 values) for Mahalanobis,
cosine-to-centroid, *k*-NN at *k* = 1/10/50, per-class KDE, Energy and ViM.
The pipeline originally collapsed each of these arrays to a single AUROC row
and discarded it; this script rebuilds them by importing the same scoring
functions the original scripts used, and refuses to write anything whose
recomputed AUROC and FPR95 do not match the published value. Energy and ViM
additionally need the classifier head, supplied via `--heads`; the head must
come from the same checkpoint as the cached embeddings, which for two seeds is
not the one the training summary names (see the checkpoint-selection note in
the paper's Limitations).

`figures/` holds only presentation code: `make_figure_power_tmlr.py` and
`make_tables_power_tmlr.py` draw Appendix C's figure and emit its two tables
from the CSVs above; `make_figure_dumbbell.py` and `_id_ood_plots.py` draw the
remaining figures. Nothing in `figures/` computes a reported quantity.

## What needs what

Stages 1–3 load trained checkpoints through the audited classifier's own model
code, so they require `torch`, that classifier's repository on the import path
(its root is located by marker file, or by the `CLASSIFIER_ROOT` environment
variable — see `scripts/_repo_paths.py`), the ISIC 2018 and PAD-UFES-20 image
data, and the 13 checkpoints. Checkpoints and the 349 MB of intermediate
embeddings are too large to include here and are released with the archival
version; the classifier repository is withheld during review because its name
identifies the authors.

Stages 4–9 and everything in `figures/` run from the CSVs in `results/` with no
GPU, no checkpoints, and no dataset download:

```bash
pip install -r requirements.txt
python3 analysis/analyze_power_and_ci.py --n-boot 20000 --n-sim 20000 --seed 0
```

That command reproduces every number in Appendix C exactly.
`analysis/seed_manifest.json` records the configuration, including the two RNG
streams the single `--seed` derives.

## The checkpoint-level matrix

`results/checkpoint_results_matrix.csv` is the primary data table: one row per
checkpoint, 13 rows, every quantity the paper analyses.

| column | meaning |
|---|---|
| `checkpoint_id`, `rung`, `lambda_orth`, `seed` | which checkpoint |
| `condition_number`, `fisher_hl`, `fisher_scalar`, `mardia_b`, `mardia_z` | geometry (Section 3.2) |
| `mahalanobis_auroc`, `cosine_auroc`, `knn_{1,10,50}_auroc` | directed OOD AUROC (Section 3.3) |
| `probe_{lr,svm,rf}_auroc` | domain-probe AUROC (Section 3.4) |

It regenerates the summary statistics in the paper exactly: condition number
75.5 ± 7.8, 559.8 ± 125.9, 5329.9 ± 2672.2 across the three rungs; pooled
directed AUROC 0.402 ± 0.024 (Mahalanobis) and 0.418 ± 0.022 (cosine).

One aggregation note, so the matrix is not misread: the paper's "0.72–0.81
AUROC" for the probes is the range over the nine rung-by-probe means. The 39
individual checkpoint-level values in this matrix span 0.69–0.84.

## Independent check on the power analysis

`figures/reproduce_power_figure.py` recomputes the three power curves from a
separate implementation, as a check that the result does not depend on one
codebase. It agrees with the published curves to within **0.009** in power
across the grid, ordinary Monte Carlo variation at 2×10⁴ simulations per point
(standard error ≈ 0.0035). It is not bit-identical and is not meant to be: the
two implementations draw random numbers in a different order.

Both use the same alternative: `y_i = θ·d_i + ε_i` with `ε ~ N(0,1)` and `d` the
**ordinal rung index 0, 1, 2** — not the λ_orth values 0, 1, 5. The statistic is
rank-based, so the coding does not change any τ, but it does change the shape of
the alternative and therefore the power curve.
