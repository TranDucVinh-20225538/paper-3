#!/usr/bin/env python3
"""
Rebuild results/checkpoint_results_matrix.csv -- one row per checkpoint.

This is the table the supplementary README calls the primary data table, and it
had no generator: it was assembled once by hand, so when the ladder grew from
13 checkpoints to 15 every other result file was extended and this one silently
stayed at 13, still reporting runB with three seeds. Same failure mode as
manuscript Figures 2-5.

Every column is joined from a file that does have a generator, and the
checkpoint each row came from is carried through so the provenance is on the
row rather than implied.

    python3 analysis/make_checkpoint_matrix.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUNGS = ["runA_grl", "runB_orth1", "runB"]
LAMBDA = {"runA_grl": 0, "runB_orth1": 1, "runB": 5}
GEO = {"condition_number": "condition_number", "fisher_ratio_HL": "fisher_hl",
       "fisher_ratio_scalar": "fisher_scalar", "mardia_kurtosis_b": "mardia_b",
       "mardia_kurtosis_z": "mardia_z"}
SCORERS = {"mahalanobis": "mahalanobis_auroc", "cosine": "cosine_auroc",
           "knn_k1": "knn_1_auroc", "knn_k10": "knn_10_auroc", "knn_k50": "knn_50_auroc"}
PROBES = {"logistic_regression": "probe_lr_auroc", "linear_svm": "probe_svm_auroc",
          "random_forest": "probe_rf_auroc"}


def main() -> None:
    g = pd.read_csv(RESULTS / "e1_geometry_metrics.csv")
    g = g[g.rung.isin(RUNGS)][["rung", "seed", "checkpoint_path", *GEO]].rename(columns=GEO)

    sc = pd.read_csv(RESULTS / "e2_6_scorer_comparison.csv")
    sc = sc[sc.rung.isin(RUNGS) & sc.scorer.isin(SCORERS)]
    sc = sc.pivot_table(index=["rung", "seed"], columns="scorer", values="auroc").rename(columns=SCORERS)

    pr = pd.read_csv(RESULTS / "e2_7_domain_probe.csv")
    pr = pr[pr.rung.isin(RUNGS)]
    pr = pr.pivot_table(index=["rung", "seed"], columns="probe", values="domain_auroc").rename(columns=PROBES)

    m = g.merge(sc.reset_index(), on=["rung", "seed"]).merge(pr.reset_index(), on=["rung", "seed"])
    if len(m) != len(g):
        raise SystemExit(f"DỪNG: {len(g)} checkpoint ở E1 nhưng chỉ {len(m)} qua được merge.")

    m["checkpoint_id"] = m.rung + "_s" + m.seed.astype(str)
    m["lambda_orth"] = m.rung.map(LAMBDA)
    cols = (["checkpoint_id", "rung", "lambda_orth", "seed", *GEO.values(),
             *SCORERS.values(), *PROBES.values(), "checkpoint_path"])
    m = m[cols].sort_values(["lambda_orth", "seed"]).reset_index(drop=True)
    m.to_csv(RESULTS / "checkpoint_results_matrix.csv", index=False)

    print(f"{len(m)} checkpoint:")
    for r in RUNGS:
        print(f"  {r:<12} seeds {sorted(m[m.rung == r].seed.tolist())}")


if __name__ == "__main__":
    main()
