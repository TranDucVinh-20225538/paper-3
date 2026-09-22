"""
analyze_norm.py -- E2.5: ID vs OOD feature-norm comparison.

Checks whether ||z_ID|| > ||z_OOD|| -- i.e. whether "OOD sits closer to ID
class centroids" (confirmed by analyze_e2_distances.py, 13/13 checkpoints on
median) is explained by OOD features simply having smaller overall
magnitude (a norm-collapse effect largely independent of direction), as
opposed to a directional/geometric effect. These are different mechanisms
with different implications: a norm collapse would suggest the domain-
adversarial objective is shrinking PAD-UFES activations generally, while a
directional effect without a norm difference would point more specifically
at PAD-UFES landing in a direction the ISIC class means also occupy.

Reads results/e2_distances/*.npz (feature_norm_id/feature_norm_ood, saved by
extract_auroc_e2.py's E2.5 extension -- no rerun needed, this data was
already captured alongside the distance arrays).

Produces:
  - A table: per-checkpoint ID/OOD norm mean and median.
  - Figure: histogram + KDE + ECDF of ||z|| for ID vs OOD, pooled per rung
    (reuses _id_ood_plots, the same plotting code analyze_e2_distances.py
    uses, not a second copy of it).
  - A boxplot, same pattern.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_e1 import PRIMARY_RUNGS, RUNG_LAMBDA
from _id_ood_plots import make_boxplot, make_distribution_figure


def load_norms(npz_dir: Path, rung: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_dir / f"{rung}_s{seed}.npz")
    return data["feature_norm_id"], data["feature_norm_ood"]


def discover_checkpoints(npz_dir: Path) -> pd.DataFrame:
    """Enumerates available (rung, seed) pairs directly from the npz filenames,
    rather than depending on distance_summary.csv also being present."""
    rows = []
    for rung in PRIMARY_RUNGS:
        for path in sorted(npz_dir.glob(f"{rung}_s*.npz")):
            seed = int(path.stem.split("_s")[-1])
            rows.append({"rung": rung, "seed": seed})
    return pd.DataFrame(rows)


def per_checkpoint_table(npz_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in manifest.iterrows():
        norm_id, norm_ood = load_norms(npz_dir, r["rung"], r["seed"])
        rows.append({
            "rung": r["rung"], "seed": r["seed"],
            "id_n": len(norm_id), "id_norm_mean": float(norm_id.mean()), "id_norm_median": float(np.median(norm_id)),
            "ood_n": len(norm_ood), "ood_norm_mean": float(norm_ood.mean()), "ood_norm_median": float(np.median(norm_ood)),
        })
    return pd.DataFrame(rows)


def pooled_norms_by_rung(npz_dir: Path, manifest: pd.DataFrame) -> dict:
    pooled = {}
    for rung in PRIMARY_RUNGS:
        seeds = manifest[manifest["rung"] == rung]["seed"].tolist()
        all_id, all_ood = [], []
        for seed in seeds:
            norm_id, norm_ood = load_norms(npz_dir, rung, seed)
            all_id.append(norm_id)
            all_ood.append(norm_ood)
        pooled[rung] = (np.concatenate(all_id), np.concatenate(all_ood))
    return pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    parser.add_argument("--npz_dir", default=str(results_dir / "e2_distances"))
    parser.add_argument("--output_dir", default=str(results_dir))
    parser.add_argument("--figures_dir", default=str(Path(__file__).resolve().parents[1] / "figures"))
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    manifest = discover_checkpoints(npz_dir)
    if len(manifest) != 13:
        print(f"[analyze_norm] WARNING: found {len(manifest)} checkpoints in {npz_dir}, expected 13 "
              "(5 runA_grl + 5 runB_orth1 + 3 runB). Proceeding with what's available.")

    table = per_checkpoint_table(npz_dir, manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "norm_summary.csv", index=False)

    print("=== Table: ||z|| (feature norm) per checkpoint ===")
    print(table.sort_values(["rung", "seed"]).to_string(index=False))

    pooled = pooled_norms_by_rung(npz_dir, manifest)

    print("\n=== Per-rung verdict: does ||z_ID|| > ||z_OOD||? (checked directly) ===")
    all_id_above_ood = True
    for rung in PRIMARY_RUNGS:
        norm_id, norm_ood = pooled[rung]
        id_med, ood_med = float(np.median(norm_id)), float(np.median(norm_ood))
        id_mean, ood_mean = float(norm_id.mean()), float(norm_ood.mean())
        holds = id_med > ood_med
        all_id_above_ood &= holds
        verdict = "||z_ID|| > ||z_OOD|| (consistent with a norm-collapse contribution)" if holds \
            else "||z_ID|| <= ||z_OOD|| -- norm collapse is NOT the (sole) explanation here"
        print(f"  {rung:12s} (pooled n_id={len(norm_id)}, n_ood={len(norm_ood)}): "
              f"median: ID={id_med:.3f} OOD={ood_med:.3f}  |  mean: ID={id_mean:.3f} OOD={ood_mean:.3f}  ->  {verdict}")

    print(
        f"\nOverall: {'every rung shows ||z_ID|| > ||z_OOD|| -- consistent with norm collapse contributing to the distance gap' if all_id_above_ood else 'NOT every rung shows ||z_ID|| > ||z_OOD|| -- if the distance gap (analyze_e2_distances.py) holds while norm does not move the same way, the effect is more directional than a simple scale/collapse story'}"
    )
    print(
        "\nNote: a norm difference does not automatically explain the Mahalanobis distance gap, since "
        "Mahalanobis distance depends on direction relative to each class mean through the precision "
        "matrix, not on ||z|| alone -- this table establishes whether norm moves in the same direction "
        "as the distance gap, not that it causes it."
    )

    make_distribution_figure(pooled, Path(args.figures_dir), "||z|| (feature norm)", "figure_norm_distributions", RUNG_LAMBDA)
    make_boxplot(pooled, Path(args.figures_dir), "||z|| (feature norm)", "figure_norm_boxplot")
    print(f"\n[analyze_norm] Figures written to {args.figures_dir}/figure_norm_distributions.{{png,pdf}} "
          f"and figure_norm_boxplot.{{png,pdf}}")
    print(f"[analyze_norm] Table written to {out_dir}/norm_summary.csv")


if __name__ == "__main__":
    main()
