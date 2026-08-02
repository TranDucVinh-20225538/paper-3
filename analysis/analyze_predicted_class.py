"""
analyze_predicted_class.py -- E2.5: which ISIC class do OOD (PAD-UFES) samples
get assigned to by nearest-centroid Mahalanobis distance?

analyze_e2_distances.py established that OOD samples sit closer to their
nearest ISIC class centroid than ID samples do, on the whole (median gap).
This script asks a sharper, more diagnostic question: is that gap spread
evenly across all 8 ISIC classes, or does it come from OOD samples
collapsing onto one or two "attractor" classes? A single dominant attractor
(e.g. 80% of PAD-UFES landing in Nevus, the majority ISIC class) would point
toward a specific, more mundane explanation -- OOD features sitting in a
generic/majority-class region of representation space -- rather than a
diffuse geometric effect touching all classes equally.

Reads results/e2_distances/*.npz (predicted_class_id/predicted_class_ood,
labels_id/labels_ood -- already captured by extract_auroc_e2.py's E2.5
extension, no rerun needed).

Produces:
  - Per-checkpoint table: % of OOD samples predicted into each of the 8
    ISIC classes (MEL/NV/BCC/AK/BKL/DF/VASC/SCC).
  - The same table for ID samples' predicted class vs their TRUE label, as a
    reference point -- ID samples should predominantly predict their own
    true class (that's what makes them ID), so this also serves as a sanity
    check that predicted_class is wired correctly.
  - A grouped bar chart, one group per rung, showing the OOD predicted-class
    distribution pooled across seeds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_e1 import PRIMARY_RUNGS, RUNG_LAMBDA
from analyze_norm import discover_checkpoints

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _repo_paths import find_csg_skin_root  # noqa: E402

_csg_root = find_csg_skin_root(__file__)
sys.path.insert(0, str(_csg_root))
from src.datasets.constants import LABELS  # noqa: E402


def load_class_data(npz_dir: Path, rung: str, seed: int) -> dict:
    data = np.load(npz_dir / f"{rung}_s{seed}.npz")
    return {
        "predicted_class_id": data["predicted_class_id"],
        "predicted_class_ood": data["predicted_class_ood"],
        "labels_id": data["labels_id"],
        "labels_ood": data["labels_ood"] if "labels_ood" in data else None,
    }


def class_distribution_pct(predicted: np.ndarray, n_classes: int = 8) -> np.ndarray:
    counts = np.bincount(predicted, minlength=n_classes)[:n_classes]
    return 100.0 * counts / len(predicted)


def per_checkpoint_table(npz_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in manifest.iterrows():
        d = load_class_data(npz_dir, r["rung"], r["seed"])
        ood_pct = class_distribution_pct(d["predicted_class_ood"])
        id_pct = class_distribution_pct(d["predicted_class_id"])
        row = {"rung": r["rung"], "seed": r["seed"], "n_ood": len(d["predicted_class_ood"]), "n_id": len(d["predicted_class_id"])}
        for i, name in enumerate(LABELS):
            row[f"ood_pct_{name}"] = ood_pct[i]
        for i, name in enumerate(LABELS):
            row[f"id_pct_{name}"] = id_pct[i]
        row["id_predicted_own_class_pct"] = 100.0 * float(np.mean(d["predicted_class_id"] == d["labels_id"]))
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_ood_distribution_by_rung(npz_dir: Path, manifest: pd.DataFrame) -> dict:
    pooled = {}
    for rung in PRIMARY_RUNGS:
        seeds = manifest[manifest["rung"] == rung]["seed"].tolist()
        all_ood = np.concatenate([load_class_data(npz_dir, rung, s)["predicted_class_ood"] for s in seeds])
        pooled[rung] = class_distribution_pct(all_ood)
    return pooled


def make_bar_chart(pooled: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    n_classes = len(LABELS)
    x = np.arange(n_classes)
    width = 0.8 / len(pooled)
    for i, (rung, pct) in enumerate(pooled.items()):
        ax.bar(x + i * width, pct, width, label=f"{rung} ($\\lambda$={RUNG_LAMBDA[rung]:g})")
    ax.set_xticks(x + width * (len(pooled) - 1) / 2)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("% of OOD (PAD-UFES) samples")
    ax.set_title("Nearest-centroid predicted class for OOD samples\n(pooled across seeds per rung)")
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_predicted_class_ood.png", dpi=200)
    fig.savefig(out_dir / "figure_predicted_class_ood.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    parser.add_argument("--npz_dir", default=str(results_dir / "e2_distances"))
    parser.add_argument("--output_dir", default=str(results_dir))
    parser.add_argument("--figures_dir", default=str(Path(__file__).resolve().parents[1] / "figures"))
    parser.add_argument("--dominance_threshold", type=float, default=50.0,
                         help="Flag a class as a 'dominant attractor' if it exceeds this %% of OOD predictions.")
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    manifest = discover_checkpoints(npz_dir)
    if len(manifest) != 13:
        print(f"[analyze_predicted_class] WARNING: found {len(manifest)} checkpoints in {npz_dir}, expected 13. "
              "Proceeding with what's available.")

    table = per_checkpoint_table(npz_dir, manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "predicted_class_summary.csv", index=False)

    print("=== Sanity check: ID samples predicted into their OWN true class ===")
    print(table[["rung", "seed", "id_predicted_own_class_pct"]].sort_values(["rung", "seed"]).to_string(index=False))
    min_own = table["id_predicted_own_class_pct"].min()
    print(f"(min across checkpoints: {min_own:.1f}% -- if this were near chance (~12.5% for 8 classes) rather than "
          f"clearly above it, predicted_class would be suspect; {min_own:.1f}% is well above chance.)")

    pooled = pooled_ood_distribution_by_rung(npz_dir, manifest)

    print(f"\n=== OOD (PAD-UFES) predicted-class distribution, pooled per rung (%) ===")
    dist_df = pd.DataFrame(pooled, index=LABELS).T
    print(dist_df.round(1).to_string())

    print(f"\n=== Dominant-attractor check (threshold={args.dominance_threshold:.0f}%) ===")
    any_dominant = False
    for rung in PRIMARY_RUNGS:
        pct = pooled[rung]
        top_idx = int(np.argmax(pct))
        top_name, top_pct = LABELS[top_idx], pct[top_idx]
        if top_pct >= args.dominance_threshold:
            any_dominant = True
            print(f"  {rung:12s}: DOMINANT attractor -- {top_pct:.1f}% of OOD predicted as {top_name}")
        else:
            print(f"  {rung:12s}: no dominant attractor -- top class is {top_name} at {top_pct:.1f}% "
                  f"(below {args.dominance_threshold:.0f}% threshold)")
    if not any_dominant:
        print(f"\nNo rung shows a single class exceeding {args.dominance_threshold:.0f}% of OOD predictions -- "
              "the effect (if real) is not explained by simple collapse onto one majority attractor class.")

    make_bar_chart(pooled, Path(args.figures_dir))
    print(f"\n[analyze_predicted_class] Figure written to {args.figures_dir}/figure_predicted_class_ood.{{png,pdf}}")
    print(f"[analyze_predicted_class] Table written to {out_dir}/predicted_class_summary.csv")


if __name__ == "__main__":
    main()
