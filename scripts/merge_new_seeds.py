#!/usr/bin/env python3
"""
Merge the runB seeds 72 and 82 produced on the cluster into results/.

The cluster writes a `results_new/` tree holding only the new checkpoints.
Copying that tree over `results/` replaces the published files instead of
extending them -- a two-row e1_geometry_metrics.csv in place of the thirteen-row
one. This script appends instead, refusing to run if a (rung, seed) it is about
to add is already present, and backing up every file it touches first.

    python3 scripts/merge_new_seeds.py <staging_dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BACKUP = RESULTS / "_pre_merge_backup"

CSVS = {
    "e1_geometry_metrics.csv": ["rung", "seed"],
    "e2_auroc.csv": ["rung", "seed"],
    "distance_summary.csv": ["rung", "seed"],
    "e2_6_scorer_comparison.csv": ["rung", "seed", "scorer"],
}
NPZ = ["runB_s72_z.npz", "runB_s82_z.npz", "runB_s72.npz", "runB_s82.npz"]
HEADS = ["lesion_classifier_heads_new.npz", "lesion_classifier_heads_new_manifest.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("staging", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan: list[str] = []
    merged: dict[str, pd.DataFrame] = {}

    for name, key in CSVS.items():
        cur, new = pd.read_csv(RESULTS / name), pd.read_csv(args.staging / name)
        if list(cur.columns) != list(new.columns):
            sys.exit(f"DỪNG: {name} khác cột giữa bản cũ và bản mới.")
        clash = set(map(tuple, new[key].values)) & set(map(tuple, cur[key].values))
        if clash:
            sys.exit(f"DỪNG: {name} đã có sẵn {sorted(clash)[:5]} -- không gộp đè.")
        out = pd.concat([cur, new], ignore_index=True)
        merged[name] = out
        plan.append(f"  {name:<32} {len(cur):>4} + {len(new):>3} = {len(out):>4} dòng")

    for f in NPZ:
        plan.append(f"  {f:<32} -> results/e2_distances/")
    for f in HEADS:
        plan.append(f"  {f:<32} -> results/")

    print("\n".join(plan))
    if args.dry_run:
        print("\n(dry-run, chưa ghi gì)")
        return

    BACKUP.mkdir(parents=True, exist_ok=True)
    for name in CSVS:
        shutil.copy2(RESULTS / name, BACKUP / name)
        merged[name].to_csv(RESULTS / name, index=False)
    (RESULTS / "e2_distances").mkdir(exist_ok=True)
    for f in NPZ:
        shutil.copy2(args.staging / f, RESULTS / "e2_distances" / f)
    for f in HEADS:
        shutil.copy2(args.staging / f, RESULTS / f)

    print(f"\nĐã gộp. Bản trước khi gộp lưu ở {BACKUP.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
