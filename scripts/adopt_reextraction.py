#!/usr/bin/env python3
"""
Adopt a full re-extraction of the ladder, replacing the published rows.

This is the opposite operation to merge_new_seeds.py. That one appends new
checkpoints and refuses when a (rung, seed) already exists; this one takes a
re-extraction of checkpoints that are already present and replaces them, which
is what a corrected checkpoint choice or a change of software environment
requires.

What it protects:
  - baseline_soft is not part of the ladder and is not in the re-extraction, so
    its rows and its embeddings must survive untouched.
  - the checkpoint each row was built from is compared before and after, and
    every change is printed, so a silent substitution is impossible.
  - everything replaced is backed up first.

    python3 scripts/adopt_reextraction.py <staging_dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BACKUP = RESULTS / "_pre_reextraction_backup"
LADDER = ["runA_grl", "runB_orth1", "runB"]
CSVS = ["e1_geometry_metrics.csv", "e2_auroc.csv", "distance_summary.csv",
        "e2_6_scorer_comparison.csv"]


def epoch_of(path: str) -> str:
    m = re.search(r"(best-\d+)\.ckpt", str(path))
    return m.group(1) if m else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("staging", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stage = args.staging
    if not (stage / "e1_geometry_metrics.csv").is_file():
        sys.exit(f"DỪNG: không thấy e1_geometry_metrics.csv trong {stage}")

    # --- which checkpoint does each row come from, before and after ---------
    old = pd.read_csv(RESULTS / "e1_geometry_metrics.csv")
    new = pd.read_csv(stage / "e1_geometry_metrics.csv")
    if sorted(new.rung.unique()) != sorted(LADDER):
        sys.exit(f"DỪNG: bản trích lại chứa rung lạ: {sorted(new.rung.unique())}")
    o = {(r.rung, r.seed): epoch_of(r.checkpoint_path) for r in old.itertuples()}
    n = {(r.rung, r.seed): epoch_of(r.checkpoint_path) for r in new.itertuples()}
    if set(n) != set(o):
        sys.exit(f"DỪNG: tập checkpoint khác nhau. thiếu={sorted(set(o)-set(n))} thừa={sorted(set(n)-set(o))}")
    changed = {k: (o[k], n[k]) for k in n if o[k] != n[k]}
    print(f"{len(n)} checkpoint được thay. Đổi file checkpoint ở {len(changed)}:")
    for (rung, seed), (a, b) in sorted(changed.items()):
        print(f"    {rung}_s{seed}: {a} -> {b}")
    if not changed:
        print("    (không có — chỉ đổi môi trường trích)")

    plan = []
    outs: dict[str, pd.DataFrame] = {}
    for name in CSVS:
        cur, nw = pd.read_csv(RESULTS / name), pd.read_csv(stage / name)
        keys = set(zip(nw.rung, nw.seed))
        kept = cur[[k not in keys for k in zip(cur.rung, cur.seed)]]
        outs[name] = pd.concat([kept, nw], ignore_index=True)
        n_base = (kept.rung == "baseline_soft").sum()
        plan.append(f"  {name:<32} giữ {len(kept):>4} (baseline_soft {n_base}) + mới {len(nw):>4} = {len(outs[name]):>4}")

    npz = sorted((stage / "e2_distances").glob("*.npz"))
    plan.append(f"  e2_distances/                    thay {len(npz)} file, giữ baseline_soft")
    print("\n".join(plan))

    if args.dry_run:
        print("\n(dry-run, chưa ghi gì)")
        return

    BACKUP.mkdir(parents=True, exist_ok=True)
    for name in CSVS:
        shutil.copy2(RESULTS / name, BACKUP / name)
        outs[name].to_csv(RESULTS / name, index=False)
    (BACKUP / "e2_distances").mkdir(exist_ok=True)
    for f in npz:
        dst = RESULTS / "e2_distances" / f.name
        if dst.is_file():
            shutil.copy2(dst, BACKUP / "e2_distances" / f.name)
        shutil.copy2(f, dst)

    left = pd.read_csv(RESULTS / "e2_6_scorer_comparison.csv")
    if not (left.rung == "baseline_soft").any():
        sys.exit("DỪNG: baseline_soft biến mất sau khi thay -- khôi phục từ _pre_reextraction_backup/")
    print(f"\nĐã thay. Bản trước khi thay ở {BACKUP.relative_to(ROOT)}/")
    print("Chạy tiếp: analyze_e1, analyze_e2, analyze_e2_7_domain_probe, dump_raw_scores,")
    print("           analyze_e2_6, analyze_power_and_ci, make_paper_figures, make_checkpoint_matrix")


if __name__ == "__main__":
    main()
