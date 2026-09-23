#!/usr/bin/env python3
"""
Re-check, from this repository alone, every claim reported to the co-author.

Each check reads a committed artifact and prints PASS/FAIL with the number it
found, so the claims can be audited without trusting a summary. Checks that
need the 208 MB embedding cache (gitignored) or the sibling classifier
repository are marked SKIP rather than silently passing.

    python3 scripts/verify_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSG = ROOT.parent / "CSG-SKin"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool | None, detail: str) -> None:
    results.append((("PASS" if ok else "FAIL") if ok is not None else "SKIP", name, detail))


# ---------------------------------------------------------------- claim (a)
# "The source dataset is ISIC 2019, not ISIC 2018."

split = pd.read_csv(ROOT / "results" / "dataset_split.csv").set_index("quantity")["value"]
tr, va, te = int(split["isic_train"]), int(split["isic_val"]), int(split["isic_test"])
total = int(split["isic_2019_metadata_rows"])
check("(a) split sums to the ISIC 2019 metadata row count",
      tr + va + te == total, f"{tr} + {va} + {te} = {tr+va+te}, metadata = {total}")

check("(a) eight classes, not the seven of ISIC 2018 Task 3",
      int(split["num_classes"]) == 8, f"num_classes = {int(split['num_classes'])}")

if (CSG / "src" / "datasets" / "constants.py").is_file():
    labels = (CSG / "src" / "datasets" / "constants.py").read_text()
    check("(a) label set contains SCC (present in ISIC 2019, absent in ISIC 2018)",
          "SCC" in labels and "AKIEC" not in labels.split("LABELS")[1][:200],
          "src/datasets/constants.py LABELS")
    hits = [p for p in (CSG / "src").rglob("*.py") if "ISIC_2019_Training_Input" in p.read_text()]
    check("(a) code resolves images under ISIC_2019_Training_Input",
          len(hits) > 0, f"{len(hits)} file(s): " + ", ".join(str(p.relative_to(CSG)) for p in hits[:3]))
else:
    check("(a) label set and image path in the classifier repository", None,
          f"sibling repository not found at {CSG}")

body = (ROOT / "paper" / "tmlr" / "common_body.tex").read_text()
check("(a) manuscript says ISIC 2019 and no longer says ISIC 2018",
      "ISIC 2019" in body and "ISIC 2018" not in body,
      f"'ISIC 2019' x{body.count('ISIC 2019')}, 'ISIC 2018' x{body.count('ISIC 2018')}")

bib = (ROOT / "paper" / "tmlr" / "references.bib").read_text()
check("(a) BCN20000 added (ISIC 2019 = BCN20000 + HAM10000 + MSK)",
      "combalia2019bcn20000" in bib and "tschandl2018ham10000" in bib,
      "references.bib")

# ---------------------------------------------------------------- claim (b)
# "Per-image scores rebuilt for all eight scorers; 109/110 reproduce the
#  published AUROC/FPR95 to machine precision."

rep_path = ROOT / "results" / "raw_scores" / "validation_report.csv"
rep = pd.read_csv(rep_path)
n_match = int((rep.status == "khớp").sum())
n_ckpt = rep.run.nunique()
n_scorer = rep.scorer.nunique()
# Counts are derived, not declared: hardcoding 110 and 109 made this script
# fail the moment the ladder grew, which is the opposite of what a check is for.
check("(b) every checkpoint x scorer combination is covered",
      len(rep) == len(rep.drop_duplicates(["run", "scorer"])),
      f"{len(rep)} rows over {n_ckpt} checkpoints and {n_scorer} scorers")
check("(b) every combination reproduces the published AUROC and FPR95",
      n_match == len(rep),
      f"{n_match}/{len(rep)} matched" +
      ("" if n_match == len(rep) else f", {len(rep)-n_match} without a published reference"))
worst = rep.d_auroc.max()
check("(b) largest deviation is at machine precision",
      worst < 1e-6, f"max |delta AUROC| = {worst:.2e}")

npzs = sorted((ROOT / "results" / "raw_scores").glob("*_scores.npz"))
check("(b) one score file per checkpoint",
      len(npzs) == n_ckpt, f"{len(npzs)} .npz files for {n_ckpt} checkpoints")

bad = []
for f in npzs:
    d = np.load(f, allow_pickle=True)
    scorers = sorted({k.split("__")[0] for k in d.files if "__" in k})
    for s in scorers:
        if len(d[f"{s}__id"]) != te or len(d[f"{s}__ood"]) != int(split["padufes_ood"]):
            bad.append(f"{f.name}:{s}")
check(f"(b) every array is {te} ID and {int(split['padufes_ood'])} OOD values",
      not bad, "all arrays correct" if not bad else f"{len(bad)} wrong: {bad[:3]}")

counts = {len({k.split("__")[0] for k in np.load(f, allow_pickle=True).files if "__" in k}) for f in npzs}
check("(b) eight scorers on the ladder, six on the baseline reference",
      counts == {8, 6}, f"scorers per file: {sorted(counts)}")


# ---------------------------------------------------------------- claim (c)
# The two questions a reader asked of this repository directly: is the ladder
# really 5/5/5, and does every row still point at the checkpoint it should?

LADDER = ["runA_grl", "runB_orth1", "runB"]
inventories, paths = {}, {}
for name in ["e1_geometry_metrics.csv", "e2_auroc.csv", "distance_summary.csv",
             "checkpoint_results_matrix.csv"]:
    d = pd.read_csv(ROOT / "results" / name)
    d = d[d.rung.isin(LADDER)]
    inventories[name] = {r: tuple(sorted(g.seed)) for r, g in d.groupby("rung")}
    if "checkpoint_path" in d.columns:
        paths[name] = {(r.rung, r.seed): Path(str(r.checkpoint_path)).name
                       for r in d.itertuples()}

sizes = {n: tuple(len(v) for _, v in sorted(inv.items())) for n, inv in inventories.items()}
uniform = len(set(sizes.values())) == 1
check("(c) every result file reports the same ladder inventory",
      uniform, ", ".join(f"{n.split('.')[0]}={'/'.join(map(str, s))}" for n, s in sizes.items())
      if not uniform else f"all {list(sizes.values())[0]}")
one = list(inventories.values())[0]
check("(c) the ladder is 5/5/5, not 5/5/3",
      all(len(v) == 5 for v in one.values()),
      ", ".join(f"{r}: {len(v)} seeds" for r, v in sorted(one.items())))

ref = paths["e1_geometry_metrics.csv"]
disagree = [k for name, m in paths.items() for k in m if m[k] != ref.get(k)]
check("(c) every file agrees on which checkpoint each row came from",
      not disagree, "consistent" if not disagree else f"{len(set(disagree))} disagree: {sorted(set(disagree))[:3]}")
# Naming an expected epoch here would go stale the next time a run is
# retrained -- it already did. What must hold instead is that the classifier
# head used to rebuild logits came from the same checkpoint as the embeddings:
# a mismatch there silently scores one model's features with another's head,
# which is the failure this repository actually hit once.
import json
heads = {}
for mf in sorted((ROOT / "results").glob("lesion_classifier_heads*_manifest.json")):
    try:
        entries = json.loads(mf.read_text())
    except json.JSONDecodeError:
        continue
    rows = entries if isinstance(entries, list) else entries.get("runs", [])
    for e in rows:
        if isinstance(e, dict) and "run_name" in e and "checkpoint_path" in e:
            heads[e["run_name"]] = Path(str(e["checkpoint_path"])).name
covered = {f"{r}_s{s_}": ref[(r, s_)] for r, s_ in ref}
mismatch = [k for k in covered if k in heads and heads[k] != covered[k]]
check("(c) classifier heads come from the same checkpoints as the embeddings",
      not mismatch and len(heads) >= len(covered),
      f"{len(heads)}/{len(covered)} runs covered by a head manifest"
      + ("" if not mismatch else f", {len(mismatch)} mismatched: {mismatch[:3]}"))

# ---------------------------------------------------------------------- out
w = max(len(n) for _, n, _ in results)
print()
for status, name, detail in results:
    print(f"  [{status}] {name:<{w}}  {detail}")
n_fail = sum(1 for s, _, _ in results if s == "FAIL")
n_skip = sum(1 for s, _, _ in results if s == "SKIP")
print(f"\n{len(results)-n_fail-n_skip} pass, {n_fail} fail, {n_skip} skipped\n")
sys.exit(1 if n_fail else 0)
