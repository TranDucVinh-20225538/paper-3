#!/usr/bin/env python3
"""
Build paper/supplementary_materials/ (and its zip) from the working repo.

The supplement is a *derived* artifact. The working copies under scripts/ and
analysis/ keep their real provenance -- upstream repository names, the exact
upstream file paths they mirror, the internal design docs they cite -- because
that provenance is what makes them auditable. This script strips the parts of
it that would deanonymise a double-blind submission, and nothing else.

What gets stripped and why: the upstream classifier's repository is public
under an account that identifies the authors, so its name, the environment
variables named after it, and the distinctive upstream file path the scripts
mirror are all searchable. They are replaced with neutral equivalents.

What does NOT get stripped, and cannot be: these scripts import from that
upstream repository (`src.models.csg_lightning`, `CSGLiteLightning`,
`--method csg`, `checkpoints/csg_lite/`). Renaming those would break the code
and misrepresent what was run. A determined reviewer can still follow them
upstream. This script removes the casual path to the authors, not every path.

    python3 scripts/build_supplement.py [--no-zip]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "supplementary_materials"
ZIP = ROOT / "paper" / "supplementary_materials.zip"

EXPERIMENT = [
    "scripts/_repo_paths.py",
    "scripts/extract_embeddings_e1.py",
    "scripts/geometry_diagnostics.py",
    "scripts/extract_auroc_e2.py",
    "scripts/extract_e2_8_extra_scorers.py",
]
ANALYSIS = [
    "analysis/analyze_e1.py",
    "analysis/analyze_e2.py",
    "analysis/analyze_e2_6.py",
    "analysis/analyze_e2_7_domain_probe.py",
    "analysis/analyze_e2_distances.py",
    "analysis/analyze_norm.py",
    "analysis/analyze_nv_attractor.py",
    "analysis/analyze_predicted_class.py",
    "analysis/analyze_power_and_ci.py",
    "analysis/dump_raw_scores.py",
]
FIGURES = [
    "analysis/make_figure_power_tmlr.py",
    "analysis/make_tables_power_tmlr.py",
    "analysis/make_figure_dumbbell.py",
    "analysis/_id_ood_plots.py",
]

# Applied in order. Longest/most specific first, so a later rule cannot
# re-match text an earlier one already neutralised.
SUBSTITUTIONS: list[tuple[str, str]] = [
    (r"cbm_revision/scripts/eval_ood_benchmarks\.py", "the canonical OOD evaluation script"),
    (r"cbm_revision/scripts/", "the upstream scripts directory, "),
    (r"eval_ood_benchmarks\.py", "the canonical OOD evaluation script"),
    (r"CSG-SKin's", "the audited classifier repository's"),
    (r"CSG-SKin", "the audited classifier repository"),
    (r"DST-Skin's", "the second classifier repository's"),
    (r"DST-Skin", "the second classifier repository"),
    (r"CSG_SKIN_ROOT", "CLASSIFIER_ROOT"),
    (r"DST_SKIN_ROOT", "SECOND_CLASSIFIER_ROOT"),
    (r"find_csg_skin_root", "find_classifier_root"),
    (r"find_dst_skin_root", "find_second_classifier_root"),
    (r"_CSG_MARKER", "_CLASSIFIER_MARKER"),
    (r"_DST_MARKER", "_SECOND_CLASSIFIER_MARKER"),
    (r"_CSG_ROOT", "_CLASSIFIER_REPO"),
    (r"the canonical OOD evaluation script::", "the canonical OOD evaluation script, function "),
    (r"paper-3/docs/", "docs/"),
    (r"paper-3/", ""),
    (r"\bpaper-3\b", "this project"),
]

# Strings that must not survive into the bundle. Checked after substitution.
FORBIDDEN = [
    "CSG-SKin", "DST-Skin", "CSG_SKIN_ROOT", "DST_SKIN_ROOT",
    "cbm_revision", "eval_ood_benchmarks",
    "cubo", "/Users/", "hust", "Hanoi", "Duc-Vinh", "Quyet-Thang",
    "github.com/", "zenodo", "orcid",
]


def anonymise(text: str) -> str:
    for pattern, replacement in SUBSTITUTIONS:
        text = re.sub(pattern, replacement, text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    for sub in ("scripts", "analysis", "figures", "results"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    written = 0
    for group, dest in ((EXPERIMENT, "scripts"), (ANALYSIS, "analysis"), (FIGURES, "figures")):
        for rel in group:
            src = ROOT / rel
            if not src.is_file():
                raise SystemExit(f"missing: {rel}")
            override = ROOT / "paper" / "supplement_overrides" / src.name
            if override.is_file():
                # Blind substitution would leave this file misleading rather
                # than merely clumsy, so it has a hand-written counterpart.
                shutil.copy2(override, OUT / dest / src.name)
            else:
                (OUT / dest / src.name).write_text(anonymise(src.read_text()))
            written += 1

    for csv in sorted((ROOT / "results").glob("*.csv")):
        shutil.copy2(csv, OUT / "results" / csv.name)
        written += 1

    # Per-image scores for every scorer: the arrays the pipeline computed but
    # never wrote. Reviewers asked for these specifically.
    raw_src = ROOT / "results" / "raw_scores"
    if raw_src.is_dir():
        raw_dst = OUT / "results" / "raw_scores"
        raw_dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(raw_src.iterdir()):
            if f.suffix in {".npz", ".csv"}:
                shutil.copy2(f, raw_dst / f.name)
                written += 1

    # Hand-written files (README, requirements, seed manifest, the independent
    # power re-implementation) are maintained in the bundle itself, not derived,
    # so they are left alone -- but they are still scanned below.

    leaks = []
    for f in sorted(OUT.rglob("*")):
        if not f.is_file() or f.suffix not in {".py", ".md", ".txt", ".json"}:
            continue
        body = f.read_text(errors="replace")
        for bad in FORBIDDEN:
            if bad.lower() in body.lower():
                leaks.append(f"{f.relative_to(OUT)}: {bad}")
    if leaks:
        raise SystemExit("anonymisation failed, these strings survived:\n  " + "\n  ".join(leaks))

    print(f"{written} derived files written, anonymisation scan clean")

    if not args.no_zip:
        ZIP.unlink(missing_ok=True)
        subprocess.run(["zip", "-rq", str(ZIP), ".", "-x", "*.DS_Store", "-x", "*__pycache__*"],
                       cwd=OUT, check=True)
        print(f"wrote {ZIP.relative_to(ROOT)} ({ZIP.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
