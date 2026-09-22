# Verifying the reported claims

Two findings were reported to the co-author. This document exists so that each
one can be re-checked against a committed artifact rather than taken on trust.

Run everything at once:

```bash
python3 scripts/verify_claims.py
```

Twelve checks, each printing the number it found. Checks needing the 208 MB
embedding cache (gitignored, archived separately) or the sibling classifier
repository report `SKIP` rather than passing silently.

---

## Claim (a) — the source dataset is ISIC 2019, not ISIC 2018

The manuscript said ISIC 2018 and cited the ISIC 2018 pair. That was wrong.

| Evidence | Where to look |
|---|---|
| Code resolves images under `ISIC_2019_Training_Input` | `../CSG-SKin/src/datasets/preprocess_metadata.py`, `skin_dataset.py` |
| Label set is the 8-class ISIC 2019 one, including `SCC` | `../CSG-SKin/src/datasets/constants.py` |
| Metadata has 25,331 rows | `results/dataset_split.csv`, and `wc -l` on the metadata CSV |
| The realised split sums to exactly that | `results/dataset_split.csv`: 16,211 + 4,053 + 5,067 = 25,331 |
| Manuscript now says ISIC 2019 | `grep -c "ISIC 201[89]" paper/tmlr/common_body.tex` |
| BCN20000 cited | `grep combalia2019bcn20000 paper/tmlr/references.bib` |

ISIC 2018 Task 3 has **seven** classes and uses `AKIEC`; ISIC 2019 has **eight**
and uses `AK` plus `SCC`. The label set decides this on its own.

**Where the error came from.** `preprocess_metadata.py` carries a compatibility
shim that renames `AKIEC` to `AK` when it sees ISIC-2018-style column headings.
That is a fallback for reading an older column naming, not a statement about
which dataset is used. It appears to have been read as the latter.

**What changed in the manuscript.** ISIC 2018 to ISIC 2019 in both places it
appeared; `combalia2019bcn20000` added, since ISIC 2019 combines BCN20000,
HAM10000 and the MSK data; and the image count, class list and split sizes
written into Methods, which previously gave none of them.

---

## Claim (b) — per-image scores now exist for all eight scorers

The pipeline computed a per-image score array for every scorer, collapsed it to
one AUROC/FPR95 row, and discarded the array. Only Mahalanobis survived, because
`extract_auroc_e2.py` happened to cache its distances alongside the embeddings.

`analysis/dump_raw_scores.py` rebuilds the rest by importing the same scoring
functions the original scripts used — not reimplementing them — and refuses to
write anything whose recomputed AUROC and FPR95 do not match the published
value.

| Artifact | Contents |
|---|---|
| `results/raw_scores/<run>_scores.npz` | `<scorer>__id` (5,067 values) and `<scorer>__ood` (2,298) for each scorer |
| `results/raw_scores/validation_report.csv` | one row per (checkpoint, scorer): recomputed AUROC/FPR95, deviation from the published value, verdict |

110 (checkpoint x scorer) combinations. 109 reproduce the published number, the
largest deviation being 9.0e-07 and most at 5.6e-17. The 110th, `baseline_soft`
with the KDE scorer, has no published value to compare against: the original
script excluded the baseline from that scorer by design, so this is a new
number, not a mismatch.

Energy and ViM additionally need the classifier head to rebuild logits. The head
must come from the checkpoint the cached embeddings were built from, which is
recorded in `results/lesion_classifier_heads_cache_manifest.json`.

The `status` column of the validation report reads `khớp` (matched) or
`không có mốc` (no published reference); `scripts/verify_claims.py` counts them
for you.

---

## What this repository cannot verify on its own

- **Re-running `dump_raw_scores.py` from a fresh clone.** It needs
  `results/e2_distances/` — 208 MB of cached embeddings, two files of which
  exceed GitHub's hard limit. They are gitignored and released separately; see
  the Data availability section of `README.md`. The validation report is the
  committed evidence that the run happened and what it found.
- **The checkpoints themselves.** Not in this repository and not on the
  development machine; they live on the training cluster.
