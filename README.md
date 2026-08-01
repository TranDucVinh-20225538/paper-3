# Paper 3

## Purpose

Paper 3 is an independent research project. It is not a continuation, refactor, or extension of Paper 1 ([`CSG-SKin/`](../CSG-SKin)) or Paper 2 ([`DST-Skin/`](../DST-Skin)) — both are treated as **frozen historical artifacts**: their checkpoints, code, and results are read from, never modified.

**Research question:**

> When do representation improvements translate into better reliability estimation?

This is **not** a methods paper, **not** an OOD-detection paper, and **not** a Mahalanobis paper. Mahalanobis distance is used here only as *the current downstream reliability estimator* through which the research question is tested — the object of study is the relationship between representation quality and the validity of a reliability estimate built on top of it, not the estimator itself.

See [`SPEC.md`](SPEC.md) for the mechanistic hypothesis (H3) and the four planned experiments (E1–E4).

## Relationship to Paper 1 and Paper 2

- **Paper 1 (CSG-SKin)** contributes a *dose-response ladder* of representation quality: four checkpoint families (`baseline_soft → runA_grl → runB_orth1 → runB`) with progressively stronger shortcut-mitigation (domain-adversarial + orthogonality) training, all sharing the same backbone family and data. This ladder is Paper 3's primary independent variable.
- **Paper 2 (DST-Skin)** contributes the Mahalanobis-based reliability-estimation methodology (Ledoit-Wolf shrinkage covariance in normalized feature space) used as the dependent-variable measurement instrument, and is also a candidate site for supplementary validation (E4) since it offers a second, independently-trained set of backbones.
- Full audit of both source repos — module map, dependency graph, reusable functions, and known pitfalls — is in [`REPOSITORY_MAP.md`](REPOSITORY_MAP.md). Read it before writing anything that touches either repo's code or checkpoints.

## Status

Scaffolding only. No experiments have been implemented yet.

## Layout

```
paper-3/
├── README.md              # this file
├── SPEC.md                 # research question, H3, and the E1–E4 experiment plan
├── REPOSITORY_MAP.md      # audit of CSG-SKin and DST-Skin, dependency graph, risks
├── scripts/                # pipeline entry points for E1–E4 (glue code, new code only)
├── analysis/                # notebooks/scripts consuming results/ to produce figures/
├── results/                # experiment outputs (CSVs, JSON summaries)
├── figures/                # generated figures for the paper
└── docs/                    # design notes, decisions, open questions
```

## Ground rules

- **`CSG-SKin/` and `DST-Skin/` are never modified.** No exceptions unless explicitly instructed otherwise for a specific change.
- **Reuse through imports, not duplication.** Where Paper 1 or Paper 2 already implement something Paper 3 needs (feature extraction, Mahalanobis fitting, checkpoint loading), import and call it — don't copy-paste or re-derive it into `paper-3/`. See `REPOSITORY_MAP.md` §3 for the current inventory of what's reusable as-is.
- **New code lives in `paper-3/`.** Every experiment script is new, added here, not patched into either source repo.
- **Every experiment must be reproducible** — pinned seeds, recorded checkpoint paths (explicit files, never a resolved-from-directory path — see `REPOSITORY_MAP.md` risk #5), and versioned outputs in `results/`.
- **Scope discipline.** This is a solo-author, 6-month project. Each experiment (E1–E4) should be scoped to what's needed to test H3, not expanded into a broader methods or benchmarking exercise.
