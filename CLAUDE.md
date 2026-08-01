# CLAUDE.md — paper-3

Project-specific instructions for Claude Code sessions working in this directory. Read this before touching anything.

## What this project is

Read [`README.md`](README.md) and [`SPEC.md`](SPEC.md) first. In one line: Paper 3 asks *when representation improvements translate into better reliability estimation*, tests it via mechanistic hypothesis H3, and runs four experiments (E1–E4) against checkpoints and code already sitting in the sibling `CSG-SKin/` and `DST-Skin/` repositories.

This is **not** a methods, OOD-detection, or Mahalanobis paper — Mahalanobis is only the current instrument used to measure reliability estimation. Do not let scope drift toward improving or comparing OOD estimators for their own sake.

## Hard constraints

- **Never modify `../CSG-SKin/` or `../DST-Skin/`.** Both are frozen historical artifacts (Paper 1 and Paper 2). No edits, no "quick fixes," no refactors — not even to fix a bug documented in `REPOSITORY_MAP.md` — unless the user explicitly instructs it for that specific change.
- **All new code lives under `paper-3/`** (`scripts/`, `analysis/`). Do not add files to the source repos.
- **Reuse via import, not duplication.** If CSG-SKin or DST-Skin already implement something needed here (checkpoint loading, feature/embedding extraction, Mahalanobis fitting, ECE, etc.), import and call it from `paper-3/` code. Do not copy-paste the implementation or write a parallel version — both source repos already suffer from exactly this kind of near-duplicate drift (documented in `REPOSITORY_MAP.md` §3/§8), don't add a third copy.
- **Checkpoints are referenced by explicit file path, never by directory.** CSG-SKin's `find_checkpoint` resolves the newest-by-mtime file in a directory and is confirmed to pick a non-best-val checkpoint for several `runB`/`runB_orth1` seeds. Always name the exact `.ckpt` file.
- **Every experiment script must be reproducible**: pinned seed, explicit checkpoint paths, and output written to `paper-3/results/` under a name that identifies the experiment and inputs (not overwritten silently on rerun, unlike some scripts in DST-Skin's `extract_once_*.py`).
- **Do not implement experiments until explicitly asked.** As of now, nothing in E1–E4 is implemented — check `SPEC.md`'s status line and `results/`/`scripts/` contents before assuming otherwise.

## Before writing any script that touches either source repo

Read [`REPOSITORY_MAP.md`](REPOSITORY_MAP.md) — it's the audit both source repos' own `docs/REPOSITORY_GUIDE.md` files were synthesized from, plus the cross-repo risks specific to this project. In particular, before starting E1/E2/E3 work, re-check:

- §6 (feasibility risks) of `SPEC.md` — uneven seed coverage on the `runB` rung (3/5 seeds), CSG's contaminated PAD-UFES "OOD" set, the two non-interchangeable Mahalanobis formulas, HAM10000 not existing on disk yet, and the label-taxonomy mismatch between the two repos.
- `REPOSITORY_MAP.md` §5 (risks) for anything not already folded into `SPEC.md`.

If something in either source repo looks like a bug worth fixing, flag it to the user instead of fixing it — see the hard constraint above.

## Open design decisions (do not silently resolve these)

Per `SPEC.md` §7: the specific geometry metric(s) for E1, the association test for E2, HAM10000 sourcing for E3, and whether E4 is in scope at all. These are the user's calls — surface them rather than picking a default and proceeding.

## Scope discipline

Solo-author, 6-month project. When a task could be done narrowly (just what's needed to test H3) or broadly (a fuller benchmark/methods contribution), default to narrow and say so, rather than expanding scope unasked.
