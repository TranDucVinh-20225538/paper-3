# Open Questions

Log only. Nothing here gets acted on until its own Action item is explicitly executed as a separate task — this file exists so open concerns don't get silently redesigned around or silently forgotten.

---

## Q1. Dose-response ladder may contain multiple simultaneous interventions

**Status**: Open.

**Context**: `internal_review.md` found that CSG-SKin's four ladder rungs (`baseline_soft → runA_grl → runB_orth1 → runB`) differ by more than `λ_orth` alone — per `REPOSITORY_MAP.md` §2.8, `baseline_soft` uses a different learning rate (2e-4 vs. 1e-4) and a different augmentation pipeline (light vs. robust transforms) than the three CSG runs. Rung 0 is not a controlled zero-dose condition.

**Action**: Verify actual per-rung training configs (learning rate, transform pipeline, epoch count) against `REPOSITORY_MAP.md`'s §2.8 table before interpreting any monotonic or rank-correlation trend from E1/E2 as caused specifically by shortcut-mitigation strength.

**Priority**: Medium.

---

## Q2. Fisher ratio's shared-precision-matrix entanglement with condition number

**Status**: Open, with a partial mitigation already specified.

**Context**: `fisher_ratio_defense.md` §5 requires a bootstrap check ($\mathrm{Corr}(\kappa, J)$ at fixed true geometry) before trusting any E1/E2 result that leans on Fisher ratio, and requires reporting the decoupled $\mathrm{tr}(S_B)/\mathrm{tr}(S_W)$ companion alongside it.

**Action**: Run the bootstrap check specified in `fisher_ratio_defense.md` §4 once real per-rung sample sizes are known. Not blocking Task 1–2; blocking before E2 interprets any Fisher-ratio result.

**Priority**: Medium — has a defined trigger point (before E2), not before E1's implementation work.

---

## Q3. `eval_ood_benchmarks.py` — the canonical E2 Mahalanobis source — resolves checkpoints by directory, not explicit file

**Status**: Open.

**Context**: found while writing `extract_embeddings_e1.py` and reading `eval_ood_benchmarks.py` directly. Its `_find_ckpt(dir_path)` picks `sorted(dir_path.glob("best-*.ckpt"), key=mtime, reverse=True)[0]`, i.e. the same newest-by-mtime, directory-based resolution `REPOSITORY_MAP.md` risk #5 already flagged as confirmed-wrong for several `runB`/`runB_orth1` seed directories. `threats_to_validity.md` #2 designated this script as the canonical E1–E3 Mahalanobis source specifically to avoid `train_csg.py`'s transform-mismatch bug — but never checked it against the separate checkpoint-resolution bug. It has it too.

**Action**: before E2 runs, verify whether `eval_ood_benchmarks.py`'s per-seed checkpoint choice (for `runB` and `runB_orth1`) matches the explicit, manifest-verified paths `extract_embeddings_e1.py` uses for E1's geometry metrics on the same seeds. If they diverge, E1's geometry and E2's AUROC would be describing two different checkpoints for the same nominal (rung, seed) row.

**Priority**: High — this one directly threatens whether E1 and E2's outputs are even about the same model, not just a downstream statistical concern.
