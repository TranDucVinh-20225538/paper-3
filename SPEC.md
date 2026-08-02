# Paper 3 — Specification

**Status**: hypothesis and experiment plan only. No experiments have been implemented. This document will be revised as the design firms up; it is not a frozen artifact the way Paper 1/2 are.

---

## 1. Research question

> **When do representation improvements translate into better reliability estimation?**

## 2. Explicit non-goals

To keep scope bounded for a solo-author, 6-month project:

- This is **not** a methods paper — no new representation-learning technique is being proposed.
- This is **not** an OOD-detection paper — OOD detection is not the object of study.
- This is **not** a Mahalanobis paper — Mahalanobis distance is not being improved, tuned, or compared against other estimators as an end in itself.

Mahalanobis-based reliability estimation (as already implemented in Paper 2 / DST-Skin) is used strictly as **the current instrument** for measuring "reliability estimation quality." If a different instrument were used, the research question would be unchanged; Mahalanobis is a means, not the subject.

## 3. Mechanistic hypothesis (H3)

> Representation improvements induced by shortcut-mitigation training change the feature geometry. These geometric changes may alter the assumptions required by Mahalanobis reliability estimation. Therefore, representation improvements may fail to translate into improved reliability estimation.

In other words: a representation can get *better* by the metric shortcut-mitigation training was designed to improve (lower domain leakage, per Paper 1's leakage probe) while simultaneously becoming *worse-suited* to a downstream reliability estimator whose validity depends on specific geometric assumptions (e.g. approximately class-conditional Gaussian clusters with a shared, well-conditioned covariance — the assumption Mahalanobis distance relies on). H3 is the candidate explanation for why representation quality and reliability-estimation quality might decouple, rather than move together as commonly assumed.

**Refined framing** (per `open_questions.md` Q4): the intended reading of H3 is *within* the disentanglement architecture family — increasing orthogonality regularization is associated with progressive geometric change that tracks Mahalanobis reliability. A conventional (non-disentangled) baseline is included as a categorical reference to contextualize the magnitude of that shift, not as a fourth dose level of the same intervention — see §4's primary/secondary split below for why.

## 4. Experiment plan

### E1 — Measure geometry: primary disentanglement ladder + categorical baseline reference

**Design correction** (`open_questions.md` Q4): `baseline_soft` was originally treated as "λ_orth = 0" on the same ordinal ladder as the three CSG methods. It is not a valid zero-dose condition — it differs from the CSG rungs in architecture (ResNet-50 vs. EfficientNet-B3-based CSGLite), representation dimensionality (2048-d backbone feature vs. 16-d `z_lesion`), and training recipe (learning rate, augmentation pipeline — `open_questions.md` Q1). Pooling it into one 4-point dose-response trend conflates "training recipe changed" with "architecture and dimensionality changed" at the very first step, and (independently) made Mardia's kurtosis's bootstrap calibration try to operate at d=2048, a regime `geometry_metric_audit.md` §3 already flagged as outside this test's validated range. E1 is therefore split into a **primary mechanistic test** and a **secondary/contextual comparison**, not one four-point ladder.

**Primary mechanistic test — the disentanglement ladder.** Independent variable: orthogonality regularization strength (λ_orth), holding architecture, representation, and (once `open_questions.md` Q1 is verified) training recipe fixed:

| Rung | Method | Mechanism | λ_orth | Representation |
|---|---|---|---|---|
| 1 | `runA_grl` | domain-adversarial only | 0.0 | 16-d `z_lesion` |
| 2 | `runB_orth1` | + orthogonality term | 1.0 | 16-d `z_lesion` |
| 3 | `runB` | + stronger orthogonality | 5.0 | 16-d `z_lesion` |

This is the ladder H3's mechanistic claim is actually tested against — three ordinal points, one thing varying.

**Secondary / contextual comparison — baseline as a categorical reference**, not a dose level:

| Condition | Method | Representation |
|---|---|---|
| Categorical reference | `baseline_soft` | 2048-d ResNet-50 backbone feature |

Answers a different, complementary question — *does introducing the disentanglement framework at all change representation geometry and reliability, relative to a conventional classifier* — not "how does geometry respond to increasing dose." Reported as a labeled comparison point against the primary ladder's range, never pooled into the primary trend statistic.

**Geometry metrics**: resolved in `geometry_metric_audit.md` §5 (condition number, Fisher ratio [Hotelling-Lawley + decoupled scalar companion], Mardia's kurtosis) — no longer an open implementation decision. Mardia's kurtosis is load-bearing only for the primary ladder (16-d `z_lesion`, its validated regime); for the baseline reference (2048-d) it is computed but treated as exploratory, consistent with the dimensionality scope `geometry_metric_audit.md` §3 (C1/C2) already specified before any code existed.

**Output**: two per-seed tables — the primary ladder (3 rungs × up to 5 seeds) and the baseline reference (1 condition × up to 5 seeds) — extracted from the existing CSG-SKin checkpoints (no retraining).

### E2 — Test whether geometry changes are associated with Mahalanobis AUROC

**Objective**: test the second link in H3 — does the geometry shift measured in E1's primary ladder predict Mahalanobis-based reliability-estimation quality (Mahalanobis OOD-detection AUROC, ISIC-test vs. PAD-UFES)?

**Primary**: associate the primary ladder's geometry metrics with per-seed Mahalanobis AUROC across the three CSG rungs only. Three ordinal points is not a well-powered regression — report a rank-based trend (e.g. Kendall's τ) and seed-level scatter, not a claim of strong statistical power dressed up as a fitted regression.

**Secondary**: compare the primary ladder's (geometry, AUROC) pattern against the baseline reference's single (geometry, AUROC) point descriptively — does geometry help explain reliability differences across training families (disentangled vs. conventional)? A categorical contrast, not a continuation of the primary trend line.

**Output**: an association result for the primary ladder (Kendall's τ + scatter), and a separate descriptive comparison against the baseline reference.

### E3 — Validate on a strictly held-out third domain (HAM10000)

**Objective**: check that the E1↔E2 relationship isn't an artifact of the ISIC/PAD-UFES split used everywhere else in Papers 1 and 2, by re-running the same geometry/reliability measurement on a domain neither CSG-SKin nor DST-Skin was trained or tuned on.

**Design**: apply the same CSG-SKin checkpoints (no retraining) and the same geometry + Mahalanobis measurement pipeline from E1/E2 to HAM10000 as a third, strictly held-out domain. Mirrors E1/E2's primary/secondary split: the primary replication check is on the three-rung disentanglement ladder; `baseline_soft` is replicated only as the same categorical reference, not folded into the trend statistic.

### E4 — Optional supplementary validation on DST-Skin (independent codebase)

**Objective**: check whether the E1↔E2 relationship generalizes beyond CSG-SKin's specific architecture and training recipe, using Paper 2's independently-trained backbones (ResNet-18/50, EfficientNet-B3) and its own Ledoit-Wolf-shrinkage Mahalanobis implementation as an independent measurement path.

**Status**: optional / supplementary, not required for the core claim. Scoped last, and only if E1–E3 support H3 and time remains.

---

## 5. Reuse and non-modification constraints

- **Paper 1 (CSG-SKin) and Paper 2 (DST-Skin) are frozen historical artifacts.** No code in either repository is modified as part of Paper 3 unless explicitly instructed otherwise for a specific, isolated change.
- **All new code lives under `paper-3/scripts/` and `paper-3/analysis/`.** Existing functionality (checkpoint loading, feature/embedding extraction, Mahalanobis fitting) is reused via **import**, not copy-paste or reimplementation.
- **Every experiment must be reproducible**: pinned seeds, checkpoints referenced by explicit file path (never a directory — see feasibility note below), and versioned outputs written to `paper-3/results/`.

## 6. Known feasibility risks and dependencies (from `REPOSITORY_MAP.md`)

These were identified during the repository audit and are worth resolving explicitly during design, before E1–E4 are implemented:

- **E1/E2 seed coverage is uneven across the primary ladder.** `runB` (λ_orth=5.0) checkpoints exist for only 3 of 5 seeds (42, 52, 62) in the current `checkpoints/` inventory; `runA_grl` and `runB_orth1` each have all 5 (as does `baseline_soft`, now the secondary reference, not part of this ladder). Per-rung statistics across the three-rung primary ladder need to either match on the common 3 seeds or explicitly report the asymmetry.
- **Checkpoint resolution must use explicit files, not directories.** CSG-SKin's `find_checkpoint` (`src/utils/ood_metrics.py`) resolves the *newest-by-mtime* file in a directory, which is confirmed to pick a non-best-val checkpoint in several `runB`/`runB_orth1` seed directories due to stale duplicates from repeated training runs. E1/E2/E3 scripts must reference specific `.ckpt` files, verified against each run's actual best epoch, not a resolved directory path.
- **CSG-SKin's own "OOD" evaluation set is not a clean held-out set.** By construction, all of PAD-UFES is both CSG's auxiliary training-domain stream and its OOD-test set (`SkinDataModule.setup`). If E2 uses CSG-SKin's own pre-computed summary.json Mahalanobis AUROC numbers directly, those numbers reflect this contamination. This is a specific reason E3's genuinely-held-out HAM10000 check matters, and also a reason E2 may need to recompute Mahalanobis AUROC on a cleaner split rather than trusting the existing per-run summaries as-is.
- **Two Mahalanobis implementations exist and are not interchangeable.** CSG-SKin's `ood_metrics.py` uses a raw pooled within-class covariance; DST-Skin's `OODScorer` uses Ledoit-Wolf shrinkage on L2-normalized features. E1/E2/E3 (CSG-based) and E4 (DST-based) will produce numbers from different formulas — this is expected given E4's role as an independent validation path, but any cross-experiment comparison must say explicitly which formula produced which number.
- **HAM10000 data is not currently present in either repository.** DST-Skin contains an orphaned preprocessing script (`split.py`) that references `data/raw/ham10000/`, which does not exist on disk in this checkout. E3 will need this data acquired and prepared before it can run — this is a hard external dependency, not a code dependency.
- **Label-taxonomy mismatch between the two repos.** CSG-SKin uses the full 8-class ISIC label space; DST-Skin uses a binarized malignant/benign relabeling, and the two repos' malignant-class definitions for PAD-UFES don't match each other's conventions. E4 (DST-Skin path) will need its own, separately-defined label handling — it cannot silently inherit CSG-SKin's label logic.

## 7. Open design decisions

- ~~The precise geometry metric(s) for E1~~ — **resolved**: `geometry_metric_audit.md` §5 (condition number, Fisher ratio, Mardia's kurtosis).
- ~~Whether E2's association test is correlation, regression, or something else~~ — **resolved**: Kendall's τ + seed-level scatter on the three-rung primary ladder, explicitly not framed as a well-powered regression (`open_questions.md` Q4).
- ~~Whether `baseline_soft` belongs on the same ordinal ladder as the CSG methods~~ — **resolved**: no. Split into primary ladder (3 CSG rungs) + secondary categorical reference (`baseline_soft`) — see §4 and `open_questions.md` Q4.
- Where/how HAM10000 data will be sourced and preprocessed for E3.
- Whether E4 is scoped at all, and if so, to what extent, given its "optional" status and the 6-month solo-author budget.
