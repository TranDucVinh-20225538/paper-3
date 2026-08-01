# Pipeline Readiness Audit

**Method**: dry-run only — no checkpoint loaded, no dataset touched, no network access assumed. Every claim below is either (a) an actual command run in this session with its real output shown, or (b) a direct comparison against an existing, already-working CSG-SKin script's exact call pattern (`cbm_revision/scripts/eval_ood_benchmarks.py`), not a guess about what "should" work.

**Scope**: only files that currently exist in `paper-3/scripts/` are classified. The Task 4 batch driver (loop over 18 checkpoints) and any E2 analysis script (`analysis/`) have not been written yet — they are not INCOMPLETE, they are not started, and are listed separately at the end so their absence isn't confused with a defect in something that does exist.

---

## `scripts/geometry_diagnostics.py` — **READY**

**Imports**: clean. Only external dependency is `numpy` (present locally: 2.4.2) plus one CSG-SKin import (`src.utils.ood_metrics.compute_mahalanobis_params_from_arrays`), which itself only needs `numpy`, `torch`, `scikit-learn` — all present locally. Verified by direct execution in this session, not by reading alone:

```
$ python3 scripts/geometry_diagnostics.py
GeometryDiagnostics(condition_number=1.38, fisher_ratio=122385.4, fisher_ratio_scalar=4.73,
                     mardia_kurtosis_b=284.2, mardia_kurtosis_z=-0.54, ...)
```

**Circular dependencies**: none. Import direction is one-way: `CSG-SKin/src/utils/ood_metrics.py` ← `geometry_diagnostics.py`. CSG-SKin has no knowledge of `paper-3/`'s existence and cannot import back into it — structurally acyclic, not just acyclic-by-luck.

**Interfaces**: `compute_geometry_diagnostics(features, labels, num_classes, reg_eps)` → `GeometryDiagnostics` namedtuple with exactly the fields `experiment_contract.md`'s E1 output schema names (`condition_number`, `fisher_ratio` → CSV's `fisher_ratio_HL`, `fisher_ratio_scalar`, `mardia_kurtosis_b`, `mardia_kurtosis_z`, plus `n_samples`/`feat_dim`/`num_classes`/`reg_eps`). Consumed correctly by `extract_embeddings_e1.py` (checked field-by-field against that script's `row = {...}` dict — no name or ordering mismatch).

**Correctness verified this session, not merely inherited**: the Mardia kurtosis null-calibration bug found and fixed earlier in execution mode is confirmed still fixed after the later `fisher_ratio_scalar` addition (re-ran the smoke test above after that change; unaffected).

**Numerical scaling caveat (not a defect, a documented limit)**: `mardia_kurtosis`'s parametric-bootstrap null (`n_bootstrap=200` default) is cheap at $d=16$ (`z_lesion`) — confirmed by how fast the 40×200 = 8,000-replicate calibration check ran in this session. It would be materially slower at `backbone_raw`'s $d\geq1536$, but `geometry_metric_audit.md` already scopes Mardia kurtosis away from that dimensionality for unrelated statistical reasons (§3, C1/C2) — the two constraints happen to agree, not by coordination but because both stem from the same $n$-vs-$d$ regime.

**Verdict: READY.** Can run standalone, right now, with no external data or checkpoints, on any (features, labels) array. Nothing here is waiting on the server.

---

## `scripts/extract_embeddings_e1.py` — **BLOCKED**

**Imports**: fails, at a precise, single, already-diagnosed point — not a design defect. Actual traceback from this session:

```
$ python3 scripts/extract_embeddings_e1.py --checkpoint /nonexistent --method csg --rung x --seed 1
  File ".../CSG-SKin/src/datasets/skin_dataset.py", line 10, in <module>
    import pytorch_lightning as pl
ModuleNotFoundError: No module named 'pytorch_lightning'
```

This is `environment_requirements.md`'s already-documented gap, not a new finding. Once `pytorch-lightning`/`torchmetrics` are installed (per that document's process — from the server's existing working environment, not invented locally), import resolution proceeds past this line; nothing else in the import chain is expected to fail for the same reason (`torch`, `torchvision`, `pandas`, `scikit-learn`, `Pillow` are all already present locally).

**Circular dependencies**: none, same reasoning as above, plus one additional link checked directly: `extract_embeddings_e1.py` does `from geometry_diagnostics import compute_geometry_diagnostics` as a bare (non-package) import. Verified in this session that this resolves regardless of invocation working directory, because Python inserts the *script's own* directory (not the caller's CWD) as `sys.path[0]` — confirmed `os.path.dirname(os.path.abspath("scripts/extract_embeddings_e1.py"))` resolves to `paper-3/scripts`, exactly where `geometry_diagnostics.py` lives, for any invocation of the form `python3 scripts/extract_embeddings_e1.py ...` run from `paper-3/`. This import could not be exercised past the `pytorch_lightning` failure point in this session (it appears later in the file), so this is a static verification of the mechanism, not a live one — flagged as such, not overstated.

**Interfaces — two issues found and fixed during this audit, not left as open findings**:
1. Compared the script's `SkinDataModule(...)`/`_build_isic_train_loader(...)` construction against `eval_ood_benchmarks.py`'s exact working call pattern, field by field. Initially flagged a suspected bug (missing `dm.setup()` call before `_build_isic_train_loader(dm)`, present in the canonical script) — traced through `SkinDataModule.__init__`/`.setup()` in `skin_dataset.py` directly and confirmed `metadata_csv`, `split_config`, and `eval_transform` are all set in `__init__`, not `.setup()`; `.setup()` is only needed for `build_id_ood_test_dataloaders`, which this script never calls. **Not a bug** — confirmed by reading the actual method bodies, not assumed.
2. Found a real inconsistency: `EffB3SingleLightning.load_from_checkpoint(...)` was called with `strict=False`, but the one existing reference call in CSG-SKin (`cbm_revision/scripts/run_effb3_control.py:229`) uses no such flag, and nothing documents a state-dict-mismatch reason for this class the way there is for `CSGLiteLightning`. **Fixed** — removed the unjustified `strict=False` for the `effb3_single` branch; `csg`'s `strict=False` is kept, since that one *is* used consistently across every CSG-SKin reference call (`check_leakage.py`, `eval_ood_benchmarks.py`, `plot_confusion_matrix_runb.py`).

**Output locations**: `results/e1_geometry_metrics.csv` and `results/e1_embeddings/{rung}_s{seed}.npz`, matching `experiment_contract.md`'s E1 output naming exactly. `results/` already exists (currently empty except `.gitkeep`) — write path is valid, not just assumed.

**Contract compliance (`experiment_contract.md`, E1)**:
- ✅ Explicit-file-only checkpoint input, directories rejected outright (`_require_explicit_file`, raises with a message citing `REPOSITORY_MAP.md` risk #5).
- ✅ ISIC-train-under-eval-transform loader, matching the canonical script, not `train_csg.py`'s augmented loader.
- ✅ Output CSV schema matches column-for-column.
- ⬜ Manifest-driven resolution across all 18 (rung, seed) checkpoints — correctly out of scope for this script (that's the not-yet-written Task 4 batch driver, one checkpoint at a time is this script's whole job).
- ✅ Dataset/checkpoint paths configurable, not hardcoded — `--checkpoint`, `--metadata_csv`, `--output_dir` are all CLI arguments.
- ✅ **CSG-SKin root resolution — was BROKEN on first real deployment, now fixed.** Both scripts originally hardcoded `Path(__file__).resolve().parents[2] / "CSG-SKin"`, assuming `paper-3/` sits *beside* `CSG-SKin/` (true locally). The actual GPU server deployment nests `paper-3/` *inside* `CSG-SKin/` (`CSG-SKin/paper-3/scripts/...`), under which that hardcoded parent-count resolves to a nonexistent `CSG-SKin/CSG-SKin`, producing `ModuleNotFoundError: No module named 'src'` — confirmed by the server's actual `--help` run. Fixed by extracting a shared `scripts/_repo_paths.py::find_csg_skin_root()`, which locates CSG-SKin's root by searching for a marker file (`src/utils/ood_metrics.py`) as either an ancestor of the script or a `CSG-SKin`-named sibling of any ancestor, with an explicit `CSG_SKIN_ROOT` env var override for any layout neither heuristic covers. Both `geometry_diagnostics.py` and `extract_embeddings_e1.py` now use this shared resolver instead of two independent hardcoded assumptions. **Verified in this session under both layouts** (actual local sibling layout, and a constructed copy reproducing the server's nested layout at `/tmp/nested_test/CSG-SKin/paper-3/`) — both now fail at the identical, already-documented `pytorch_lightning` point instead of at path resolution.

**Verdict: BLOCKED**, on two external preconditions only (`pytorch-lightning`/`torchmetrics` installation per `environment_requirements.md`; real checkpoint + dataset paths per the GPU server). Everything checkable without those two preconditions has been checked, and two real issues were found and fixed in the process rather than deferred.

---

## Not yet created (not classified — nothing to audit)

- **Task 4 batch driver** (loop `extract_embeddings_e1.py` over all 18 checkpoints, build the checkpoint manifest `open_questions.md` Q3 already flagged as a prerequisite) — not written, not requested yet.
- **E2 association script** (`analysis/`, Spearman-ρ test per `experiment_contract.md`) — not written.
- **E3/E4 scripts** — not written; E4 in particular would run in a structurally different environment (local, DST-Skin-only, no `pytorch-lightning` dependency at all — see `environment_requirements.md` §6) from anything audited here.

## Housekeeping note (not a readiness issue)

Running `geometry_diagnostics.py` created a `scripts/__pycache__/` directory. `paper-3/` has no `.gitignore` yet (the parent `Research/` directory isn't a git repository at all currently). Not blocking anything, but worth having before this gets copied to the server, so compiled bytecode and any future local-only scratch files don't get carried along.

---

## `scripts/_repo_paths.py` — **READY**

Added after first real server deployment surfaced the parent-count bug described above. Pure path-resolution logic, no experiment code. Verified in this session under both the actual local sibling layout and a constructed copy reproducing the server's nested layout (`CSG-SKin/paper-3/scripts/...`) — both resolve `CSG-SKin`'s root correctly and fail (only) at the same, already-documented `pytorch_lightning` import point.

## Summary

| Script | Status | What's actually blocking it |
|---|---|---|
| `geometry_diagnostics.py` | **READY** | Nothing — runs today, verified. |
| `_repo_paths.py` | **READY** | Nothing — verified under both known deployment layouts. |
| `extract_embeddings_e1.py` | **BLOCKED** | Two external preconditions (environment, data/checkpoints) — code itself checked as far as static analysis allows, three real issues found and fixed in the process (missing `dm.setup()` ruled out as a false alarm; `strict=False` mismatch fixed; CSG-root path resolution fixed after real server feedback). |

No script is INCOMPLETE or BROKEN. The intended end-state — `git pull` then `python extract_embeddings_e1.py ...` with no `ImportError`/`ModuleNotFound`/wrong-path debugging loop on the server — depends on `environment_requirements.md`'s action items (server-side `pip freeze`, pinned `requirements-e1.txt`) and on real checkpoint/dataset paths being supplied. The one thing that *did* need fixing after real deployment feedback (path resolution) has been fixed and verified under the server's actual layout, not just the local one.
