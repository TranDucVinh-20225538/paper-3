"""
E2.8 -- Energy, ViM and per-class KDE density on the embeddings E2.6 already
cached. Formulas were fixed in docs/experiment_contract.md before this file
was written.

Unlike extract_auroc_e2.py this loads no dataloader, touches no image and
imports no pytorch_lightning. It needs only the cached embeddings in
results/e2_distances/*_z.npz and two tensors read straight out of each
checkpoint's state_dict, model.lesion_classifier.{weight,bias}, which let
logits be rebuilt offline as z @ W.T + b.

That reconstruction was checked rather than assumed: on runA_grl_s42's
cached z_id/z_train it reproduces the classifier's own argmax at 81.0% and
99.4% accuracy against true labels, confirming the cached z_lesion is
post-lesion_bn -- the stage lesion_classifier consumes.

Primary ladder only. baseline_soft is a different architecture with a
different classifier head and is excluded.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity

from _repo_paths import find_csg_skin_root, find_dst_skin_root

_CSG_ROOT = find_csg_skin_root(__file__)
_DST_ROOT = find_dst_skin_root(__file__)


def _load_module_from_path(module_name: str, file_path: Path):
    """CSG-SKin and DST-Skin both use the generic top-level package name `src`, so a plain
    `sys.path.insert` + `import src.utils.X` for both in the same process silently resolves the
    second repo's `src` to whatever got imported first (ModuleNotFoundError or, worse, the wrong
    file). Loading each needed file directly by path under a unique module name sidesteps the
    collision entirely -- no two-repo sys.path interaction to reason about."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# CSG-SKin's ood_metrics.py imports standalone (only torch/numpy/sklearn, no pytorch_lightning).
_csg_ood_metrics = _load_module_from_path("csg_ood_metrics", _CSG_ROOT / "src" / "utils" / "ood_metrics.py")
ood_metrics = _csg_ood_metrics

# DST-Skin's ood_vim_react.py must be loaded before scoring.py, which does `from .ood_vim_react
# import ...` via a relative import -- register it under the exact relative name Python expects
# by first creating a `dst_src_utils` package stand-in is more machinery than needed here; instead
# load ood_vim_react.py standalone (fit_vim/vim_score, needed directly anyway) and load scoring.py
# with its relative import satisfied by placing DST-Skin's `src` package on sys.path ONLY for this
# one import, then removing it immediately so it can't collide with CSG-SKin's `src` afterward.
_dst_vim_react = _load_module_from_path("dst_ood_vim_react", _DST_ROOT / "src" / "utils" / "ood_vim_react.py")
fit_vim = _dst_vim_react.fit_vim
vim_score = _dst_vim_react.vim_score

sys.path.insert(0, str(_DST_ROOT))
try:
    import src.utils.scoring as _dst_scoring  # noqa: E402
    OODScorer = _dst_scoring.OODScorer
finally:
    sys.path.remove(str(_DST_ROOT))
    for _mod_name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[_mod_name]

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DISTANCES_DIR = RESULTS_DIR / "e2_distances"
NUM_CLASSES = 8
FEAT_DIM = 16
SCORER_NAMES = ("energy", "vim", "density_kde")

# Same 13-checkpoint primary ladder as E2/E2.6 -- baseline_soft excluded, see module docstring.
PRIMARY_LADDER = [
    ("runA_grl", s) for s in (42, 52, 62, 72, 82)
] + [
    ("runB_orth1", s) for s in (42, 52, 62, 72, 82)
] + [
    ("runB", s) for s in (42, 52, 62)
]


def resolve_checkpoint(rung: str, seed: int, cached_checkpoint_path: str) -> Path:
    """
    Cached checkpoint_path strings are relative to whatever cwd extract_auroc_e2.py
    was run from originally, which is not this machine's cwd. Resolve by basename
    under CSG-SKin's own checkpoints/csg_lite/{rung}_s{seed}/ instead:
    layout-agnostic, and still an explicit verified file rather than a
    directory-newest guess.
    """
    basename = Path(cached_checkpoint_path).name
    resolved = _CSG_ROOT / "checkpoints" / "csg_lite" / f"{rung}_s{seed}" / basename
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected checkpoint at {resolved} (from cached path {cached_checkpoint_path!r})")
    return resolved


def load_classifier_head(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    """Reads model.lesion_classifier.{weight,bias} directly out of the checkpoint's state_dict --
    no pytorch_lightning import, no full CSGLiteLightning instantiation needed for this."""
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    W = sd["model.lesion_classifier.weight"].numpy().astype(np.float64)
    b = sd["model.lesion_classifier.bias"].numpy().astype(np.float64)
    assert W.shape == (NUM_CLASSES, FEAT_DIM), f"unexpected lesion_classifier shape {W.shape}"
    return W, b


def reconstruct_logits(z: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Spurious "invalid value encountered in matmul" RuntimeWarnings are a known benign
    # numpy+Apple-Accelerate quirk on this machine -- verified (experiment_contract.md E2.8
    # numerics check) that the actual output contains zero NaN/Inf and matches float64 matmul
    # to ~2e-6, i.e. ordinary float32 precision, not a real numerical problem.
    with np.errstate(all="ignore"):
        return z.astype(np.float64) @ W.T + b


def energy_scores(logits: np.ndarray) -> np.ndarray:
    """Higher = more OOD-like. DST-Skin's OODScorer.score_energy returns the opposite
    convention ('higher = more ID/confident' per its own docstring) -- negated here at the
    call site, not reimplemented (experiment_contract.md E2.8, scorer #4)."""
    return -OODScorer.score_energy(logits.astype(np.float32), T=1.0)


def density_kde_scores(z_train: np.ndarray, y_train: np.ndarray, z_query: np.ndarray) -> np.ndarray:
    """Per-class Gaussian-kernel KDE, Scott's-rule bandwidth, score = -max_c[log p_c(z)]
    (experiment_contract.md E2.8, scorer #6). Higher = more OOD-like (low likelihood under
    every class's own density model)."""
    d = z_train.shape[1]
    log_probs = np.empty((z_query.shape[0], NUM_CLASSES), dtype=np.float64)
    for c in range(NUM_CLASSES):
        z_c = z_train[y_train == c]
        n_c = len(z_c)
        sigma_bar = z_c.std(axis=0, ddof=1).mean()
        bandwidth = max(sigma_bar * (n_c ** (-1.0 / (d + 4))), 1e-6)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(z_c)
        log_probs[:, c] = kde.score_samples(z_query)
    return -log_probs.max(axis=1)


def auroc_fpr95_from_scores(s_id: np.ndarray, s_ood: np.ndarray) -> tuple[float, float]:
    """Same convention as extract_auroc_e2.py's own function of the same name (not imported
    directly -- that module requires pytorch_lightning, which this script deliberately avoids;
    the two lines of glue below are not a reimplementation of any scoring LOGIC, only of the
    trivial concatenate-and-call step, with fpr_at_95_tpr itself still imported from
    CSG-SKin's ood_metrics)."""
    y_ood_binary = np.concatenate([np.zeros(len(s_id), dtype=np.int64), np.ones(len(s_ood), dtype=np.int64)])
    scores = np.concatenate([s_id, s_ood])
    auroc = float(roc_auc_score(y_ood_binary, scores))
    fpr95 = ood_metrics.fpr_at_95_tpr(y_ood_binary, scores)
    return auroc, fpr95


def append_scorer_comparison_row(csv_path: Path, row: dict) -> None:
    import csv

    fieldnames = ["rung", "method", "seed", "checkpoint_path", "scorer", "auroc", "fpr95"]
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    scorer_comparison_path = RESULTS_DIR / "e2_6_scorer_comparison.csv"

    for rung, seed in PRIMARY_LADDER:
        z_path = DISTANCES_DIR / f"{rung}_s{seed}_z.npz"
        d = np.load(z_path, allow_pickle=True)
        z_train, y_train = d["z_train"], d["y_train"]
        z_id, z_ood = d["z_id"], d["z_ood"]
        cached_checkpoint_path = str(d["checkpoint_path"])

        checkpoint = resolve_checkpoint(rung, seed, cached_checkpoint_path)
        W, b = load_classifier_head(checkpoint)

        logits_train = reconstruct_logits(z_train, W, b)
        logits_id = reconstruct_logits(z_id, W, b)
        logits_ood = reconstruct_logits(z_ood, W, b)

        # sanity check: classifier argmax on cached train embeddings should be highly accurate,
        # same check as experiment_contract.md's E2.8 numerics verification -- catches a silent
        # mismatch (wrong checkpoint, wrong cached array) rather than producing a quietly wrong row.
        train_acc = (logits_train.argmax(axis=1) == y_train).mean()
        if train_acc < 0.9:
            raise RuntimeError(
                f"{rung} s{seed}: reconstructed-logit train accuracy {train_acc:.3f} is implausibly "
                "low for a trained classifier -- checkpoint/embedding mismatch suspected, stopping."
            )

        scores = {}

        e_id, e_ood = energy_scores(logits_id), energy_scores(logits_ood)
        scores["energy"] = (e_id, e_ood)

        vim_params = fit_vim(z_train.astype(np.float64), logits_train, W, b)
        v_id = vim_score(z_id.astype(np.float64), vim_params)
        v_ood = vim_score(z_ood.astype(np.float64), vim_params)
        scores["vim"] = (v_id, v_ood)

        k_id = density_kde_scores(z_train, y_train, z_id)
        k_ood = density_kde_scores(z_train, y_train, z_ood)
        scores["density_kde"] = (k_id, k_ood)

        print(f"=== {rung} seed={seed} (train_acc={train_acc:.3f}) ===")
        for scorer_name in SCORER_NAMES:
            sc_id, sc_ood = scores[scorer_name]
            auroc, fpr95 = auroc_fpr95_from_scores(sc_id, sc_ood)
            append_scorer_comparison_row(scorer_comparison_path, {
                "rung": rung, "method": "csg", "seed": seed,
                # the cached relative form, not str(checkpoint): the resolved
                # absolute path is machine-specific and would not match the
                # provenance string extract_auroc_e2.py wrote for the same row
                "checkpoint_path": cached_checkpoint_path, "scorer": scorer_name,
                "auroc": auroc, "fpr95": fpr95,
            })
            print(f"  {scorer_name:12s} AUROC={auroc:.4f} FPR95={fpr95:.4f}")

    print(f"\n[E2.8] scorer comparison appended to {scorer_comparison_path}")


if __name__ == "__main__":
    main()
