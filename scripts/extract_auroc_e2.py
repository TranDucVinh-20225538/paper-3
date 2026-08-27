"""
E2 input -- one checkpoint in, one Mahalanobis AUROC/FPR95 row out.

Refactored from CSG-SKin's cbm_revision/scripts/eval_ood_benchmarks.py, the
canonical Mahalanobis source for E1-E3, into a single-checkpoint script with
an explicit path, matching extract_embeddings_e1.py. A batch driver over all
13 primary-ladder checkpoints is a separate step.

Two departures from the canonical script, both forced by the
explicit-checkpoint requirement:

  1. No directory globbing. The canonical script resolves each seed's
     checkpoint to the newest-by-mtime best-*.ckpt in a directory, which is
     confirmed wrong for several runB/runB_orth1 seeds (open_questions.md
     Q3, REPOSITORY_MAP.md risk 5). Here --checkpoint is an explicit file
     path; --rung and --seed are recorded on the output row and never used
     to locate anything.

  2. Dispatch is by --method (baseline/csg/effb3_single), not by a per-rung
     directory table. The canonical table covers Baseline Soft, Run A/GRL
     and Run B (orth=1.0) but has no entry for runB (orth=5.0), so it could
     not produce a number for 3 of the 13 checkpoints. Since --method here
     selects only an architecture -- identical across the three CSG rungs,
     which differ in lambda_orth during training but not in how features are
     read out of a trained checkpoint -- runB needs no special case.

Everything downstream of feature extraction is the canonical script's own
code, imported rather than reimplemented.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

# CSG-SKin's root is located via _repo_paths (marker-file search,
# layout-agnostic) -- see _repo_paths.py / geometry_diagnostics.py's own use
# of it after the server's nested-layout deployment broke a hardcoded
# parent-count assumption.
from _repo_paths import find_csg_skin_root  # noqa: E402

_CSG_ROOT = find_csg_skin_root(__file__)
if str(_CSG_ROOT) not in sys.path:
    sys.path.insert(0, str(_CSG_ROOT))

from src.datasets.skin_dataset import SkinDataModule, SkinDataset  # noqa: E402
from src.datasets.splits import build_id_ood_test_dataloaders, load_filtered_master  # noqa: E402
from src.models.baseline import BaselineResNet50  # noqa: E402
from src.models.csg_lightning import CSGLiteLightning  # noqa: E402
from src.models.effb3_single import EffB3SingleLightning  # noqa: E402
from src.utils import ood_metrics  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

NUM_CLASSES = 8  # CSG-SKin's fixed 8-class ISIC label space (src/datasets/constants.py)
REG_EPS = 1e-5  # deliberately NOT eval_ood_benchmarks.py's 1e-3 -- matches E1's geometry metrics instead, per open_questions.md Q5
GLOBAL_SEED = 42  # eval_ood_benchmarks.py's own fixed seed, independent of the checkpoint's training seed
METHODS = {"baseline", "csg", "effb3_single"}
K_VALUES = (1, 10, 50)  # E2.6 k-NN grid, pre-registered in experiment_contract.md -- all always reported
PRIMARY_K = 10  # E2.6 headline value, per experiment_contract.md; the other K_VALUES are the robustness grid


def _require_explicit_file(path_str: str) -> Path:
    """Refuse directories outright -- REPOSITORY_MAP.md risk #5 / open_questions.md Q3."""
    p = Path(path_str)
    if not p.is_file():
        raise ValueError(
            f"--checkpoint must be an explicit file, got: {path_str!r} "
            f"(is_dir={p.is_dir()}, exists={p.exists()}). "
            "Directory-based checkpoint resolution is explicitly disallowed for this project "
            "(REPOSITORY_MAP.md risk #5: find_checkpoint is confirmed to pick non-best checkpoints "
            "for several runB/runB_orth1 seed directories; open_questions.md Q3: "
            "eval_ood_benchmarks.py's own _find_ckpt has the identical bug)."
        )
    return p


def _build_isic_train_loader(dm: SkinDataModule, batch_size: int, num_workers: int) -> DataLoader:
    """Byte-for-byte the same construction as extract_embeddings_e1.py's / eval_ood_benchmarks.py's."""
    df = load_filtered_master(dm.metadata_csv)
    isic_df = df[df["domain"] == "isic"].copy()
    isic_train_val, _ = train_test_split(
        isic_df,
        test_size=dm.split_config.isic_test_fraction,
        stratify=isic_df["label_idx"],
        random_state=dm.split_config.random_state,
    )
    isic_train, _ = train_test_split(
        isic_train_val,
        test_size=dm.split_config.val_fraction,
        stratify=isic_train_val["label_idx"],
        random_state=dm.split_config.random_state,
    )
    dataset = SkinDataset(isic_train, transform=dm.eval_transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def _extract_baseline(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    model.eval()
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        _logits, f = model(images, return_features=True)
        feats.append(f.cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


@torch.no_grad()
def _extract_csg(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    model.eval()
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        _logits, _d_ctx, _d_adv, z_lesion, _z_ctx = model(images, x_lesion=None, return_latents=True)
        feats.append(z_lesion.cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


@torch.no_grad()
def _extract_effb3(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    model.eval()
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        z = model.extract_embedding(images)
        feats.append(z.cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def load_model(checkpoint: Path, method: str, device):
    """
    Loads the checkpoint ONCE and returns (model, extract_fn) so the caller can
    reuse the same in-memory model across the train/id/ood loaders -- matching
    eval_ood_benchmarks.py's own efficiency (it loads each checkpoint once and
    calls its collector three times), not extract_embeddings_e1.py's per-loader
    load_and_extract (E1 only ever needed one loader per invocation, so it
    never had this redundant-reload consideration).
    """
    if method == "baseline":
        lit = BaselineResNet50.load_from_checkpoint(str(checkpoint))
        return lit.to(device), _extract_baseline
    if method == "csg":
        lit = CSGLiteLightning.load_from_checkpoint(str(checkpoint), strict=False)
        return lit.model.to(device), _extract_csg
    if method == "effb3_single":
        # No strict=False here, matching extract_embeddings_e1.py's precedent
        # (cbm_revision/scripts/run_effb3_control.py:229 is the one real
        # reference call for this class, and it uses no such flag).
        lit = EffB3SingleLightning.load_from_checkpoint(str(checkpoint))
        return lit.net.to(device), _extract_effb3
    raise ValueError(f"Unknown method {method!r}, expected one of {METHODS}")


def compute_mahalanobis_scores(
    z_train: np.ndarray, y_train: np.ndarray, z_id: np.ndarray, z_ood: np.ndarray,
    num_classes: int, reg_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mirrors eval_ood_benchmarks.py::_compute_scores's Mahalanobis branch
    exactly: fit on (z_train, y_train), score z_id/z_ood by min squared
    Mahalanobis distance to the nearest class mean. Returns the raw
    per-sample (s_id, s_ood) arrays -- not just AUROC/FPR95 derived from
    them -- so the actual distance distributions can be inspected directly,
    plus (means, precision) so compute_extended_diagnostics can reuse the
    exact same fitted object rather than refitting. A one-checkpoint
    temporary-print check (runA_grl s42, open_questions.md Q6) already
    confirmed median(OOD)=7.82 < median(ID)=13.78 and mean(OOD)=16.47 <
    mean(ID)=20.64 -- crossing the pre-declared threshold for running this
    in full across all 13 checkpoints.
    """
    means, precision = ood_metrics.compute_mahalanobis_params_from_arrays(
        z_train, y_train, num_classes=num_classes, reg_eps=reg_eps
    )
    s_id = ood_metrics.mahalanobis_min_squared_distances(z_id, means, precision)
    s_ood = ood_metrics.mahalanobis_min_squared_distances(z_ood, means, precision)
    return s_id, s_ood, means, precision


def cosine_centroid_scores(z: np.ndarray, means: np.ndarray) -> np.ndarray:
    """
    E2.6 scorer #2 (experiment_contract.md): score(z) = min_c [1 - cosine_similarity(z, mean_c)].
    `means` is the SAME per-class means array already fit for Mahalanobis (compute_mahalanobis_scores),
    reused rather than refit -- the only change from Mahalanobis is dropping the covariance/precision
    step, isolating exactly that one variable. No normalization decision needed: cosine similarity is
    scale-invariant in both arguments by construction.
    """
    z_unit = z / np.linalg.norm(z, axis=1, keepdims=True)
    means_unit = means / np.linalg.norm(means, axis=1, keepdims=True)
    cosine_sim = z_unit @ means_unit.T  # (n, K)
    return (1.0 - cosine_sim).min(axis=1)


def compute_knn_scores(
    z_train: np.ndarray, z_id: np.ndarray, z_ood: np.ndarray, k_values: tuple[int, ...],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    E2.6 scorer #3 (experiment_contract.md): pooled k-NN distance (Sun et al. 2022-style) --
    Euclidean distance to the k-th nearest neighbor in the FULL ISIC-train set, all 8 classes
    pooled together (no per-class structure, no covariance, the maximally assumption-free
    instrument in this comparison -- decided explicitly over a per-class-then-min variant that
    would have kept too much structural similarity to Mahalanobis to serve as a real contrast).
    z_train/z_id/z_ood are disjoint splits, so no self-match exclusion is needed. Fits one
    NearestNeighbors index at k_max and slices it for every k in k_values, rather than refitting
    per k.
    """
    k_max = max(k_values)
    index = NearestNeighbors(n_neighbors=k_max).fit(z_train)
    dist_id, _ = index.kneighbors(z_id)
    dist_ood, _ = index.kneighbors(z_ood)
    return {k: (dist_id[:, k - 1], dist_ood[:, k - 1]) for k in k_values}


def _all_class_squared_distances(z: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """
    (n, K) matrix of squared Mahalanobis distance from each sample to EVERY
    class mean, not just the minimum ood_metrics.mahalanobis_min_squared_distances
    returns. Looped over K classes (K=8, cheap) with a vectorized BLAS matmul
    per class -- same pattern as geometry_diagnostics.py's
    _squared_mahalanobis_distances, not CSG's own O(n*K) pure-Python double
    loop. Needed to recover the nearest-centroid PREDICTED class (argmin),
    which the min-only function discards.
    """
    num_classes = means.shape[0]
    d2 = np.empty((z.shape[0], num_classes), dtype=np.float64)
    for c in range(num_classes):
        diff = z - means[c]
        d2[:, c] = np.sum((diff @ precision) * diff, axis=1)
    return d2


def compute_extended_diagnostics(
    z_id: np.ndarray, z_ood: np.ndarray, y_id: np.ndarray, y_ood: np.ndarray,
    s_id: np.ndarray, s_ood: np.ndarray, means: np.ndarray, precision: np.ndarray,
) -> dict:
    """
    Additional per-sample diagnostics beyond the min-distance scores AUROC
    uses -- feature norms (norm-collapse check), true labels, and the
    nearest-centroid predicted class -- computed once so future
    investigation (norm collapse, class imbalance, class-specific distance)
    never needs a rerun. Purely additive: does not affect AUROC/FPR95.

    Cross-checks the vectorized argmin computation's own minimum against
    s_id/s_ood (from CSG's own mahalanobis_min_squared_distances) and warns,
    rather than silently proceeding, if they disagree beyond floating-point
    tolerance -- the two must be measuring the same quantity for
    predicted_class to be trustworthy.
    """
    d2_id = _all_class_squared_distances(z_id, means, precision)
    d2_ood = _all_class_squared_distances(z_ood, means, precision)

    if not np.allclose(d2_id.min(axis=1), s_id, rtol=1e-5, atol=1e-3):
        print("[extract_auroc_e2] WARNING: vectorized per-class min distance (ID) does not match "
              "ood_metrics.mahalanobis_min_squared_distances -- predicted_class_id may be unreliable.")
    if not np.allclose(d2_ood.min(axis=1), s_ood, rtol=1e-5, atol=1e-3):
        print("[extract_auroc_e2] WARNING: vectorized per-class min distance (OOD) does not match "
              "ood_metrics.mahalanobis_min_squared_distances -- predicted_class_ood may be unreliable.")

    return {
        "feature_norm_id": np.linalg.norm(z_id, axis=1),
        "feature_norm_ood": np.linalg.norm(z_ood, axis=1),
        "labels_id": y_id,
        "labels_ood": y_ood,
        "predicted_class_id": d2_id.argmin(axis=1),
        "predicted_class_ood": d2_ood.argmin(axis=1),
    }


def auroc_fpr95_from_scores(s_id: np.ndarray, s_ood: np.ndarray) -> tuple[float, float]:
    """y=1 for OOD (PAD-UFES), higher score = more OOD-like -- same label/score convention as
    eval_ood_benchmarks.py, not reinterpreted."""
    y_ood_binary = np.concatenate([np.zeros(len(s_id), dtype=np.int64), np.ones(len(s_ood), dtype=np.int64)])
    scores = np.concatenate([s_id, s_ood])
    auroc = float(roc_auc_score(y_ood_binary, scores))
    fpr95 = ood_metrics.fpr_at_95_tpr(y_ood_binary, scores)
    return auroc, fpr95


def append_auroc_row(csv_path: Path, row: dict) -> None:
    fieldnames = ["rung", "method", "seed", "checkpoint_path", "auroc", "fpr95"]
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_distance_summary_row(csv_path: Path, row: dict) -> None:
    fieldnames = [
        "rung", "method", "seed", "checkpoint_path",
        "id_n", "id_mean", "id_median", "id_p95",
        "ood_n", "ood_mean", "ood_median", "ood_p95",
    ]
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_scorer_comparison_row(csv_path: Path, row: dict) -> None:
    fieldnames = ["rung", "method", "seed", "checkpoint_path", "scorer", "auroc", "fpr95"]
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_raw_embeddings(
    npz_dir: Path, rung: str, seed: int, checkpoint_path: str,
    z_train: np.ndarray, y_train: np.ndarray, z_id: np.ndarray, y_id: np.ndarray,
    z_ood: np.ndarray, y_ood: np.ndarray,
) -> Path:
    """
    E2.6 input requirement (experiment_contract.md): the full raw z_lesion embeddings for all
    three splits, never subsampled -- the k-NN reference pool must be the complete train set or
    k-th-neighbor distances aren't meaningful. Byproduct: also resolves the UMAP-visualization
    blocker, since UMAP can subsample from this full array at analysis time without a further
    rerun. ~1.5MB/checkpoint at 16-d float32 (train+id+ood together), ~20MB total across 13
    checkpoints.
    """
    npz_dir.mkdir(parents=True, exist_ok=True)
    path = npz_dir / f"{rung}_s{seed}_z.npz"
    np.savez(
        path,
        z_train=z_train, y_train=y_train,
        z_id=z_id, y_id=y_id,
        z_ood=z_ood, y_ood=y_ood,
        rung=rung, seed=seed, checkpoint_path=checkpoint_path,
    )
    return path


def save_raw_distances(npz_dir: Path, rung: str, seed: int, checkpoint_path: str,
                        s_id: np.ndarray, s_ood: np.ndarray, diagnostics: dict) -> Path:
    """
    Persists the full per-sample distance arrays and the extended
    diagnostics (feature norms, true labels, nearest-centroid predicted
    class) -- not just distance_summary.csv's mean/median/p95, which can
    hide a bimodal or overlapping distribution, and not just the distances
    alone, which can't answer norm-collapse / class-imbalance / class-specific
    questions without a rerun. A few extra MB per checkpoint now, in
    exchange for never needing to re-load a checkpoint to answer those
    questions later.
    """
    npz_dir.mkdir(parents=True, exist_ok=True)
    path = npz_dir / f"{rung}_s{seed}.npz"
    np.savez(
        path,
        s_id=s_id, s_ood=s_ood,
        feature_norm_id=diagnostics["feature_norm_id"], feature_norm_ood=diagnostics["feature_norm_ood"],
        labels_id=diagnostics["labels_id"], labels_ood=diagnostics["labels_ood"],
        predicted_class_id=diagnostics["predicted_class_id"], predicted_class_ood=diagnostics["predicted_class_ood"],
        rung=rung, seed=seed, checkpoint_path=checkpoint_path,
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Explicit .ckpt file path. Directories are rejected.")
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--rung", required=True, help="e.g. baseline_soft, runA_grl, runB_orth1, runB")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--metadata_csv", default=str(_CSG_ROOT / "data" / "master_metadata_lesion_only_soft.csv"))
    parser.add_argument("--batch_size", type=int, default=128)  # matches eval_ood_benchmarks.py's own default
    parser.add_argument("--num_workers", type=int, default=8)   # matches eval_ood_benchmarks.py's own default (E1 used 4)
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = parser.parse_args()

    checkpoint = _require_explicit_file(args.checkpoint)

    seed_everything(GLOBAL_SEED)
    pl.seed_everything(GLOBAL_SEED, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = SkinDataModule(metadata_csv=args.metadata_csv, batch_size=args.batch_size, num_workers=args.num_workers)
    dm.setup()  # REQUIRED here -- build_id_ood_test_dataloaders needs dm.pad_df_all etc. populated
    id_loader, ood_loader = build_id_ood_test_dataloaders(dm)
    train_loader = _build_isic_train_loader(dm, args.batch_size, args.num_workers)

    model, extract_fn = load_model(checkpoint, args.method, device)
    z_train, y_train = extract_fn(model, train_loader, device)
    z_id, y_id = extract_fn(model, id_loader, device)
    z_ood, y_ood = extract_fn(model, ood_loader, device)

    s_id, s_ood, means, precision = compute_mahalanobis_scores(z_train, y_train, z_id, z_ood, NUM_CLASSES, REG_EPS)
    auroc, fpr95 = auroc_fpr95_from_scores(s_id, s_ood)

    output_dir = Path(args.output_dir)

    row = {
        "rung": args.rung,
        "method": args.method,
        "seed": args.seed,
        "checkpoint_path": str(checkpoint),
        "auroc": auroc,
        "fpr95": fpr95,
    }
    append_auroc_row(output_dir / "e2_auroc.csv", row)

    # E2.5 (open_questions.md Q6, restored + extended): persist the raw
    # per-sample distances AND feature norms / true labels / nearest-centroid
    # predicted class -- saving all of it now (a few extra MB per checkpoint)
    # means norm-collapse, class-imbalance, and class-specific-distance
    # questions can be answered later without reloading any checkpoint. A
    # one-checkpoint temporary-print check already confirmed OOD < ID on both
    # mean and median for runA_grl s42, crossing the pre-declared threshold
    # for running this across the full ladder -- see
    # analysis/analyze_e2_distances.py for histogram/KDE/ECDF/boxplot.
    diagnostics = compute_extended_diagnostics(z_id, z_ood, y_id, y_ood, s_id, s_ood, means, precision)
    npz_path = save_raw_distances(
        output_dir / "e2_distances", args.rung, args.seed, str(checkpoint), s_id, s_ood, diagnostics
    )
    distance_row = {
        "rung": args.rung, "method": args.method, "seed": args.seed, "checkpoint_path": str(checkpoint),
        "id_n": len(s_id), "id_mean": float(np.mean(s_id)), "id_median": float(np.median(s_id)),
        "id_p95": float(np.percentile(s_id, 95)),
        "ood_n": len(s_ood), "ood_mean": float(np.mean(s_ood)), "ood_median": float(np.median(s_ood)),
        "ood_p95": float(np.percentile(s_ood, 95)),
    }
    append_distance_summary_row(output_dir / "distance_summary.csv", distance_row)

    print(f"AUROC={auroc:.4f} FPR95={fpr95:.4f}")
    print(row)
    print(f"Raw distances saved: {npz_path}")
    print(distance_row)

    # E2.6 (open_questions.md Q6 / experiment_contract.md): Cosine-to-centroid and pooled k-NN
    # on the identical embeddings, formulas locked before this code was written. All K_VALUES
    # are always computed and written -- k=10 (PRIMARY_K) is the headline value, {1,50} are the
    # pre-registered robustness grid, none selected after seeing results.
    cosine_s_id = cosine_centroid_scores(z_id, means)
    cosine_s_ood = cosine_centroid_scores(z_ood, means)
    knn_scores = compute_knn_scores(z_train, z_id, z_ood, K_VALUES)

    scorer_scores = {"mahalanobis": (s_id, s_ood), "cosine": (cosine_s_id, cosine_s_ood)}
    scorer_scores.update({f"knn_k{k}": scores for k, scores in knn_scores.items()})

    scorer_comparison_path = output_dir / "e2_6_scorer_comparison.csv"
    for scorer_name, (sc_id, sc_ood) in scorer_scores.items():
        sc_auroc, sc_fpr95 = auroc_fpr95_from_scores(sc_id, sc_ood)
        append_scorer_comparison_row(scorer_comparison_path, {
            "rung": args.rung, "method": args.method, "seed": args.seed,
            "checkpoint_path": str(checkpoint), "scorer": scorer_name,
            "auroc": sc_auroc, "fpr95": sc_fpr95,
        })
        print(f"[E2.6] {scorer_name:10s} AUROC={sc_auroc:.4f} FPR95={sc_fpr95:.4f}")

    z_path = save_raw_embeddings(
        output_dir / "e2_distances", args.rung, args.seed, str(checkpoint),
        z_train, y_train, z_id, y_id, z_ood, y_ood,
    )
    print(f"[E2.6] scorer comparison appended to {scorer_comparison_path}")
    print(f"[E2.6] raw embeddings saved: {z_path}")


if __name__ == "__main__":
    main()
