"""
Geometry diagnostics for E1.

Three metrics, one per concern raised in docs/geometry_metric_audit.md 5:

    condition number    covariance conditioning
    Fisher ratio        class separation
    Mardia's kurtosis   distributional shape

The audit rejects a longer list of candidates (trace, log-determinant,
participation ratio, effective rank, anisotropy, spectral entropy, unratioed
scatter, silhouette, Davies-Bouldin, Henze-Zirkler, Mardia's skewness); none
of them is implemented here. Adding one means updating the audit first.

All three read the same (class_means, precision) pair that CSG-SKin's
src.utils.ood_metrics.compute_mahalanobis_params_from_arrays produces, so
the geometry described is the geometry the Mahalanobis estimator actually
sees rather than a separately derived approximation.

A pure-numpy library over already-extracted (features, labels) arrays: no
checkpoint loading, no dataset loading, nothing written to results/.
"""

from __future__ import annotations

import sys
import time
from typing import NamedTuple

import numpy as np

# CSG-SKin is not an installed package. Its root is located via _repo_paths
# (marker-file search, layout-agnostic) rather than a hardcoded parent-count,
# since paper-3/ has been deployed both as a CSG-SKin sibling and nested
# inside it -- see _repo_paths.py.
from _repo_paths import find_csg_skin_root  # noqa: E402

_CSG_ROOT = find_csg_skin_root(__file__)
if str(_CSG_ROOT) not in sys.path:
    sys.path.insert(0, str(_CSG_ROOT))

from src.utils.ood_metrics import compute_mahalanobis_params_from_arrays  # noqa: E402


# ---------------------------------------------------------------------------
# TEMPORARY PROFILING INSTRUMENTATION (per-metric timing, GPU-server
# performance investigation). Gated behind debug=False everywhere -- inert
# by default, zero output and negligible overhead (one time.perf_counter()
# pair per stage) unless a caller explicitly opts in. Safe to delete this
# block and every `debug`/`timings` parameter below once no longer needed.
# ---------------------------------------------------------------------------

def _log_stage_time(label: str, elapsed_seconds: float) -> None:
    print(f"[geometry-timing] {label}: {elapsed_seconds * 1000.0:9.2f} ms", flush=True)


def _render_timing_bar(timings: list[tuple[str, float]], bar_width: int = 30) -> None:
    """ASCII bar chart of per-stage wall-clock time, scaled to the slowest stage."""
    if not timings:
        return
    max_elapsed = max(elapsed for _, elapsed in timings) or 1e-12
    label_width = max(len(label) for label, _ in timings)
    print("[geometry-timing] ---- summary ----", flush=True)
    for label, elapsed in timings:
        filled = int(round(bar_width * elapsed / max_elapsed))
        if elapsed > 0:
            filled = max(filled, 1)  # nonzero stages get a visible sliver, not indistinguishable from 0
        bar = "#" * filled + "-" * (bar_width - filled)
        print(
            f"[geometry-timing] {label:<{label_width}}  {elapsed * 1000.0:9.2f} ms  |{bar}|",
            flush=True,
        )


class GeometryDiagnostics(NamedTuple):
    """One (rung, seed, representation)'s worth of geometry measurements."""

    condition_number: float
    fisher_ratio: float
    fisher_ratio_scalar: float
    mardia_kurtosis_b: float
    mardia_kurtosis_z: float
    n_samples: int
    feat_dim: int
    num_classes: int
    reg_eps: float


def condition_number(precision: np.ndarray) -> float:
    """
    kappa(Sigma) = lambda_max / lambda_min, read off the precision matrix
    rather than inverting back to Sigma. Inversion reciprocates every eigenvalue,
    swapping which is max and which is min without changing their ratio, so
    kappa(Sigma) == kappa(Sigma^-1) and a second inversion is avoided.
    """
    precision = np.asarray(precision, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(precision)
    eigvals = eigvals[eigvals > 0]
    if eigvals.size < 2:
        raise ValueError("Precision matrix has fewer than 2 positive eigenvalues.")
    return float(eigvals.max() / eigvals.min())


def _between_class_scatter(class_means: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """S_B = sum_c n_c (mu_c - mu_bar)(mu_c - mu_bar)^T. Shared by both Fisher-ratio forms below."""
    class_means = np.asarray(class_means, dtype=np.float64)
    labels = np.asarray(labels)
    num_classes, _feat_dim = class_means.shape
    counts = np.array(
        [np.sum(labels == c) for c in range(num_classes)], dtype=np.float64
    )
    if np.any(counts == 0):
        raise ValueError("Every class in class_means must have at least one labeled sample.")
    overall_mean = (counts[:, None] * class_means).sum(axis=0) / counts.sum()
    deviations = class_means - overall_mean  # (num_classes, feat_dim)
    return (deviations * counts[:, None]).T @ deviations  # (feat_dim, feat_dim)


def fisher_ratio(
    class_means: np.ndarray,
    precision: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    J = tr(Sigma^-1 @ S_B), the multivariate Fisher/LDA criterion, using the
    between-class scatter and the same regularized precision the Mahalanobis
    estimator already fit -- one estimation call, three read-outs
    (docs/geometry_metric_audit.md 5).

    Both this and condition_number are functions of that same precision matrix,
    so they share estimation noise by construction. Report it alongside
    fisher_ratio_scalar, never instead of it (docs/fisher_ratio_defense.md 3).
    """
    precision = np.asarray(precision, dtype=np.float64)
    s_between = _between_class_scatter(class_means, labels)
    return float(np.trace(precision @ s_between))


def fisher_ratio_scalar(
    features: np.ndarray,
    class_means: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    tr(S_B)/tr(S_W), the decoupled companion to fisher_ratio: no precision
    matrix, no matrix inverse, no reg_eps dependence, so it cannot inherit
    condition_number's estimation noise. tr(S_W) is summed directly as
    sum_i ||x_i - mu_{y_i}||^2. Isotropic, so it ignores which directions carry
    class-discriminating information -- the price of that independence.
    """
    features = np.asarray(features, dtype=np.float64)
    class_means = np.asarray(class_means, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)

    centered = features - class_means[labels]
    trace_s_within = float(np.sum(centered ** 2))
    if trace_s_within == 0:
        raise ValueError("tr(S_W) is zero — degenerate within-class scatter.")

    trace_s_between = float(np.trace(_between_class_scatter(class_means, labels)))
    return trace_s_between / trace_s_within


def _squared_mahalanobis_distances(centered: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """
    Row-wise (x_i - mu)^T Sigma^-1 (x_i - mu).

    Not np.einsum("ij,jk,ik->i", ...): without optimize=True, einsum runs a
    generic single-threaded C loop instead of routing the (n,d)@(d,d) product
    through BLAS. Profiling put that call at roughly half the runtime of
    mardia_kurtosis's bootstrap calibration. This form is mathematically
    identical and hits BLAS gemm; results agree to ~1e-12 relative, differing
    only in summation order.
    """
    return np.sum((centered @ precision) * centered, axis=1)


def _simulate_mardia_null(
    class_means: np.ndarray,
    precision: np.ndarray,
    labels: np.ndarray,
    reg_eps: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Parametric-bootstrap null for b2,d under this exact estimator: K per-class
    means, pooled covariance with reg_eps, N-K denominator.

    Treats the fitted (class_means, Sigma) as the generating model, draws fresh
    per-class-sized synthetic samples, refits
    compute_mahalanobis_params_from_arrays on each draw rather than scoring
    against the original fit, and computes b2,d in-sample on the refit. That
    reproduces the fit-then-score coupling responsible for the statistic's
    finite-sample bias.
    """
    sigma = np.linalg.inv(precision)
    num_classes, feat_dim = class_means.shape
    counts = [int(np.sum(labels == c)) for c in range(num_classes)]

    # sigma is IDENTICAL across every one of the n_bootstrap x num_classes
    # sampling calls below (only the per-class mean differs). numpy's
    # rng.multivariate_normal defaults to method="svd", which re-decomposes
    # sigma on every single call -- profiling showed 1600 redundant SVD
    # calls (n_bootstrap x num_classes) decomposing the same matrix.
    # Cholesky-decompose it ONCE here and sample manually
    # (standard_normal @ L.T + mean) instead -- mathematically the same
    # multivariate-normal distribution (sigma is guaranteed strictly
    # positive-definite by construction, so Cholesky always exists), just
    # decomposed a single time instead of 1600. NOTE: this changes which
    # pseudorandom draws a fixed --seed consumes (a different, still-valid
    # sampling recipe), not the distribution sampled from or the estimator's
    # statistical validity -- flagged since nothing on the server has
    # produced committed seeded results yet that this would need to match.
    chol_l = np.linalg.cholesky(sigma)

    b2d_null = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        synthetic = np.concatenate(
            [
                class_means[c] + rng.standard_normal((counts[c], feat_dim)) @ chol_l.T
                for c in range(num_classes)
            ]
        )
        means_sim, precision_sim = compute_mahalanobis_params_from_arrays(
            synthetic, labels, num_classes=num_classes, reg_eps=reg_eps
        )
        centered_sim = synthetic - means_sim[labels]
        d2_sim = _squared_mahalanobis_distances(centered_sim, precision_sim)
        b2d_null[i] = np.mean(d2_sim ** 2)
    return b2d_null


def mardia_kurtosis(
    features: np.ndarray,
    class_means: np.ndarray,
    precision: np.ndarray,
    labels: np.ndarray,
    reg_eps: float = 1e-5,
    n_bootstrap: int = 200,
    seed: int = 0,
    debug: bool = False,
    timings: list[tuple[str, float]] | None = None,
) -> tuple[float, float]:
    """
    Mardia's (1970) multivariate kurtosis, adapted from its single-Gaussian form
    to the shared-covariance, per-class-mean model the Mahalanobis estimator
    assumes: residuals are taken against each sample's own class mean and
    measured through the shared precision matrix.

        b2,d = (1/n) sum_i [ (x_i - mu_{y_i})^T Sigma^-1 (x_i - mu_{y_i}) ]^2

    which is the empirical fourth moment of the squared Mahalanobis distances
    the estimator itself computes.

    The classical closed-form null does not apply here. Mardia's mean d(d+2) and
    variance 8d(d+2)/n assume one global mean estimated from all n samples; this
    centres on K class means and inverts a covariance fit with an N-K
    denominator. On synthetic data that is Gaussian by construction
    (n_classes=8, feat_dim=16) the closed form rejects at ~72% against a nominal
    5%, because the finite-sample mean of b2,d under this estimator sits below
    d(d+2) and tracks a per-class (n_c-1)/(n_c+1) correction. It is wrong here,
    not merely approximate.

    The null is therefore calibrated by parametric bootstrap (see
    _simulate_mardia_null) and the observed b2,d is z-scored against it. Same
    statistic, different calibration.

    debug and timings are profiling instrumentation, inert by default.
    """
    features = np.asarray(features, dtype=np.float64)
    class_means = np.asarray(class_means, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)

    t0 = time.perf_counter()
    centered = features - class_means[labels]
    d2 = _squared_mahalanobis_distances(centered, precision)
    b2d = float(np.mean(d2 ** 2))
    elapsed = time.perf_counter() - t0
    if debug:
        _log_stage_time("Mardia statistic (observed b2,d)", elapsed)
    if timings is not None:
        timings.append(("Mardia statistic", elapsed))

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    null_dist = _simulate_mardia_null(class_means, precision, labels, reg_eps, n_bootstrap, rng)
    z_score = float((b2d - null_dist.mean()) / null_dist.std())
    elapsed = time.perf_counter() - t0
    if debug:
        _log_stage_time(f"bootstrap calibration ({n_bootstrap} reps)", elapsed)
    if timings is not None:
        timings.append(("bootstrap calibration", elapsed))

    return b2d, z_score


def compute_geometry_diagnostics(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    reg_eps: float = 1e-5,
    debug: bool = False,
) -> GeometryDiagnostics:
    """
    Single entry point. Fits CSG-SKin's Mahalanobis parameters on
    (features, labels), then reads the approved metric set off that one fit:
    condition_number, fisher_ratio, fisher_ratio_scalar and mardia_kurtosis.

    debug prints per-stage wall-clock timings and a summary bar; it is off by
    default and can be removed along with the _log_stage_time and
    _render_timing_bar calls once no longer needed.
    """
    timings: list[tuple[str, float]] = []

    t0 = time.perf_counter()
    class_means, precision = compute_mahalanobis_params_from_arrays(
        features, labels, num_classes=num_classes, reg_eps=reg_eps
    )
    elapsed = time.perf_counter() - t0
    if debug:
        _log_stage_time("covariance fitting", elapsed)
    timings.append(("covariance fitting", elapsed))

    t0 = time.perf_counter()
    kappa = condition_number(precision)
    elapsed = time.perf_counter() - t0
    if debug:
        _log_stage_time("condition number", elapsed)
    timings.append(("condition number", elapsed))

    t0 = time.perf_counter()
    fisher = fisher_ratio(class_means, precision, labels)
    fisher_scalar = fisher_ratio_scalar(features, class_means, labels)
    elapsed = time.perf_counter() - t0
    if debug:
        _log_stage_time("Fisher ratio (HL + scalar)", elapsed)
    timings.append(("Fisher ratio", elapsed))

    b2d, z_score = mardia_kurtosis(
        features, class_means, precision, labels, reg_eps=reg_eps, debug=debug, timings=timings
    )

    if debug:
        _render_timing_bar(timings)

    n_samples, feat_dim = features.shape
    return GeometryDiagnostics(
        condition_number=kappa,
        fisher_ratio=fisher,
        fisher_ratio_scalar=fisher_scalar,
        mardia_kurtosis_b=b2d,
        mardia_kurtosis_z=z_score,
        n_samples=n_samples,
        feat_dim=feat_dim,
        num_classes=num_classes,
        reg_eps=reg_eps,
    )


if __name__ == "__main__":
    # Smoke test only — synthetic data, no checkpoints/datasets touched.
    # This verifies the module runs correctly; it is not an E1 experiment.
    rng = np.random.default_rng(42)
    n_classes, feat_dim, per_class = 8, 16, 200

    true_means = rng.normal(scale=3.0, size=(n_classes, feat_dim))
    synth_features = np.concatenate(
        [rng.normal(loc=true_means[c], scale=1.0, size=(per_class, feat_dim)) for c in range(n_classes)]
    )
    synth_labels = np.repeat(np.arange(n_classes), per_class)

    diagnostics = compute_geometry_diagnostics(synth_features, synth_labels, num_classes=n_classes, debug=True)
    print(diagnostics)
