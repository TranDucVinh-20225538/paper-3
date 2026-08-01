"""
Geometry diagnostics for E1 (paper-3/SPEC.md).

Implements exactly the 3-metric minimal set justified in
paper-3/docs/geometry_metric_audit.md §5 — no more, no fewer:

  1. Condition number       (Concern 1: covariance conditioning)
  2. Fisher ratio           (Concern 2: class-separation geometry)
  3. Mardia's kurtosis      (Concern 3: distributional shape / Gaussianity)

Every metric that geometry_metric_audit.md §5 rejects (trace, log-determinant,
participation ratio, effective rank, covariance anisotropy, spectral entropy,
unratioed within/between scatter, silhouette, Davies-Bouldin, Henze-Zirkler,
Mardia's skewness) is deliberately NOT implemented here. Do not add any of
them without first updating the audit's rejection rationale.

Reuse, not reimplementation: all three metrics are computed from the single
`(class_means, precision)` pair produced by CSG-SKin's own
`src.utils.ood_metrics.compute_mahalanobis_params_from_arrays` — the exact
regularized precision matrix the Mahalanobis reliability estimator itself
uses, imported rather than re-derived, per paper-3/CLAUDE.md's reuse rule.

This module contains no experiment-execution code: no checkpoint loading, no
dataset loading, no results/ writing. It is a pure-numpy diagnostics library
over already-extracted (features, labels) arrays, to be wired into an E1
experiment script later. CSG-SKin itself is never modified — this module
only imports from it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

# CSG-SKin is a sibling repo to paper-3/, not an installed package — mirror
# CSG-SKin's own scripts' convention (sys.path insert of the repo root)
# rather than duplicating compute_mahalanobis_params_from_arrays's logic.
_CSG_ROOT = Path(__file__).resolve().parents[2] / "CSG-SKin"
if str(_CSG_ROOT) not in sys.path:
    sys.path.insert(0, str(_CSG_ROOT))

from src.utils.ood_metrics import compute_mahalanobis_params_from_arrays  # noqa: E402


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
    kappa(Sigma) = lambda_max(Sigma) / lambda_min(Sigma).

    Computed directly from the precision matrix (Sigma^-1), not by inverting
    back to Sigma: condition number is invariant under matrix inversion
    (kappa(Sigma) == kappa(Sigma^-1)), since inversion just reciprocates every
    eigenvalue, which flips which one is "max" and which is "min" without
    changing their ratio. This avoids a second, separately-conditioned
    inversion of an already-inverted, already-regularized matrix.
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
    J = tr(Sigma^-1 @ S_B), the multivariate Fisher/LDA discriminability
    criterion, using the between-class scatter matrix S_B and the SAME
    regularized precision matrix (Sigma^-1) the Mahalanobis estimator
    already fit — not a separately-estimated within-class scatter inverse —
    so this is read off the exact object the reliability estimator inverts,
    per geometry_metric_audit.md §5's "one estimation call, three read-outs"
    design.

    Entangled with condition_number by construction (both are functions of
    the same `precision`) — see fisher_ratio_defense.md §3. Report alongside
    fisher_ratio_scalar, not instead of it.
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
    Decoupled companion to fisher_ratio, made mandatory (not optional) by
    fisher_ratio_defense.md §5: tr(S_B)/tr(S_W), computed WITHOUT the
    precision matrix at all, so it cannot share condition_number's
    estimation-noise entanglement with fisher_ratio the way that metric
    does. tr(S_W) is computed directly as sum_i ||x_i - mu_{y_i}||^2 — no
    matrix inverse, no reg_eps dependence. Coarser (isotropic, ignores which
    directions carry more class-discriminating information) in exchange for
    that independence — the trade-off fisher_ratio_defense.md §3 describes
    as irreducible, not a bug to fix.
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


def _simulate_mardia_null(
    class_means: np.ndarray,
    precision: np.ndarray,
    labels: np.ndarray,
    reg_eps: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Parametric-bootstrap null for b2,d under this exact estimator (K
    per-class means, pooled covariance with reg_eps, N-K denominator).

    Treats the fitted (class_means, Sigma=precision^-1) as the true
    generating model, draws fresh per-class-sized synthetic samples from it,
    REFITS compute_mahalanobis_params_from_arrays on each synthetic draw
    (not just scoring against the original fit), and computes b2,d in-sample
    on that refit -- reproducing the exact fit-then-score coupling that
    produces the real statistic's finite-sample bias, rather than assuming a
    closed-form asymptotic formula that does not apply to this estimator
    (see module docstring / mardia_kurtosis below for why not).
    """
    sigma = np.linalg.inv(precision)
    num_classes, feat_dim = class_means.shape
    counts = [int(np.sum(labels == c)) for c in range(num_classes)]

    b2d_null = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        synthetic = np.concatenate(
            [rng.multivariate_normal(class_means[c], sigma, size=counts[c]) for c in range(num_classes)]
        )
        means_sim, precision_sim = compute_mahalanobis_params_from_arrays(
            synthetic, labels, num_classes=num_classes, reg_eps=reg_eps
        )
        centered_sim = synthetic - means_sim[labels]
        d2_sim = np.einsum("ij,jk,ik->i", centered_sim, precision_sim, centered_sim)
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
) -> tuple[float, float]:
    """
    Mardia's (1970) multivariate kurtosis statistic, adapted from its
    classical single-Gaussian form to the shared-covariance,
    per-class-conditional-mean model CSG-SKin's Mahalanobis estimator
    actually assumes (Lee et al., 2018): residuals are taken relative to
    each sample's OWN class mean, not one global mean, then measured against
    the shared precision matrix both repos' Mahalanobis distance itself uses.

        b2,d = (1/n) * sum_i [ (x_i - mu_{y_i})^T Sigma^-1 (x_i - mu_{y_i}) ]^2

    which is exactly the empirical fourth moment of the per-sample squared
    Mahalanobis distances the reliability estimator computes (not merely
    correlated with them) — see geometry_metric_audit.md §5.

    THE STATISTIC IS b2,d; THE NULL IS NOT MARDIA'S CLOSED-FORM ASYMPTOTIC
    FORMULA. Mardia's (1970) mean d(d+2) / variance 8*d*(d+2)/n is derived
    for a SINGLE global mean estimated from all n samples. This function
    instead centers each sample on its OWN class mean (K means, each fit
    from ~n/K samples) and inverts a covariance fit with an N-K denominator
    -- a structurally different estimator. Verified empirically (parametric
    bootstrap vs. exactly-Gaussian synthetic data, n_classes=8, feat_dim=16):
    the closed-form formula gives a ~72% false-rejection rate at nominal 5%
    significance on data that is Gaussian by construction, because the true
    finite-sample mean of b2,d under this estimator is measurably below
    d(d+2), tracking a per-class (n_c-1)/(n_c+1) correction rather than the
    classical single-mean formula. Do not reintroduce the closed-form
    z-score; it is not a conservative approximation here, it is wrong.

    The null used below is instead calibrated by parametric bootstrap: treat
    the fitted (class_means, Sigma) as the true model, simulate synthetic
    data at the same per-class sample sizes, refit and rescore exactly as
    the real pipeline does, and z-score the observed b2,d against the
    resulting empirical null mean/std. This still tests the same statistic
    approved in geometry_metric_audit.md §5 — only its significance
    calibration changed.
    """
    features = np.asarray(features, dtype=np.float64)
    class_means = np.asarray(class_means, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)

    centered = features - class_means[labels]
    d2 = np.einsum("ij,jk,ik->i", centered, precision, centered)
    b2d = float(np.mean(d2 ** 2))

    rng = np.random.default_rng(seed)
    null_dist = _simulate_mardia_null(class_means, precision, labels, reg_eps, n_bootstrap, rng)
    z_score = float((b2d - null_dist.mean()) / null_dist.std())

    return b2d, z_score


def compute_geometry_diagnostics(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    reg_eps: float = 1e-5,
) -> GeometryDiagnostics:
    """
    Single entry point: fits CSG-SKin's own Mahalanobis parameters (imported,
    not reimplemented) on (features, labels), then computes exactly the
    approved metric set from that one fit: condition_number, fisher_ratio
    (Hotelling-Lawley, entangled with condition_number by construction),
    fisher_ratio_scalar (decoupled companion, mandatory per
    fisher_ratio_defense.md §5), and mardia_kurtosis (bootstrap-calibrated).
    """
    class_means, precision = compute_mahalanobis_params_from_arrays(
        features, labels, num_classes=num_classes, reg_eps=reg_eps
    )

    kappa = condition_number(precision)
    fisher = fisher_ratio(class_means, precision, labels)
    fisher_scalar = fisher_ratio_scalar(features, class_means, labels)
    b2d, z_score = mardia_kurtosis(features, class_means, precision, labels, reg_eps=reg_eps)

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

    diagnostics = compute_geometry_diagnostics(synth_features, synth_labels, num_classes=n_classes)
    print(diagnostics)
