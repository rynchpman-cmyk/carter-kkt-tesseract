"""Projection and bounded rapidity-to-current map."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def minkowski_metric() -> Array:
    return np.diag([-1.0, 1.0, 1.0, 1.0])


def lower(vector: Array, metric: Array) -> Array:
    """Lower a leading four-vector index, allowing trailing batch axes."""
    return np.einsum("mn,n...->m...", metric, vector)


def inner(left: Array, right: Array, metric: Array) -> Array:
    return np.einsum("m...,mn,n...->...", left, metric, right)


def normalize_four_velocity(u: Array, metric: Array | None = None) -> Array:
    metric = minkowski_metric() if metric is None else np.asarray(metric, dtype=float)
    norm_sq = inner(u, u, metric)
    if np.any(norm_sq >= 0.0):
        raise ValueError("four-velocity must be timelike")
    return np.asarray(u, dtype=float) / np.sqrt(-norm_sq)[None, ...]


def project_spatial(vector: Array, u: Array, metric: Array | None = None) -> Array:
    """Return Delta^mu_nu vector^nu for a normalized timelike ``u``."""
    metric = minkowski_metric() if metric is None else np.asarray(metric, dtype=float)
    u = normalize_four_velocity(np.asarray(u, dtype=float), metric)
    coefficient = inner(u, vector, metric)
    return np.asarray(vector, dtype=float) + u * coefficient[None, ...]


def tanh_over_x(x: Array | float) -> Array:
    """Stable evaluation of tanh(x)/x, including the origin."""
    x = np.asarray(x, dtype=float)
    x2 = x * x
    series = 1.0 - x2 / 3.0 + 2.0 * x2 * x2 / 15.0 - 17.0 * x2**3 / 315.0
    safe_x = np.where(np.abs(x) < 1.0e-8, 1.0, x)
    direct = np.tanh(x) / safe_x
    return np.where(np.abs(x) < 1.0e-4, series, direct)


def causal_current(
    number_density: Array | float,
    u: Array,
    xi: Array,
    metric: Array | None = None,
    *,
    reproject: bool = True,
) -> tuple[Array, Array, Array]:
    """Construct ``K = n (u + tanh(chi) xi/chi)``.

    Returns ``(K, v, chi)``. By default ``xi`` is projected to protect the
    causal guarantee from numerical orthogonality drift.
    """
    metric = minkowski_metric() if metric is None else np.asarray(metric, dtype=float)
    u = normalize_four_velocity(np.asarray(u, dtype=float), metric)
    xi = project_spatial(xi, u, metric) if reproject else np.asarray(xi, dtype=float)
    chi_sq = inner(xi, xi, metric)
    if np.any(chi_sq < -1.0e-12):
        raise ValueError("projected rapidity has a negative squared norm")
    chi = np.sqrt(np.maximum(chi_sq, 0.0))
    v = tanh_over_x(chi)[None, ...] * xi
    current = np.asarray(number_density, dtype=float) * (u + v)
    return current, v, chi


def causal_margin(current: Array, metric: Array | None = None) -> Array:
    """Return ``-K^2``; positive values are timelike margins."""
    metric = minkowski_metric() if metric is None else np.asarray(metric, dtype=float)
    return -inner(current, current, metric)
