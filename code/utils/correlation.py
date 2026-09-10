"""Correlation measures relating a reservoir's quantum resource to its performance.

Two measures, answering two different questions about the same twelve curves of Figure 5.

`sweep_spearman` correlates the seed-mean resource curve against the seed-mean performance
curve across the coupling sweep. That is the quantification of the visual comparison the
figure invites, and nothing more: the points are samples of one swept curve rather than
independent draws, so it carries no p-value and no confidence interval.

`per_coupling_spearman` correlates resource against performance across the seeds *at one
coupling*, where the coupling strength -- which drives both quantities -- is held exactly
constant instead of swept. `fisher_pool` combines those per-coupling estimates.

Every function here is pure. In particular the sign convention is the caller's business:
callers pass performance already signed so that larger is better, which for the Lorenz
panels means -NRMSE. This module never knows which panel it is serving.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

# Couplings whose seed-mean resource falls below this fraction of the subplot's peak are
# excluded from the pool. Below it the DV negativity is ~1e-7, which is numerical noise,
# and a correlation against numerical noise measures nothing.
PRESENCE_FRAC = 0.01

# arctanh diverges at |rho| = 1. A saturated estimate is dropped from the pool rather than
# clipped, so a pooled value can never be manufactured by a clip.
_MAX_ABS_RHO = 0.999

# Fisher's z has variance 1/(n-3) for a Pearson correlation, but these are *Spearman*
# coefficients, whose z-variance carries an extra factor: Fieller, Hartley and Pearson,
# Biometrika 44, 470 (1957). Using the Pearson value here would make every interval about
# 3% too narrow.
SPEARMAN_VARIANCE_FACTOR = 1.06


def _spearman_or_nan(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman correlation, or nan when either input has no spread to rank."""
    if np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return float("nan")
    return float(spearmanr(first, second).statistic)


def sweep_spearman(
    resource_means: Sequence[float], performance_means: Sequence[float]
) -> float:
    """Rank correlation of the two seed-mean curves across the coupling sweep."""
    resource = np.asarray(resource_means, dtype=float)
    performance = np.asarray(performance_means, dtype=float)
    if resource.shape != performance.shape:
        raise ValueError(
            f"resource has shape {resource.shape} but performance has "
            f"{performance.shape}; both must be one value per coupling"
        )
    if resource.ndim != 1 or resource.size < 3:
        raise ValueError(f"need at least 3 couplings, got {resource.size}")
    return _spearman_or_nan(resource, performance)


def per_coupling_spearman(
    pairs_by_coupling: Mapping[float, Sequence[tuple[float, float]]],
) -> dict[float, float]:
    """One rank correlation per coupling, taken across that coupling's seeds.

    Keys are coupling strengths; values are (resource, performance) pairs, one per seed.
    A coupling whose resource does not vary across seeds yields nan rather than raising,
    because that is a real state of the data and the caller pools around it.
    """
    out: dict[float, float] = {}
    for coupling, pairs in pairs_by_coupling.items():
        values = np.asarray(pairs, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(
                f"coupling {coupling}: expected (n_seeds, 2) pairs, got shape "
                f"{values.shape}"
            )
        out[float(coupling)] = _spearman_or_nan(values[:, 0], values[:, 1])
    return out


def fisher_pool(rhos: Sequence[float], n_seeds: int) -> tuple[float, float, int]:
    """Pools per-coupling correlations through Fisher's z-transform.

    z = arctanh(rho) is variance-stabilising and approximately normal, so the mean is
    taken in z-space and mapped back with tanh rather than averaging rho, which is
    bounded on [-1, 1] and skewed near the ends.

    The variance of z carries the rank-correlation factor: for a *Spearman* coefficient
    it is SPEARMAN_VARIANCE_FACTOR / (n - 3), not the Pearson value 1 / (n - 3)
    (Fieller, Hartley and Pearson, Biometrika 44, 470 (1957)).

    Returns (pooled_rho, standard_error_of_z, n_pooled). The standard error treats the
    per-coupling estimates as independent; they are not, because the same seeds recur at
    every coupling, so it is a lower bound on the uncertainty. That is disclosed in the
    manuscript's Methods rather than corrected here.
    """
    if n_seeds < 4:
        raise ValueError(f"need n_seeds >= 4 for a Fisher standard error, got {n_seeds}")
    usable = [
        float(rho)
        for rho in rhos
        if np.isfinite(rho) and abs(float(rho)) < _MAX_ABS_RHO
    ]
    count = len(usable)
    if count == 0:
        return float("nan"), float("nan"), 0
    z_values = np.arctanh(np.asarray(usable, dtype=float))
    pooled = float(np.tanh(z_values.mean()))
    standard_error = float(
        np.sqrt(SPEARMAN_VARIANCE_FACTOR / (count * (n_seeds - 3)))
    )
    return pooled, standard_error, count


def fisher_interval(rho: float, se: float, z: float = 1.96) -> tuple[float, float]:
    """Two-sided confidence interval for a pooled correlation, in rho space."""
    if not np.isfinite(rho) or not np.isfinite(se):
        return float("nan"), float("nan")
    centre = np.arctanh(rho)
    return float(np.tanh(centre - z * se)), float(np.tanh(centre + z * se))


def resource_present(
    resource_means: Sequence[float], frac: float = PRESENCE_FRAC
) -> np.ndarray:
    """Boolean mask of the couplings at which the resource is numerically real."""
    resource = np.asarray(resource_means, dtype=float)
    peak = np.nanmax(resource) if resource.size else float("nan")
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros(resource.shape, dtype=bool)
    return resource > frac * peak


def peak_offset_decades(
    couplings: Sequence[float],
    resource_means: Sequence[float],
    performance_means: Sequence[float],
) -> float:
    """Decades between the resource's peak coupling and performance's best coupling.

    Negative means performance peaks at the weaker coupling. Performance is assumed
    already signed so that larger is better, so both peaks are argmaxes.
    """
    coupling = np.asarray(couplings, dtype=float)
    if coupling.size == 0 or np.any(coupling <= 0.0):
        raise ValueError("couplings must all be positive; the offset is in decades")
    resource = np.asarray(resource_means, dtype=float)
    performance = np.asarray(performance_means, dtype=float)
    if not (coupling.shape == resource.shape == performance.shape):
        raise ValueError("couplings, resource and performance must have equal shape")
    best_performance = coupling[int(np.argmax(performance))]
    peak_resource = coupling[int(np.argmax(resource))]
    return float(np.log10(best_performance) - np.log10(peak_resource))
