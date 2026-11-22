"""SOBI blind source separation and removal of ocular components.

SOBI (second-order blind identification, Belouchrani et al. 1997) separates sources that have
*different autocorrelation structure*: it whitens the data, estimates the covariance matrices at
several time lags, and finds the orthogonal rotation that diagonalises all of them at once
(approximate joint diagonalisation with Jacobi rotations, Cardoso & Souloumiac 1996). Unlike
FastICA it uses the temporal structure instead of non-Gaussianity, and it is deterministic: the
same data always give the same components.

This implementation works on real data with *symmetrised* lagged covariances, so every rotation
is real. Components come out with unit variance, sorted by how much of the data's variance they
carry. The sign and scale of a component are arbitrary, as in every ICA.

``remove_eog_components`` correlates every component with an EOG trace and zeroes the ones that
follow it. Unlike a "drop the best match no matter what" rule, it only removes components whose
absolute correlation reaches ``min_abs_r``, so a recording without blinks is left alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["EogRemoval", "SobiResult", "remove_eog_components", "sobi"]


@dataclass(frozen=True, eq=False)
class SobiResult:
    mixing: np.ndarray  # (n_channels, n_components): x ~= mean + mixing @ sources
    sources: np.ndarray  # (n_components, n_samples), unit variance
    unmixing: np.ndarray  # (n_components, n_channels): sources = unmixing @ (x - mean)
    mean: np.ndarray  # (n_channels,)
    n_sweeps: int
    converged: bool


def _joint_diagonalise(
    mats: np.ndarray, tol: float, max_sweeps: int
) -> tuple[np.ndarray, int, bool]:
    """Orthogonal ``V`` minimising the off-diagonal energy of ``V.T @ M_k @ V`` for all ``k``.

    ``mats`` has shape ``(K, m, m)`` and holds symmetric matrices; it is modified in place.
    Jacobi sweeps over all index pairs; each pair is rotated by the closed-form angle of
    Cardoso & Souloumiac (real case): with ``g_k = (a_pp - a_qq, 2 a_pq)``, the leading
    eigenvector of ``sum_k g_k g_k^T`` gives ``(cos 2t, sin 2t)``.
    """
    m = mats.shape[1]
    v = np.eye(m)
    for sweep in range(1, max_sweeps + 1):
        rotated = False
        for p in range(m - 1):
            for q in range(p + 1, m):
                g = np.stack(
                    [mats[:, p, p] - mats[:, q, q], mats[:, p, q] + mats[:, q, p]]
                )  # (2, K)
                _, vecs = np.linalg.eigh(g @ g.T)
                x, y = vecs[:, -1]
                if x < 0:
                    x, y = -x, -y
                c = math.sqrt(0.5 + 0.5 * x)
                s = 0.5 * y / c
                if abs(s) <= tol:
                    continue
                rotated = True
                rot = np.array([[c, -s], [s, c]])
                pair = [p, q]
                mats[:, :, pair] = mats[:, :, pair] @ rot
                mats[:, pair, :] = rot.T @ mats[:, pair, :]
                v[:, pair] = v[:, pair] @ rot
        if not rotated:
            return v, sweep, True
    return v, max_sweeps, False


def sobi(
    x: np.ndarray,
    lags: Sequence[int] | None = None,
    n_components: int | None = None,
    max_sweeps: int = 200,
    tol: float | None = None,
) -> SobiResult:
    """Separate ``x`` (``(n_channels, n_samples)``) into temporally distinct components.

    ``lags`` are sample lags of the covariance matrices (default ``1..min(100, n/3)``);
    ``n_components`` keeps that many principal directions before separating (default: every
    direction with non-negligible variance).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be (n_channels, n_samples)")
    m, n = x.shape
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN or infinite values")
    if n < 10 * m:
        raise ValueError(f"too few samples ({n}) for {m} channels")
    mean = x.mean(axis=1)
    xc = x - mean[:, None]

    # PCA whitening
    cov = xc @ xc.T / (n - 1)
    w, e = np.linalg.eigh(cov)
    w, e = w[::-1], e[:, ::-1]
    if w[0] <= 0:
        raise ValueError("all channels are constant")
    rank = int(np.sum(w > w[0] * 1e-10))
    k = rank if n_components is None else min(n_components, rank)
    if k < 1:
        raise ValueError("n_components must be at least 1")
    whiten = (e[:, :k] / np.sqrt(w[:k])).T  # (k, m)
    z = whiten @ xc

    max_lag = min(100, n // 3)
    lag_list = list(range(1, max_lag + 1)) if lags is None else [int(lag) for lag in lags]
    if not lag_list or min(lag_list) < 1 or max(lag_list) >= n:
        raise ValueError("lags must be positive and shorter than the recording")
    mats = np.empty((len(lag_list), k, k))
    for i, lag in enumerate(lag_list):
        r = z[:, lag:] @ z[:, :-lag].T / (n - lag)
        mats[i] = 0.5 * (r + r.T)

    threshold = 1.0 / math.sqrt(n) / 100.0 if tol is None else tol
    v, sweeps, converged = _joint_diagonalise(mats, threshold, max_sweeps)

    sources = v.T @ z
    mixing = (e[:, :k] * np.sqrt(w[:k])) @ v  # pinv(whiten) @ v
    order = np.argsort(-(mixing**2).sum(axis=0))
    mixing, sources, v = mixing[:, order], sources[order], v[:, order]
    return SobiResult(
        mixing=mixing,
        sources=sources,
        unmixing=v.T @ whiten,
        mean=mean,
        n_sweeps=sweeps,
        converged=converged,
    )


@dataclass(frozen=True, eq=False)
class EogRemoval:
    cleaned: np.ndarray  # (n_channels, n_samples)
    correlations: np.ndarray  # (n_components,) Pearson r between each component and the EOG
    removed: tuple[int, ...]  # indices of the components that were zeroed
    sobi: SobiResult


def remove_eog_components(
    eeg: np.ndarray,
    eog: np.ndarray,
    n_remove: int = 1,
    min_abs_r: float = 0.3,
    lags: Sequence[int] | None = None,
) -> EogRemoval:
    """Remove the components of ``eeg`` that follow the ``eog`` trace.

    Each channel is scaled to unit standard deviation before the separation (so that large
    frontal channels do not dominate) and scaled back after the reconstruction. Up to
    ``n_remove`` components are zeroed, in order of decreasing ``|r|``, and only those with
    ``|r| >= min_abs_r``.
    """
    eeg = np.asarray(eeg, dtype=float)
    eog = np.asarray(eog, dtype=float)
    if eeg.ndim != 2:
        raise ValueError("eeg must be (n_channels, n_samples)")
    if eog.shape != (eeg.shape[1],):
        raise ValueError("eog must be 1-D with one value per sample of eeg")
    if n_remove < 0:
        raise ValueError("n_remove must be >= 0")
    scale = eeg.std(axis=1)
    if np.any(scale == 0):
        raise ValueError("a channel is constant")
    res = sobi(eeg / scale[:, None], lags=lags)

    s = res.sources - res.sources.mean(axis=1, keepdims=True)
    ref = eog - eog.mean()
    denom = np.sqrt((s**2).sum(axis=1) * (ref**2).sum())
    r = np.divide(s @ ref, denom, out=np.zeros(s.shape[0]), where=denom > 0)
    ranked = np.argsort(-np.abs(r))
    removed = tuple(int(i) for i in ranked[:n_remove] if abs(r[i]) >= min_abs_r)

    keep = [i for i in range(s.shape[0]) if i not in removed]
    recon = res.mean[:, None] + res.mixing[:, keep] @ res.sources[keep]
    cleaned = recon * scale[:, None]
    return EogRemoval(cleaned=cleaned, correlations=r, removed=removed, sobi=res)
