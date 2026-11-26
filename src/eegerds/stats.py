"""Paired statistics across subjects, with multiple-comparison control.

The unit of analysis is the **subject**: every test compares one value per subject under two
conditions (or two channels). With few subjects two things matter, and both are reported:

* the smallest p-value an exact two-sided paired test can reach is ``2 / 2**n`` (``p_floor``
  column). With ``n = 8`` it is 0.0078: a Bonferroni correction over more than 6 windows could
  never reject anything, and Benjamini-Hochberg needs several windows to sit at that floor at
  the same time (at least 6 of 32). That is a property of the test, not of the effect;
* testing many windows at ``alpha = 0.01`` uncorrected inflates false positives. Two corrections
  are offered: Benjamini-Hochberg FDR on the Wilcoxon p-values, and a **sign-flip max-T**
  permutation test (Nichols & Holmes, 2002) that controls the family-wise error rate and is exact
  for ``n <= 12`` (all ``2**n`` sign patterns are enumerated).

Methods
-------
``"wilcoxon_fdr"`` (default)   Wilcoxon signed-rank per window, BH-adjusted over the family.
``"wilcoxon"``                 the same, uncorrected (``p_adj == p_raw``).
``"signflip_maxt"``            paired-t statistics, max-T family-wise adjusted p-values.
``"cluster"``                  cluster-based permutation test over neighbouring windows (Maris &
                               Oostenveld, 2007); one family-wise p-value per cluster.

The family is everything tested in one call (all channels and windows of the table).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from .erds import GroupErds

__all__ = [
    "METHODS",
    "compare_amplitudes",
    "compare_channels",
    "compare_conditions",
    "compare_series",
    "fdr_bh",
    "paired_wilcoxon",
    "signflip_cluster",
    "signflip_maxt",
]

Method = Literal["wilcoxon_fdr", "wilcoxon", "signflip_maxt", "cluster"]
METHODS = ("wilcoxon_fdr", "wilcoxon", "signflip_maxt", "cluster")


def fdr_bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN entries are ignored and stay NaN)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    values = p[ok]
    m = values.size
    if m == 0:
        return out
    order = np.argsort(values)
    scaled = values[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    result = np.empty(m)
    result[order] = adjusted
    out[ok] = result
    return out


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided Wilcoxon signed-rank test per column of ``a`` and ``b`` (``(n_subjects, k)``).

    Returns ``(statistic, p, effect)``. ``effect`` is the matched-pairs rank-biserial
    correlation ``(W+ - W-) / (W+ + W-)`` in ``[-1, 1]`` (positive: ``a > b``). Subjects with a
    NaN in a column are dropped from that column; a column with no non-zero difference gets
    ``p = 1`` and ``effect = 0``; a column with no valid subject gets NaN.
    """
    a = np.atleast_2d(np.asarray(a, dtype=float).T).T
    b = np.atleast_2d(np.asarray(b, dtype=float).T).T
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    k = a.shape[1]
    stat, p, eff = np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan)
    for j in range(k):
        d = a[:, j] - b[:, j]
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        d = d[d != 0]
        if d.size == 0:
            stat[j], p[j], eff[j] = 0.0, 1.0, 0.0
            continue
        ranks = stats.rankdata(np.abs(d))
        w_pos, w_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
        eff[j] = (w_pos - w_neg) / (w_pos + w_neg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # normal approximation with ties
            res = stats.wilcoxon(d, alternative="two-sided")
        stat[j], p[j] = res.statistic, res.pvalue
    return stat, p, eff


def _t_from_sums(total: np.ndarray, sumsq: np.ndarray, n: int) -> np.ndarray:
    """One-sample t from the sum and the sum of squares of each column.

    Zero spread is decided *relative to the size of the data*: when every value equals the mean,
    ``sumsq - n * mean**2`` is the difference of two nearly equal numbers and its rounding noise
    would give an arbitrary t (and, for the sign pattern that reproduces the data, p = 0, which a
    permutation test can never have). Such columns get ``t = +-inf`` (or 0 when the mean is 0).
    """
    mean = total / n
    spread = sumsq - n * mean**2  # sum of squares about the mean
    degenerate = spread <= 1e-12 * sumsq
    var = np.where(degenerate, 1.0, spread / (n - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        regular = mean / np.sqrt(var / n)
        constant = np.where(mean != 0, np.sign(mean) * np.inf, 0.0)
    return np.where(degenerate, constant, regular)


def signflip_maxt(
    diff: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
    max_exhaustive: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Family-wise adjusted p-values from the sign-flip max-T permutation test.

    ``diff`` is ``(n_subjects, k)``: paired differences for ``k`` simultaneous tests, without
    NaN. Under the null the differences are symmetric around zero, so flipping the sign of a
    subject's whole row is a valid relabelling. For each relabelling the largest ``|t|`` over
    the ``k`` tests is recorded; test ``j``'s adjusted p is the share of relabellings whose max
    reaches ``|t_j|``. With ``n_subjects <= max_exhaustive`` all ``2**n`` patterns are used
    (exact); otherwise ``n_perm`` random patterns with the usual ``(1 + count) / (1 + n_perm)``.

    Returns ``(t, p_adj)``.
    """
    d = np.asarray(diff, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    n, k = d.shape
    if n < 2:
        raise ValueError("need at least 2 subjects")
    if not np.all(np.isfinite(d)):
        raise ValueError("diff must not contain NaN or infinite values")
    sumsq = (d**2).sum(axis=0)
    t_obs = _t_from_sums(d.sum(axis=0), sumsq, n)
    if n <= max_exhaustive:
        bits = (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1
        signs = (2 * bits - 1).astype(float)
        extra = 0
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        extra = 1
    max_abs = np.abs(_t_from_sums(signs @ d, sumsq, n)).max(axis=1)
    reach = max_abs[:, None] >= np.abs(t_obs)[None, :] * (1 - 1e-12)
    p_adj = (reach.sum(axis=0) + extra) / (len(max_abs) + extra)
    return t_obs, p_adj


# A window whose differences are all identical (and not 0) has t = +-inf. Inside a sum over a
# cluster that would give inf - inf = NaN, so it is replaced by a value that no real t comes near
# but that keeps every sum finite: such a window still dominates any cluster it belongs to.
_T_CAP = 1e9
_CHUNK_CELLS = 2_000_000  # t-values held at once while the relabellings are processed


def _cap(t: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(t, nan=0.0), -_T_CAP, _T_CAP)


def _largest_cluster(t: np.ndarray, threshold: float) -> np.ndarray:
    """Largest cluster mass along the last axis of ``t``: the sum of ``|t|`` over a run.

    A cluster is a run of neighbouring windows that all exceed ``threshold`` in the same
    direction; a positive run and a negative run next to each other are two clusters. Works on
    any leading shape, so every relabelling is handled at once.
    """
    best = np.zeros(t.shape[:-1])
    for direction in (1.0, -1.0):
        v = direction * t
        inside = v > threshold
        total = np.cumsum(np.where(inside, v, 0.0), axis=-1)
        # What the running total was at the last window outside a cluster is what a new run
        # starts from; the total minus that is the mass of the run so far.
        start = np.maximum.accumulate(np.where(inside, 0.0, total), axis=-1)
        best = np.maximum(best, (total - start).max(axis=-1))
    return best


def find_clusters(t: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """``(first, last, mass)`` of every cluster of one series of ``t`` values, in time order."""
    found: list[tuple[int, int, float]] = []
    first, direction, mass = 0, 0, 0.0
    for j, value in enumerate(np.append(np.asarray(t, dtype=float), 0.0)):  # the 0 ends a run
        now = 1 if value > threshold else -1 if value < -threshold else 0
        if now != direction:
            if direction:
                found.append((first, j - 1, mass))
            first, direction, mass = j, now, 0.0
        if now:
            mass += float(abs(value))
    return found


def signflip_cluster(
    diff: np.ndarray,
    threshold: float,
    n_perm: int = 10000,
    seed: int = 0,
    max_exhaustive: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cluster-based permutation test (Maris & Oostenveld, 2007) with sign-flip relabelling.

    ``diff`` is ``(n_subjects, n_series, n_windows)``: paired differences, without NaN. In each
    series, neighbouring windows whose paired ``|t|`` exceeds ``threshold`` in the same
    direction form a cluster, whose mass is the sum of ``|t|`` over it. Under the null the
    differences are symmetric around zero, so flipping a subject's whole row (all series and
    windows together) is a valid relabelling; for each one the largest cluster mass over all
    series is recorded, and a cluster's adjusted p is the share of relabellings whose largest
    cluster reaches its mass. That controls the family-wise error rate over every series and
    window of the call. The permutations are exhaustive up to ``max_exhaustive`` subjects (the
    smallest p is then ``2 / 2**n``, as for ``signflip_maxt``); beyond that ``n_perm`` random
    patterns give ``(1 + count) / (1 + n_perm)``.

    Returns ``(t, p_adj, cluster, mass)``, each ``(n_series, n_windows)``: the paired t of every
    window; the adjusted p of the cluster the window belongs to (1 outside any cluster); the
    cluster number (``1, 2, ...`` in series and time order, 0 outside); and the mass of that
    cluster (0 outside).
    """
    d = np.asarray(diff, dtype=float)
    if d.ndim != 3:
        raise ValueError("diff must be (n_subjects, n_series, n_windows)")
    n, n_series, n_windows = d.shape
    if n < 2:
        raise ValueError("need at least 2 subjects")
    if not np.all(np.isfinite(d)):
        raise ValueError("diff must not contain NaN or infinite values")
    flat = d.reshape(n, n_series * n_windows)
    sumsq = (flat**2).sum(axis=0)
    t_obs = _cap(_t_from_sums(flat.sum(axis=0), sumsq, n)).reshape(n_series, n_windows)

    if n <= max_exhaustive:
        bits = (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1
        signs = (2 * bits - 1).astype(float)
        extra = 0
    else:
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_perm, n))
        extra = 1
    biggest = np.empty(len(signs))
    step = max(1, _CHUNK_CELLS // flat.shape[1])  # relabellings per chunk: bounds the memory
    for start in range(0, len(signs), step):
        t_perm = _cap(_t_from_sums(signs[start : start + step] @ flat, sumsq, n))
        t_perm = t_perm.reshape(-1, n_series, n_windows)
        biggest[start : start + step] = _largest_cluster(t_perm, threshold).max(axis=1)

    p_adj = np.ones((n_series, n_windows))
    cluster = np.zeros((n_series, n_windows), dtype=int)
    mass = np.zeros((n_series, n_windows))
    number = 0
    for i in range(n_series):
        for first, last, size in find_clusters(t_obs[i], threshold):
            number += 1
            reach = (biggest >= size * (1 - 1e-12)).sum()
            p_adj[i, first : last + 1] = (reach + extra) / (len(biggest) + extra)
            cluster[i, first : last + 1] = number
            mass[i, first : last + 1] = size
    return t_obs, p_adj, cluster, mass


def compare_series(
    a: np.ndarray,
    b: np.ndarray,
    names: Sequence[str],
    times: np.ndarray,
    window: float,
    method: Method = "wilcoxon_fdr",
    alpha: float = 0.05,
    n_perm: int = 10000,
    seed: int = 0,
    cluster_p: float = 0.05,
) -> pd.DataFrame:
    """Paired comparison of ``a`` and ``b`` for every series and window.

    ``a`` and ``b`` have shape ``(n_subjects, n_series, n_windows)``; ``names`` labels the series
    and ``times`` holds the window centres. One row per (series, window).

    With ``method="cluster"`` neighbouring windows of one series whose paired ``|t|`` is beyond
    the two-sided ``cluster_p`` threshold of the t distribution form a cluster, and the table
    gets two more columns: ``cluster`` (its number, 0 outside any cluster) and ``cluster_mass``
    (the sum of ``|t|`` over it). ``p_adj`` is then the family-wise p of the cluster the window
    belongs to (1 outside any cluster), and ``significant`` marks the windows of the clusters
    below ``alpha``: the cluster is what is significant, not each window in it.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "cluster" and not 0 < cluster_p < 1:
        raise ValueError("cluster_p must be between 0 and 1")
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("a and b must both be (n_subjects, n_series, n_windows)")
    n_sub, n_ser, n_win = a.shape
    if len(names) != n_ser or len(times) != n_win:
        raise ValueError("names/times do not match the arrays")
    flat_a = a.reshape(n_sub, n_ser * n_win)
    flat_b = b.reshape(n_sub, n_ser * n_win)
    valid = np.isfinite(flat_a) & np.isfinite(flat_b)
    n_used = valid.sum(axis=0)

    extra: dict[str, np.ndarray] = {}
    if method == "cluster":
        if not valid.all():
            raise ValueError("cluster needs complete data (no NaN)")
        threshold = float(stats.t.ppf(1 - cluster_p / 2, n_sub - 1)) if n_sub > 1 else np.inf
        t, adjusted, number, size = signflip_cluster(a - b, threshold, n_perm=n_perm, seed=seed)
        statistic, p_adj = t.reshape(-1), adjusted.reshape(-1)
        extra = {"cluster": number.reshape(-1), "cluster_mass": size.reshape(-1)}
        _, p_raw, effect = paired_wilcoxon(flat_a, flat_b)  # for the effect size only
        p_raw = np.full_like(p_raw, np.nan)  # a cluster has a p-value, a window does not
    elif method == "signflip_maxt":
        if not valid.all():
            raise ValueError("signflip_maxt needs complete data (no NaN)")
        statistic, p_adj = signflip_maxt(flat_a - flat_b, n_perm=n_perm, seed=seed)
        _, p_raw, effect = paired_wilcoxon(flat_a, flat_b)  # for the effect size only
        p_raw = np.full_like(p_raw, np.nan)  # the permutation test has no unadjusted p here
    else:
        statistic, p_raw, effect = paired_wilcoxon(flat_a, flat_b)
        p_adj = fdr_bh(p_raw) if method == "wilcoxon_fdr" else p_raw

    series = np.repeat(np.asarray(names, dtype=object), n_win)
    centre = np.tile(np.asarray(times, dtype=float), n_ser)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a = np.where(valid, flat_a, 0.0).sum(axis=0) / n_used
        mean_b = np.where(valid, flat_b, 0.0).sum(axis=0) / n_used
    table = pd.DataFrame(
        {
            "series": series,
            "time": centre,
            "t_start": centre - window / 2,
            "t_stop": centre + window / 2,
            "n": n_used,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "statistic": statistic,
            "p_raw": p_raw,
            "p_adj": p_adj,
            "effect": effect,
            "p_floor": np.minimum(1.0, 2.0 ** (1.0 - n_used.astype(float))),
            "significant": p_adj < alpha,
        }
    )
    for name, column in extra.items():
        table[name] = column
    return table


def compare_conditions(
    a: GroupErds,
    b: GroupErds,
    channels: Sequence[str],
    method: Method = "wilcoxon_fdr",
    alpha: float = 0.05,
    n_perm: int = 10000,
    seed: int = 0,
    cluster_p: float = 0.05,
) -> pd.DataFrame:
    """Same channels, two conditions (``a`` vs ``b``), paired over subjects, per window.

    ``method="cluster"`` is the cluster-based permutation test; see :func:`compare_series`.
    """
    _check_pair(a, b)
    xa = np.stack([a.channel(c) for c in channels], axis=1)
    xb = np.stack([b.channel(c) for c in channels], axis=1)
    return compare_series(
        xa, xb, list(channels), a.times, a.config.window, method, alpha, n_perm, seed, cluster_p
    )


def compare_channels(
    group: GroupErds,
    pairs: Sequence[tuple[str, str]],
    method: Method = "wilcoxon_fdr",
    alpha: float = 0.05,
    n_perm: int = 10000,
    seed: int = 0,
    cluster_p: float = 0.05,
) -> pd.DataFrame:
    """Two channels of the same condition (e.g. ``("C3", "C4")``), paired over subjects."""
    xa = np.stack([group.channel(p[0]) for p in pairs], axis=1)
    xb = np.stack([group.channel(p[1]) for p in pairs], axis=1)
    names = [f"{p[0]} - {p[1]}" for p in pairs]
    return compare_series(
        xa, xb, names, group.times, group.config.window, method, alpha, n_perm, seed, cluster_p
    )


def _check_pair(a: GroupErds, b: GroupErds) -> None:
    if a.subjects != b.subjects:
        raise ValueError("the two groups must contain the same subjects in the same order")
    if a.values.shape != b.values.shape or not np.allclose(a.times, b.times):
        raise ValueError("the two groups must share channels and time axis")


def compare_amplitudes(
    a: pd.DataFrame,
    b: pd.DataFrame,
    method: Method = "wilcoxon_fdr",
    alpha: float = 0.05,
    n_perm: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired comparison of ERP component amplitudes (tables from ``measure_amplitudes``).

    ``a`` and ``b`` hold one row per ``(subject, channel, component)``; subjects are paired by
    name. One row per ``(channel, component)`` in the result; amplitudes stay in volts.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "cluster":
        raise ValueError(
            "the cluster method needs neighbouring time windows, and amplitudes of components "
            "have none: use compare_conditions or compare_channels, or another method"
        )
    cols = ["channel", "component"]
    pa = a.pivot(index="subject", columns=cols, values="amplitude")
    pb = b.pivot(index="subject", columns=cols, values="amplitude")
    if set(pa.index) != set(pb.index) or set(pa.columns) != set(pb.columns):
        raise ValueError("the two tables must hold the same subjects, channels and components")
    pb = pb.loc[pa.index, pa.columns]
    xa, xb = pa.to_numpy(), pb.to_numpy()
    valid = np.isfinite(xa) & np.isfinite(xb)
    n_used = valid.sum(axis=0)
    if method == "signflip_maxt":
        if not valid.all():
            raise ValueError("signflip_maxt needs complete data (no NaN)")
        statistic, p_adj = signflip_maxt(xa - xb, n_perm=n_perm, seed=seed)
        _, _, effect = paired_wilcoxon(xa, xb)
        p_raw = np.full(xa.shape[1], np.nan)
    else:
        statistic, p_raw, effect = paired_wilcoxon(xa, xb)
        p_adj = fdr_bh(p_raw) if method == "wilcoxon_fdr" else p_raw
    latency = a.groupby(cols)["latency"].first().reindex(pa.columns).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a = np.where(valid, xa, 0.0).sum(axis=0) / n_used
        mean_b = np.where(valid, xb, 0.0).sum(axis=0) / n_used
    return pd.DataFrame(
        {
            "channel": [c for c, _ in pa.columns],
            "component": [k for _, k in pa.columns],
            "latency": latency,
            "n": n_used,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "statistic": statistic,
            "p_raw": p_raw,
            "p_adj": p_adj,
            "effect": effect,
            "p_floor": np.minimum(1.0, 2.0 ** (1.0 - n_used.astype(float))),
            "significant": p_adj < alpha,
        }
    )
