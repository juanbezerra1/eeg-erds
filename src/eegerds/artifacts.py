"""Epoch rejection with three explicit rules and a reason for every dropped epoch.

For each channel a threshold is set from the signal's own spread::

    threshold = sd_factor * (typical per-epoch standard deviation of that channel)

An epoch is rejected when *any* channel breaks *any* of these rules:

``run``
    more than ``max_run_percent`` of the epoch *in a row* beyond ``+threshold`` or beyond
    ``-threshold`` (a sustained deflection: drift, a blink, a movement);
``total``
    more than ``max_total_percent`` of the epoch beyond ``+threshold`` or beyond ``-threshold``
    (not necessarily in a row);
``amplitude``
    any sample with ``|x| > max_abs`` volts (a spike or a saturated amplifier);
``variance``
    *optional* (``max_window_sd_ratio``): any 0.5 s window whose standard deviation exceeds that
    many times the channel's typical window SD. The three rules above only see large, sustained
    deflections; a burst of muscle noise lasting a second is invisible to them, and this one
    catches it.

"Typical" spread is, by default, the *median* over all epochs (robust: it survives up to half the
epochs being contaminated). Pass ``reference`` (indices of epochs you know to be clean) to use
their *mean* instead. Epochs should be baseline-corrected; ``demean=True`` also removes each
epoch's mean per channel before the rules are applied.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .data import Epochs

__all__ = ["REASONS", "RejectionConfig", "RejectionReport", "longest_run", "reject_epochs"]

REASONS = ("run", "total", "amplitude", "variance")


@dataclass(frozen=True)
class RejectionConfig:
    sd_factor: float = 3.0
    max_run_percent: float = 5.0
    max_total_percent: float = 10.0
    max_abs: float = 150e-6  # volts
    demean: bool = True
    max_window_sd_ratio: float | None = None  # optional fourth rule; None switches it off
    window_sd_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.sd_factor <= 0:
            raise ValueError("sd_factor must be positive")
        if not 0 < self.max_run_percent <= 100 or not 0 < self.max_total_percent <= 100:
            raise ValueError("percentages must be in (0, 100]")
        if self.max_abs <= 0:
            raise ValueError("max_abs must be positive")
        if self.max_window_sd_ratio is not None and self.max_window_sd_ratio <= 1:
            raise ValueError("max_window_sd_ratio must be greater than 1 (or None)")
        if self.window_sd_seconds <= 0:
            raise ValueError("window_sd_seconds must be positive")


@dataclass(frozen=True, eq=False)
class RejectionReport:
    keep: np.ndarray  # (n_epochs,) bool
    reasons: tuple[tuple[str, ...], ...]  # per epoch, subset of REASONS (empty when kept)
    thresholds: np.ndarray  # (n_channels,) volts
    bad_channels: dict[str, int]  # channel -> number of epochs in which it broke a rule

    @property
    def n_rejected(self) -> int:
        return int((~self.keep).sum())

    @property
    def fraction_rejected(self) -> float:
        return self.n_rejected / len(self.keep)

    def counts(self) -> dict[str, int]:
        """Number of epochs flagged by each rule (an epoch can be flagged by several)."""
        return {r: sum(r in rs for rs in self.reasons) for r in REASONS}

    def apply(self, epochs: Epochs) -> Epochs:
        """The epochs that were kept."""
        return epochs.keep(self.keep)


def longest_run(flags: np.ndarray) -> np.ndarray:
    """Length of the longest run of ``True`` along the last axis (vectorised)."""
    flags = np.asarray(flags, dtype=bool)
    idx = np.arange(flags.shape[-1])
    last_false = np.maximum.accumulate(np.where(flags, -1, idx), axis=-1)
    return np.where(flags, idx - last_false, 0).max(axis=-1)


def _percent_to_samples(n_times: int, percent: float) -> int:
    return math.ceil(round(n_times * percent / 100.0, 9))


def _window_variance_flags(x: np.ndarray, fs: float, cfg: RejectionConfig) -> np.ndarray:
    """``(n_epochs, n_channels)``: some window is much noisier than the channel's typical one."""
    assert cfg.max_window_sd_ratio is not None
    w = max(2, round(cfg.window_sd_seconds * fs))
    n_win = x.shape[2] // w
    if n_win < 2:
        raise ValueError("epochs are too short for window_sd_seconds")
    blocks = x[:, :, : n_win * w].reshape(x.shape[0], x.shape[1], n_win, w)
    sd = blocks.std(axis=3)  # (n_epochs, n_channels, n_windows)
    typical = np.median(sd.transpose(1, 0, 2).reshape(x.shape[1], -1), axis=1)
    typical = np.maximum(typical, 1e-30)
    return (sd > cfg.max_window_sd_ratio * typical[None, :, None]).any(axis=2)


def reject_epochs(
    epochs: Epochs,
    config: RejectionConfig | None = None,
    reference: Sequence[int] | None = None,
) -> RejectionReport:
    """Apply the rules in the module docstring and report what was rejected and why."""
    cfg = config or RejectionConfig()
    if not np.all(np.isfinite(epochs.data)):
        raise ValueError("epochs contain NaN or infinite values")
    x = epochs.data
    if cfg.demean:
        x = x - x.mean(axis=2, keepdims=True)
    spread = x.std(axis=2)  # (n_epochs, n_channels)
    if reference is None:
        typical = np.median(spread, axis=0)
    else:
        ref = np.asarray(list(reference), dtype=int)
        if ref.size == 0 or ref.min() < 0 or ref.max() >= epochs.n_epochs:
            raise ValueError("reference indices must be valid, non-empty epoch indices")
        typical = spread[ref].mean(axis=0)
    thresholds = cfg.sd_factor * typical
    thr = thresholds[None, :, None]

    above, below = x > thr, x < -thr
    n_run = _percent_to_samples(epochs.n_times, cfg.max_run_percent)
    n_total = _percent_to_samples(epochs.n_times, cfg.max_total_percent)
    run = np.maximum(longest_run(above), longest_run(below)) > n_run
    total = np.maximum(above.sum(axis=2), below.sum(axis=2)) > n_total
    amplitude = (np.abs(x) > cfg.max_abs).any(axis=2)

    flags = {"run": run, "total": total, "amplitude": amplitude}  # each (n_epochs, n_channels)
    if cfg.max_window_sd_ratio is not None:
        flags["variance"] = _window_variance_flags(x, epochs.fs, cfg)
    else:
        flags["variance"] = np.zeros_like(run)
    any_flag = run | total | amplitude | flags["variance"]
    keep = ~any_flag.any(axis=1)
    reasons = tuple(tuple(r for r in REASONS if flags[r][e].any()) for e in range(epochs.n_epochs))
    bad = {
        name: int(any_flag[:, j].sum())
        for j, name in enumerate(epochs.ch_names)
        if any_flag[:, j].any()
    }
    return RejectionReport(keep=keep, reasons=reasons, thresholds=thresholds, bad_channels=bad)
