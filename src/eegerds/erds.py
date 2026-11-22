"""Event-related (de)synchronisation: ERD/ERS in percent of a reference power.

    ERD/ERS(t) = (P(t) - R) / R * 100

where ``P(t)`` is the band power around time ``t`` and ``R`` the mean power in a reference
interval before the event. Negative values are ERD (power decrease), positive values are ERS
(power increase, "rebound").

Two ways of getting ``P(t)`` from epochs ``x_e(t)`` that were band-passed (zero phase):

``"intertrial_variance"`` (default)
    the variance across epochs at every sample. It subtracts the across-epoch mean, i.e. the
    phase-locked (evoked) part, so only the induced, non-phase-locked activity is measured
    (Kalcher & Pfurtscheller, 1995).
``"power"``
    the mean of ``x_e(t)**2`` across epochs (the classical Pfurtscheller & Aranibar method). It
    also contains the evoked response, so a phase-locked burst inside the band shows up as ERS.

The power is then averaged in windows (default 0.5 s, 50 % overlap), expressed relative to the
reference, and may be smoothed over neighbouring windows for display. Statistics should use the
unsmoothed, preferably non-overlapping, windows (``overlap=0``): overlap and smoothing make
neighbouring values dependent.

Two practical points, both measured on noise with a known flat spectrum:

* **Edges.** Band-passing each epoch separately distorts its first and last half second, because
  a narrow band has a long memory. With one second of odd-extension padding (what this module
  uses) the power in the last 0.5 s is off by about -4 % instead of -18 %. For exact results
  filter the *continuous* recording in the band of interest (``bandpass_recording``), epoch it,
  and pass ``prefiltered=True``; and keep the reference interval away from the epoch edge if you
  can. ``prefiltered=True`` on data that are *not* limited to the band measures broadband power,
  and the ERD comes out less than half as deep.
* **Reference.** Every window is divided by the same reference power, so any error in it shifts
  the whole curve. A narrow-band rhythm averaged over a one-second reference holds only a few
  independent cycles: prefer a longer reference and many epochs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .data import Epochs
from .filters import bandpass

__all__ = [
    "ErdsConfig",
    "ErdsCurve",
    "GroupErds",
    "erds_curve",
    "erds_group",
    "sample_power",
    "window_centres",
    "windowed_mean",
]

Method = Literal["intertrial_variance", "power"]


@dataclass(frozen=True)
class ErdsConfig:
    """Settings of the ERD/ERS estimate (defaults follow common motor-imagery practice)."""

    band: tuple[float, float] = (8.0, 13.0)  # Hz; mu
    order: int = 4  # Butterworth band-pass order (applied forwards and backwards)
    window: float = 0.5  # seconds
    overlap: float = 0.5  # fraction of a window shared with the next one, in [0, 1)
    reference: tuple[float, float] = (-2.0, -1.0)  # seconds, relative to the event
    method: Method = "intertrial_variance"
    #: True only if the epochs were already band-passed *to* ``band`` (for instance with
    #: :func:`~eegerds.filters.bandpass_recording` before epoching). With broadband data it
    #: silently measures broadband power and dilutes the ERD.
    prefiltered: bool = False

    def __post_init__(self) -> None:
        if not self.band[0] < self.band[1]:
            raise ValueError("band must be (low, high) with low < high")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.method not in ("intertrial_variance", "power"):
            raise ValueError("method must be 'intertrial_variance' or 'power'")
        if not self.reference[0] < self.reference[1]:
            raise ValueError("reference must be (start, stop) with start < stop")


def _window_geometry(n_times: int, fs: float, window: float, overlap: float) -> tuple[int, int]:
    length = round(window * fs)
    step = max(1, round(length * (1.0 - overlap)))
    if length < 2:
        raise ValueError("window is shorter than two samples")
    if length > n_times:
        raise ValueError(f"window ({window} s) is longer than the epoch")
    return length, step


def windowed_mean(
    x: np.ndarray, fs: float, window: float, overlap: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``x`` (along the last axis) in sliding windows.

    Returns ``(starts, means)``: the first sample of every window and the window means, shape
    ``x.shape[:-1] + (n_windows,)``.
    """
    x = np.asarray(x, dtype=float)
    length, step = _window_geometry(x.shape[-1], fs, window, overlap)
    starts = np.arange(0, x.shape[-1] - length + 1, step)
    csum = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, axis=-1)], axis=-1)
    means = (csum[..., starts + length] - csum[..., starts]) / length
    return starts, means


def window_centres(epochs: Epochs, window: float, overlap: float) -> np.ndarray:
    """Centre time (s) of each window, relative to the event."""
    length, step = _window_geometry(epochs.n_times, epochs.fs, window, overlap)
    starts = np.arange(0, epochs.n_times - length + 1, step)
    return epochs.tmin + (starts + (length - 1) / 2) / epochs.fs


def sample_power(epochs: Epochs, config: ErdsConfig | None = None) -> np.ndarray:
    """Band power at every sample, shape ``(n_channels, n_times)`` (see the module docstring)."""
    cfg = config or ErdsConfig()
    if not np.all(np.isfinite(epochs.data)):
        raise ValueError("epochs contain NaN or infinite values; reject or repair them first")
    if cfg.method == "intertrial_variance" and epochs.n_epochs < 2:
        raise ValueError("the inter-trial variance method needs at least 2 epochs")
    if epochs.n_epochs < 1:
        raise ValueError("no epochs")
    x = epochs.data
    if not cfg.prefiltered:
        pad = min(epochs.n_times - 1, round(epochs.fs))  # one second of odd extension
        x = bandpass(x, epochs.fs, cfg.band[0], cfg.band[1], order=cfg.order, axis=-1, padlen=pad)
    if cfg.method == "intertrial_variance":
        return x.var(axis=0, ddof=1)
    return (x**2).mean(axis=0)


@dataclass(frozen=True, eq=False)
class ErdsCurve:
    """ERD/ERS of one subject (and one set of epochs): values in percent per window."""

    times: np.ndarray  # (n_windows,) window centres, seconds
    values: np.ndarray  # (n_channels, n_windows), percent
    ch_names: tuple[str, ...]
    n_epochs: int
    config: ErdsConfig
    subject: str = ""

    def channel(self, name: str) -> np.ndarray:
        return self.values[self._index(name)]

    def _index(self, name: str) -> int:
        if name not in self.ch_names:
            raise KeyError(f"channel {name!r} not found; available: {list(self.ch_names)}")
        return self.ch_names.index(name)

    def smoothed(self, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Moving average over ``n`` neighbouring windows (``"valid"``: no edge bias).

        Returns ``(times, values)``, both shortened by ``n - 1``.
        """
        return _moving_average(self.times, self.values, n)


@dataclass(frozen=True, eq=False)
class GroupErds:
    """ERD/ERS of several subjects on a common time axis."""

    times: np.ndarray  # (n_windows,)
    values: np.ndarray  # (n_subjects, n_channels, n_windows), percent
    subjects: tuple[str, ...]
    ch_names: tuple[str, ...]
    config: ErdsConfig

    @property
    def n_subjects(self) -> int:
        return int(self.values.shape[0])

    def channel(self, name: str) -> np.ndarray:
        """``(n_subjects, n_windows)`` values of one channel."""
        if name not in self.ch_names:
            raise KeyError(f"channel {name!r} not found; available: {list(self.ch_names)}")
        return self.values[:, self.ch_names.index(name), :]

    def mean(self) -> np.ndarray:
        """Mean over subjects, ``(n_channels, n_windows)`` (each subject counts once)."""
        return self.values.mean(axis=0)

    def sem(self) -> np.ndarray:
        """Standard error of the mean over subjects."""
        if self.n_subjects < 2:
            return np.full(self.values.shape[1:], np.nan)
        return self.values.std(axis=0, ddof=1) / np.sqrt(self.n_subjects)

    def smoothed_mean(self, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
        return _moving_average(self.times, self.mean(), n)


def _moving_average(times: np.ndarray, values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be at least 1")
    if n > values.shape[-1]:
        raise ValueError("smoothing length exceeds the number of windows")
    kernel = np.ones(n) / n
    smooth = np.apply_along_axis(lambda row: np.convolve(row, kernel, "valid"), -1, values)
    return np.convolve(times, kernel, "valid"), smooth


def erds_curve(epochs: Epochs, config: ErdsConfig | None = None) -> ErdsCurve:
    """ERD/ERS (%) of every channel of ``epochs`` (one subject, one condition)."""
    cfg = config or ErdsConfig()
    power = sample_power(epochs, cfg)
    i0, i1 = epochs.sample_range(*cfg.reference)
    reference = power[:, i0 : i1 + 1].mean(axis=1)
    if np.any(reference <= 0):
        bad = [n for n, r in zip(epochs.ch_names, reference, strict=True) if r <= 0]
        raise ValueError(f"reference power is zero in channel(s) {bad}; flat signal?")
    _, win_power = windowed_mean(power, epochs.fs, cfg.window, cfg.overlap)
    values = (win_power - reference[:, None]) / reference[:, None] * 100.0
    return ErdsCurve(
        times=window_centres(epochs, cfg.window, cfg.overlap),
        values=values,
        ch_names=epochs.ch_names,
        n_epochs=epochs.n_epochs,
        config=cfg,
        subject=epochs.subject,
    )


def erds_group(epochs_per_subject: Sequence[Epochs], config: ErdsConfig | None = None) -> GroupErds:
    """One :func:`erds_curve` per subject, stacked. All subjects must share the time axis."""
    cfg = config or ErdsConfig()
    if not epochs_per_subject:
        raise ValueError("no subjects")
    first = epochs_per_subject[0]
    for ep in epochs_per_subject[1:]:
        if (ep.fs, ep.tmin, ep.n_times, ep.ch_names) != (
            first.fs,
            first.tmin,
            first.n_times,
            first.ch_names,
        ):
            raise ValueError("subjects differ in sampling rate, time axis or channels")
    curves = [erds_curve(ep, cfg) for ep in epochs_per_subject]
    return GroupErds(
        times=curves[0].times,
        values=np.stack([c.values for c in curves]),
        subjects=tuple(c.subject for c in curves),
        ch_names=first.ch_names,
        config=replace(cfg),
    )
