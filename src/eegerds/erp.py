"""Event-related potentials: averaging, ordered peak search and amplitude measurement.

Peak picking follows two rules that keep it honest:

* **Ordered search.** Each component is searched in an explicit time window, and later components
  are searched *relative to earlier ones* (a P300 can only come after the N200 it follows), so a
  noisy trace cannot give an N400 before the P300.
* **Explicit failure.** If the window holds no samples, or the component it depends on was not
  found, the result is ``found=False`` with ``NaN`` values instead of a stale number. A peak
  that sits on the edge of its window is flagged (``at_edge``): the true extremum is probably
  outside the window.

Amplitudes for statistics are measured as the **mean in a short window around the group-level
peak latency**, per subject. Picking each subject's own extremum is biased upwards by noise
(the maximum of a noisy trace is always positive); a fixed latency is not. Choose that latency
from an average that is blind to the contrast you will test (all conditions pooled).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .data import Epochs

__all__ = [
    "DEFAULT_COMPONENTS",
    "GroupErp",
    "Peak",
    "PeakSpec",
    "find_group_peaks",
    "find_peaks_ordered",
    "group_erp",
    "mean_amplitude",
    "measure_amplitudes",
]


@dataclass(frozen=True)
class PeakSpec:
    """Where to look for one component (times in seconds after the event)."""

    name: str
    polarity: Literal["neg", "pos"]
    start: float
    stop: float
    after: str | None = None  # name of an earlier component this one must follow
    min_gap: float = 0.0  # earliest latency = latency(after) + min_gap
    max_gap: float | None = None  # latest latency = latency(after) + max_gap

    def __post_init__(self) -> None:
        if self.polarity not in ("neg", "pos"):
            raise ValueError("polarity must be 'neg' or 'pos'")
        if not self.start < self.stop:
            raise ValueError(f"{self.name}: start must be smaller than stop")
        if self.max_gap is not None and self.max_gap < self.min_gap:
            raise ValueError(f"{self.name}: max_gap must not be smaller than min_gap")


DEFAULT_COMPONENTS: tuple[PeakSpec, ...] = (
    PeakSpec("N200", "neg", 0.12, 0.30),
    PeakSpec("P300", "pos", 0.20, 0.55, after="N200", min_gap=0.02, max_gap=0.27),
    PeakSpec("N400", "neg", 0.30, 0.70, after="P300", min_gap=0.02, max_gap=0.20),
)


@dataclass(frozen=True)
class Peak:
    name: str
    latency: float  # seconds; NaN when not found
    amplitude: float  # volts; NaN when not found
    found: bool
    at_edge: bool = False


def find_peaks_ordered(
    times: np.ndarray, erp: np.ndarray, specs: Sequence[PeakSpec] = DEFAULT_COMPONENTS
) -> list[Peak]:
    """Peaks of one ERP trace (1-D), searched in the order of ``specs``."""
    times = np.asarray(times, dtype=float)
    erp = np.asarray(erp, dtype=float)
    if erp.shape != times.shape or erp.ndim != 1:
        raise ValueError("times and erp must be 1-D arrays of the same length")
    known: dict[str, Peak] = {}
    for spec in specs:
        if spec.name in known:
            raise ValueError(f"duplicate component name {spec.name!r}")
        lo, hi = spec.start, spec.stop
        usable = True
        if spec.after is not None:
            prev = known.get(spec.after)
            if prev is None:
                raise ValueError(
                    f"{spec.name} follows {spec.after!r}, which is not listed before it"
                )
            if prev.found:
                lo = max(lo, prev.latency + spec.min_gap)
                if spec.max_gap is not None:
                    hi = min(hi, prev.latency + spec.max_gap)
            else:
                usable = False
        window = (
            (times >= lo - 1e-9) & (times <= hi + 1e-9) if usable else np.zeros_like(times, bool)
        )
        if window.sum() < 3 or not np.all(np.isfinite(erp[window])):
            known[spec.name] = Peak(spec.name, np.nan, np.nan, False)
            continue
        segment = erp[window]
        idx = int(np.argmax(segment) if spec.polarity == "pos" else np.argmin(segment))
        position = np.flatnonzero(window)[idx]
        known[spec.name] = Peak(
            spec.name,
            float(times[position]),
            float(erp[position]),
            True,
            at_edge=idx in (0, len(segment) - 1),
        )
    return [known[s.name] for s in specs]


def mean_amplitude(times: np.ndarray, erp: np.ndarray, latency: float, half_window: float) -> float:
    """Mean of ``erp`` in ``[latency - half_window, latency + half_window]`` (NaN if empty)."""
    times = np.asarray(times, dtype=float)
    mask = (times >= latency - half_window - 1e-9) & (times <= latency + half_window + 1e-9)
    return float(np.mean(np.asarray(erp)[mask])) if mask.any() else float("nan")


@dataclass(frozen=True, eq=False)
class GroupErp:
    """Per-subject ERPs (mean over each subject's epochs) on a common time axis."""

    times: np.ndarray  # (n_times,)
    values: np.ndarray  # (n_subjects, n_channels, n_times), volts
    subjects: tuple[str, ...]
    ch_names: tuple[str, ...]

    @property
    def grand(self) -> np.ndarray:
        """Average over subjects (each subject counts once), ``(n_channels, n_times)``."""
        return self.values.mean(axis=0)

    def channel(self, name: str) -> np.ndarray:
        if name not in self.ch_names:
            raise KeyError(f"channel {name!r} not found; available: {list(self.ch_names)}")
        return self.values[:, self.ch_names.index(name), :]


def group_erp(
    epochs_per_subject: Sequence[Epochs],
    tags: Sequence[str] = (),
    baseline: tuple[float, float] | None = None,
) -> GroupErp:
    """Average each subject's epochs (optionally only those with ``tags``), then stack.

    Averaging within subjects first means a subject with more clean epochs does not weigh more.
    """
    if not epochs_per_subject:
        raise ValueError("no subjects")
    first = epochs_per_subject[0]
    rows, subjects = [], []
    for ep in epochs_per_subject:
        if (ep.fs, ep.tmin, ep.n_times, ep.ch_names) != (
            first.fs,
            first.tmin,
            first.n_times,
            first.ch_names,
        ):
            raise ValueError("subjects differ in sampling rate, time axis or channels")
        sel = ep.select(*tags) if tags else ep
        if sel.n_epochs == 0:
            raise ValueError(f"subject {ep.subject or '?'} has no epochs with tags {tuple(tags)}")
        if baseline is not None:
            sel = sel.baseline_corrected(baseline)
        rows.append(sel.data.mean(axis=0))
        subjects.append(ep.subject)
    return GroupErp(first.times, np.stack(rows), tuple(subjects), first.ch_names)


def find_group_peaks(
    group: GroupErp,
    channels: Sequence[str],
    specs: Sequence[PeakSpec] = DEFAULT_COMPONENTS,
) -> dict[str, list[Peak]]:
    """Ordered peaks of the grand average, per channel."""
    grand = group.grand
    out = {}
    for name in channels:
        if name not in group.ch_names:
            raise KeyError(f"channel {name!r} not found")
        out[name] = find_peaks_ordered(group.times, grand[group.ch_names.index(name)], specs)
    return out


def measure_amplitudes(
    group: GroupErp,
    peaks: Mapping[str, Sequence[Peak]],
    half_window: float = 0.025,
) -> pd.DataFrame:
    """Per-subject mean amplitude around each group-level peak latency.

    Columns: ``subject, channel, component, latency, amplitude`` (seconds, volts). Components
    that were not found yield ``NaN`` amplitudes.
    """
    rows = []
    for channel, plist in peaks.items():
        j = group.ch_names.index(channel)
        for peak in plist:
            for s, subject in enumerate(group.subjects):
                amp = (
                    mean_amplitude(group.times, group.values[s, j], peak.latency, half_window)
                    if peak.found
                    else float("nan")
                )
                rows.append((subject, channel, peak.name, peak.latency, amp))
    return pd.DataFrame(rows, columns=["subject", "channel", "component", "latency", "amplitude"])
