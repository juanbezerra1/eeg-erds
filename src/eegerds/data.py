"""Containers for continuous recordings and epoched data.

Conventions (the same as MNE-Python, so converting is cheap):

* signals are in **volts**;
* continuous data have shape ``(n_channels, n_samples)``;
* epoched data have shape ``(n_epochs, n_channels, n_times)``;
* each epoch carries a label such as ``"left/high"``; ``Epochs.select("left")`` keeps every
  epoch whose label contains the tag ``left`` (tags are separated by ``/``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

__all__ = ["Epochs", "Recording", "epoch"]


def _as_names(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(n) for n in names)


@dataclass(frozen=True, eq=False)
class Epochs:
    """Equal-length segments of multichannel EEG, one label per segment."""

    data: np.ndarray
    fs: float
    tmin: float
    ch_names: tuple[str, ...]
    labels: tuple[str, ...]
    subject: str = ""

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=float)
        if data.ndim != 3:
            raise ValueError(f"data must be (n_epochs, n_channels, n_times); got {data.shape}")
        if not self.fs > 0:
            raise ValueError("fs must be positive")
        names = _as_names(self.ch_names)
        labels = tuple(str(label) for label in self.labels)
        if len(names) != data.shape[1]:
            raise ValueError(f"{len(names)} channel names for {data.shape[1]} channels")
        if len(set(names)) != len(names):
            raise ValueError("channel names must be unique")
        if len(labels) != data.shape[0]:
            raise ValueError(f"{len(labels)} labels for {data.shape[0]} epochs")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "ch_names", names)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "fs", float(self.fs))
        object.__setattr__(self, "tmin", float(self.tmin))

    # ----------------------------------------------------------------- basic geometry
    @property
    def n_epochs(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.data.shape[2])

    @property
    def times(self) -> np.ndarray:
        """Time of every sample in seconds, relative to the event."""
        return self.tmin + np.arange(self.n_times) / self.fs

    def sample_range(self, start: float, stop: float) -> tuple[int, int]:
        """Inclusive sample indices covering ``[start, stop]`` seconds.

        Raises ``ValueError`` when the interval is not inside the epoch or has fewer than two
        samples.
        """
        if not start < stop:
            raise ValueError(f"empty interval ({start}, {stop})")
        first = math.ceil(round((start - self.tmin) * self.fs, 9))
        last = math.floor(round((stop - self.tmin) * self.fs, 9))
        if first < 0 or last > self.n_times - 1:
            raise ValueError(
                f"interval ({start}, {stop}) s is outside the epoch "
                f"[{self.times[0]:.3f}, {self.times[-1]:.3f}] s"
            )
        if last - first < 1:
            raise ValueError(f"interval ({start}, {stop}) s holds fewer than two samples")
        return first, last

    # ----------------------------------------------------------------- selection
    def with_data(self, data: np.ndarray, ch_names: Sequence[str] | None = None) -> Epochs:
        names = self.ch_names if ch_names is None else tuple(ch_names)
        return replace(self, data=data, ch_names=names)

    def select(self, *tags: str) -> Epochs:
        """Epochs whose label has every tag (labels are ``/``-separated, like MNE event ids)."""
        if not tags:
            raise ValueError("select() needs at least one tag")
        wanted = set(tags)
        mask = np.array([wanted <= set(label.split("/")) for label in self.labels], dtype=bool)
        return self.keep(mask)

    def keep(self, mask: np.ndarray) -> Epochs:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_epochs,):
            raise ValueError(f"mask must have shape ({self.n_epochs},)")
        labels = tuple(label for label, ok in zip(self.labels, mask, strict=True) if ok)
        return replace(self, data=self.data[mask], labels=labels)

    def pick(self, channels: Sequence[str]) -> Epochs:
        """Keep ``channels`` (in the order given)."""
        index = []
        for name in channels:
            if name not in self.ch_names:
                raise KeyError(f"channel {name!r} not found; available: {list(self.ch_names)}")
            index.append(self.ch_names.index(name))
        return replace(self, data=self.data[:, index, :], ch_names=tuple(channels))

    def baseline_corrected(self, interval: tuple[float, float]) -> Epochs:
        """Subtract, per epoch and channel, the mean over ``interval`` (seconds)."""
        i0, i1 = self.sample_range(*interval)
        base = self.data[:, :, i0 : i1 + 1].mean(axis=2, keepdims=True)
        return replace(self, data=self.data - base)

    def __len__(self) -> int:
        return self.n_epochs


@dataclass(frozen=True, eq=False)
class Recording:
    """A continuous multichannel recording with an event list.

    ``events`` is an integer array ``(n_events, 2)``: sample index of the cue and an event code;
    ``event_labels`` maps codes to labels. ``eog`` is an optional dedicated EOG channel (same
    length as ``data``). ``truth`` is only filled by :mod:`eegerds.synth`.
    """

    data: np.ndarray
    fs: float
    ch_names: tuple[str, ...]
    events: np.ndarray
    event_labels: Mapping[int, str]
    eog: np.ndarray | None = None
    subject: str = ""
    truth: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"data must be (n_channels, n_samples); got {data.shape}")
        names = _as_names(self.ch_names)
        if len(names) != data.shape[0]:
            raise ValueError(f"{len(names)} channel names for {data.shape[0]} channels")
        events = np.asarray(self.events, dtype=int).reshape(-1, 2)
        unknown = set(events[:, 1].tolist()) - set(self.event_labels)
        if unknown:
            raise ValueError(f"event codes without a label: {sorted(unknown)}")
        eog = None if self.eog is None else np.asarray(self.eog, dtype=float)
        if eog is not None and eog.shape != (data.shape[1],):
            raise ValueError("eog must be 1-D with one value per sample")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "ch_names", names)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "eog", eog)
        object.__setattr__(self, "fs", float(self.fs))

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration(self) -> float:
        return self.n_samples / self.fs

    def with_data(self, data: np.ndarray, eog: np.ndarray | None | str = "keep") -> Recording:
        new_eog = self.eog if isinstance(eog, str) else eog
        return replace(self, data=data, eog=new_eog)


def epoch(
    rec: Recording,
    tmin: float,
    tmax: float,
    baseline: tuple[float, float] | None = None,
) -> Epochs:
    """Cut ``[tmin, tmax)`` seconds around every event.

    Events whose window does not fit in the recording are dropped. ``baseline`` (seconds,
    relative to the event) is subtracted per epoch and channel when given.
    """
    if not tmin < tmax:
        raise ValueError("tmin must be smaller than tmax")
    start_off = round(tmin * rec.fs)
    n_times = round((tmax - tmin) * rec.fs)
    if n_times < 2:
        raise ValueError("epoch shorter than two samples")
    segments, labels = [], []
    for sample, code in rec.events:
        a = int(sample) + start_off
        b = a + n_times
        if a < 0 or b > rec.n_samples:
            continue
        segments.append(rec.data[:, a:b])
        labels.append(rec.event_labels[int(code)])
    if not segments:
        raise ValueError("no event has a full window inside the recording")
    out = Epochs(
        data=np.stack(segments),
        fs=rec.fs,
        tmin=start_off / rec.fs,
        ch_names=rec.ch_names,
        labels=tuple(labels),
        subject=rec.subject,
    )
    return out.baseline_corrected(baseline) if baseline is not None else out
