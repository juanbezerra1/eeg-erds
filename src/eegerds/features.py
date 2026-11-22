"""Trial-level ERD/ERS features, as a table that other tools can read.

``trial_features`` turns every epoch into one row of band-power changes: for each channel, band
and time window, ``ln(P_window / P_reference)`` of that single epoch (the natural log of the
power ratio, i.e. ``ln(1 + ERD% / 100)``; it is symmetric around 0 and far less skewed than
the percentage). It is the feature set a motor-imagery classifier would use.

The columns follow a small CSV contract shared by the sibling projects (``biosigkit`` and
``subjectwise-ml``):

* ``subject``: who the row belongs to (the grouping variable for subject-wise validation);
* ``level``: the class label of the epoch (named ``level`` so ``subjectwise`` finds it by default;
  pass ``label_column`` to rename);
* ``trial``: running index of the epoch within its subject;
* one numeric column per feature, named ``<channel>_<band>_<start>-<stop>s``.

Feeding that file to ``subjectwise evaluate`` gives a leave-one-subject-out accuracy instead of the
optimistic number a random split of epochs would give.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import Epochs
from .filters import bandpass

__all__ = ["DEFAULT_BANDS", "default_windows", "features_table", "trial_features"]

DEFAULT_BANDS: dict[str, tuple[float, float]] = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}


def default_windows(
    epochs: Epochs, width: float = 1.0, start: float = 0.0
) -> list[tuple[float, float]]:
    """Consecutive windows of ``width`` seconds from ``start`` to the end of the epoch."""
    out, t = [], start
    while t + width <= epochs.times[-1] + 1e-9:
        out.append((t, t + width))
        t += width
    if not out:
        raise ValueError("epoch too short for the requested window width")
    return out


def _levels(epochs: Epochs, classes: Sequence[str] | None) -> tuple[np.ndarray, list[str]]:
    if classes is None:
        return np.ones(epochs.n_epochs, dtype=bool), list(epochs.labels)
    wanted = set(classes)
    keep, levels = [], []
    for label in epochs.labels:
        hits = wanted & set(label.split("/"))
        if len(hits) > 1:
            raise ValueError(f"label {label!r} contains more than one of {sorted(wanted)}")
        keep.append(bool(hits))
        levels.append(next(iter(hits)) if hits else "")
    return np.array(keep), levels


def trial_features(
    epochs: Epochs,
    bands: Mapping[str, tuple[float, float]] | None = None,
    windows: Sequence[tuple[float, float]] | None = None,
    reference: tuple[float, float] = (-2.0, -1.0),
    classes: Sequence[str] | None = None,
    order: int = 4,
    label_column: str = "level",
) -> pd.DataFrame:
    """One row per epoch: ``subject, <label_column>, trial`` and the log power ratios.

    ``classes`` reduces a composite label (``"left/high"``) to one tag (``"left"``): epochs
    carrying none of the classes are left out. By default the whole label is the class.
    """
    bands = dict(DEFAULT_BANDS if bands is None else bands)
    wins = list(
        default_windows(epochs, start=max(0.0, epochs.tmin)) if windows is None else windows
    )
    if not bands or not wins:
        raise ValueError("bands and windows must not be empty")
    keep, levels = _levels(epochs, classes)
    i0, i1 = epochs.sample_range(*reference)
    index = [epochs.sample_range(a, b) for a, b in wins]
    pad = min(epochs.n_times - 1, round(epochs.fs))  # one second of padding: see erds.py

    columns: dict[str, np.ndarray] = {}
    for band, (low, high) in bands.items():
        power = bandpass(epochs.data, epochs.fs, low, high, order=order, axis=-1, padlen=pad) ** 2
        ref = power[:, :, i0 : i1 + 1].mean(axis=2)  # (n_epochs, n_channels)
        if np.any(ref <= 0) or not np.all(np.isfinite(ref)):
            raise ValueError(f"reference power is zero or not finite in band {band!r}")
        for (a, b), (j0, j1) in zip(wins, index, strict=True):
            win = power[:, :, j0 : j1 + 1].mean(axis=2)
            ratio = np.log(win / ref)
            for c, name in enumerate(epochs.ch_names):
                columns[f"{name}_{band}_{a:g}-{b:g}s"] = ratio[:, c]

    frame = pd.DataFrame(
        {
            "subject": epochs.subject or "S",
            label_column: levels,
            "trial": np.arange(epochs.n_epochs),
        }
    )
    frame = pd.concat([frame, pd.DataFrame(columns)], axis=1)
    return frame.loc[keep].reset_index(drop=True)


def features_table(epochs_per_subject: Sequence[Epochs], **kwargs) -> pd.DataFrame:
    """:func:`trial_features` of several subjects, stacked into one table."""
    if not epochs_per_subject:
        raise ValueError("no subjects")
    frames = [trial_features(ep, **kwargs) for ep in epochs_per_subject]
    return pd.concat(frames, ignore_index=True)
