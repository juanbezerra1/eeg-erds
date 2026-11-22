"""Reading and writing epochs, and converting from/to MNE-Python.

``save_epochs``/``load_epochs`` use a plain ``.npz`` archive. Loading never unpickles anything
(``allow_pickle=False``), so opening a file from somebody else cannot run code.

``from_mne`` and ``to_mne`` work with ``mne.Epochs`` objects. MNE is an optional dependency:
``from_mne`` only reads attributes (no import), ``to_mne`` imports it when called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .data import Epochs

__all__ = ["from_mne", "load_epochs", "save_epochs", "to_mne"]


def save_epochs(path: str | Path, epochs: Epochs) -> Path:
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=epochs.data,
        fs=np.float64(epochs.fs),
        tmin=np.float64(epochs.tmin),
        ch_names=np.array(epochs.ch_names, dtype=str),
        labels=np.array(epochs.labels, dtype=str),
        subject=np.array(epochs.subject, dtype=str),
    )
    return path


def load_epochs(path: str | Path) -> Epochs:
    with np.load(Path(path), allow_pickle=False) as z:
        missing = {"data", "fs", "tmin", "ch_names", "labels"} - set(z.files)
        if missing:
            raise ValueError(f"{path} is not an eegerds epochs file (missing {sorted(missing)})")
        subject = str(z["subject"]) if "subject" in z.files else ""
        return Epochs(
            data=z["data"],
            fs=float(z["fs"]),
            tmin=float(z["tmin"]),
            ch_names=tuple(str(c) for c in z["ch_names"]),
            labels=tuple(str(label) for label in z["labels"]),
            subject=subject,
        )


def from_mne(mne_epochs: Any, subject: str = "", picks: str = "eeg") -> Epochs:
    """Convert an ``mne.Epochs`` (labels come from ``event_id``; data stay in volts)."""
    picked = mne_epochs.copy().pick(picks)
    inverse = {code: name for name, code in mne_epochs.event_id.items()}
    labels = [inverse[int(code)] for code in mne_epochs.events[:, 2]]
    return Epochs(
        data=picked.get_data(),
        fs=float(mne_epochs.info["sfreq"]),
        tmin=float(mne_epochs.tmin),
        ch_names=tuple(picked.ch_names),
        labels=tuple(labels),
        subject=subject,
    )


def to_mne(epochs: Epochs) -> Any:
    """Convert to an ``mne.EpochsArray`` (requires ``pip install eeg-erds[mne]``)."""
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - exercised only without mne
        raise ImportError("to_mne needs MNE-Python: pip install 'eeg-erds[mne]'") from exc
    names = sorted(set(epochs.labels))
    event_id = {name: i + 1 for i, name in enumerate(names)}
    codes = [event_id[label] for label in epochs.labels]
    events = np.column_stack(
        [np.arange(epochs.n_epochs) * epochs.n_times, np.zeros(epochs.n_epochs, int), codes]
    )
    info = mne.create_info(list(epochs.ch_names), epochs.fs, "eeg")
    return mne.EpochsArray(
        epochs.data, info, events=events, tmin=epochs.tmin, event_id=event_id, verbose="ERROR"
    )
