"""The whole chain from a continuous recording to clean epochs.

    notch -> band-pass -> SOBI blink removal -> epoching (+ baseline) -> epoch rejection

Every step is optional and every intermediate result is returned, so the effect of each choice
(was the blink removed? how many epochs were dropped, and why?) can be inspected.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .artifacts import RejectionConfig, RejectionReport, reject_epochs
from .data import Epochs, Recording, epoch
from .filters import bandpass, notch
from .ica import EogRemoval, remove_eog_components

__all__ = [
    "PipelineConfig",
    "SubjectResult",
    "blink_residual",
    "clean_recording",
    "process_group",
    "process_recording",
]


@dataclass(frozen=True)
class PipelineConfig:
    band: tuple[float, float] | None = (1.0, 45.0)  # broadband filter before everything else
    filter_order: int = 2
    notch_hz: float | None = 50.0  # mains frequency, None to skip
    remove_blinks: bool = True
    n_remove: int = 1  # at most this many components are removed
    min_abs_r: float = 0.3  # ... and only if they correlate with the EOG at least this much
    epoch: tuple[float, float] = (-2.0, 6.0)
    baseline: tuple[float, float] | None = (-2.0, 0.0)
    rejection: RejectionConfig | None = field(default_factory=RejectionConfig)


@dataclass(frozen=True, eq=False)
class SubjectResult:
    subject: str
    epochs: Epochs  # after rejection
    epochs_all: Epochs  # before rejection
    report: RejectionReport | None
    eog: EogRemoval | None
    cleaned: Recording


def clean_recording(
    rec: Recording, cfg: PipelineConfig | None = None
) -> tuple[Recording, EogRemoval | None]:
    """Filter and (optionally) remove blink components from a continuous recording."""
    cfg = cfg or PipelineConfig()
    data = rec.data
    if cfg.notch_hz is not None:
        data = notch(data, rec.fs, cfg.notch_hz)
    if cfg.band is not None:
        data = bandpass(data, rec.fs, cfg.band[0], cfg.band[1], order=cfg.filter_order)
    removal = None
    if cfg.remove_blinks:
        if rec.eog is None:
            raise ValueError("remove_blinks needs an EOG trace (Recording.eog)")
        eog = rec.eog
        if cfg.band is not None:  # compare like with like
            eog = bandpass(eog, rec.fs, cfg.band[0], cfg.band[1], order=cfg.filter_order)
        removal = remove_eog_components(data, eog, n_remove=cfg.n_remove, min_abs_r=cfg.min_abs_r)
        data = removal.cleaned
    return rec.with_data(data), removal


def process_recording(rec: Recording, cfg: PipelineConfig | None = None) -> SubjectResult:
    """Run the full chain on one subject."""
    cfg = cfg or PipelineConfig()
    cleaned, removal = clean_recording(rec, cfg)
    epochs_all = epoch(cleaned, cfg.epoch[0], cfg.epoch[1], baseline=cfg.baseline)
    if cfg.rejection is None:
        return SubjectResult(rec.subject, epochs_all, epochs_all, None, removal, cleaned)
    report = reject_epochs(epochs_all, cfg.rejection)
    if not report.keep.any():
        raise ValueError(f"subject {rec.subject or '?'}: every epoch was rejected")
    return SubjectResult(
        rec.subject, report.apply(epochs_all), epochs_all, report, removal, cleaned
    )


def process_group(
    recordings: Sequence[Recording], cfg: PipelineConfig | None = None
) -> list[SubjectResult]:
    """:func:`process_recording` for each subject."""
    if not recordings:
        raise ValueError("no recordings")
    return [process_recording(rec, cfg) for rec in recordings]


def blink_residual(result: SubjectResult, truth: np.ndarray, channel: str = "Fp1") -> float:
    """Correlation between ``channel`` of the cleaned data and a known blink signal."""
    j = result.cleaned.ch_names.index(channel)
    return float(np.corrcoef(result.cleaned.data[j], truth)[0, 1])
