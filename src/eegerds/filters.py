"""Zero-phase digital filters (applied along the last axis by default).

All filters are run forwards and backwards (``sosfiltfilt``/``filtfilt``), so they add no phase
delay and square the magnitude response. That matters for ERP latencies and for ERD/ERS timing.
Second-order sections are used for the Butterworth designs because a narrow band-pass at a low
normalised frequency (8-13 Hz at 600 Hz, say) is numerically fragile in transfer-function form.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import signal

from .data import Recording

__all__ = ["bandpass", "bandpass_recording", "lowpass", "notch"]


def _check_band(fs: float, *edges: float) -> None:
    nyquist = fs / 2
    for edge in edges:
        if not 0 < edge < nyquist:
            raise ValueError(f"cut-off {edge} Hz must be between 0 and Nyquist ({nyquist} Hz)")


def bandpass(
    x: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 4,
    axis: int = -1,
    padlen: int | None = None,
) -> np.ndarray:
    """Butterworth band-pass of the given ``order``, zero phase.

    ``padlen`` is the number of samples of odd extension added at each end before filtering
    (``None``: SciPy's default, a few dozen samples). Narrow bands have a long memory, so for
    short epochs a longer pad (about one second) removes most of the edge transient.
    """
    if not low < high:
        raise ValueError("low must be smaller than high")
    _check_band(fs, low, high)
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=float), axis=axis, padlen=padlen)


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 2, axis: int = -1) -> np.ndarray:
    """Butterworth low-pass, zero phase."""
    _check_band(fs, cutoff)
    sos = signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=float), axis=axis)


def notch(x: np.ndarray, fs: float, freq: float, q: float = 30.0, axis: int = -1) -> np.ndarray:
    """Narrow notch at ``freq`` Hz (quality factor ``q``), zero phase."""
    _check_band(fs, freq)
    b, a = signal.iirnotch(freq, q, fs)
    return signal.filtfilt(b, a, np.asarray(x, dtype=float), axis=axis)


def bandpass_recording(rec: Recording, low: float, high: float, order: int = 4) -> Recording:
    """A copy of ``rec`` with every channel (and the EOG trace, if any) band-passed.

    This is the exact way to prepare ERD/ERS: filter the *continuous* signal in the band of
    interest, epoch it, then call the ERD/ERS functions with ``ErdsConfig(prefiltered=True)``.
    Nothing is distorted at the epoch edges because the filter never saw them.
    """
    eog = None if rec.eog is None else bandpass(rec.eog, rec.fs, low, high, order=order)
    return replace(rec, data=bandpass(rec.data, rec.fs, low, high, order=order), eog=eog)
