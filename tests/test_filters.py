import dataclasses

import numpy as np
import pytest

import eegerds as ee
from eegerds.filters import bandpass, lowpass, notch

FS = 500.0
T = np.arange(0, 20, 1 / FS)


def amplitude(x):
    """Amplitude of a sinusoid from its RMS, ignoring the filter's edge transients."""
    core = x[len(x) // 4 : -len(x) // 4]
    return np.sqrt(2) * core.std()


def sine(freq):
    return np.sin(2 * np.pi * freq * T)


def test_bandpass_passes_the_band_and_attenuates_outside():
    assert amplitude(bandpass(sine(10), FS, 8, 13)) == pytest.approx(1.0, abs=0.15)
    assert amplitude(bandpass(sine(2), FS, 8, 13)) < 0.02
    assert amplitude(bandpass(sine(40), FS, 8, 13)) < 0.02


def test_bandpass_is_zero_phase():
    x = sine(10)
    y = bandpass(x, FS, 8, 13)
    core = slice(len(x) // 4, -len(x) // 4)
    lags = np.arange(-20, 21)
    xc = [np.dot(x[core], np.roll(y, k)[core]) for k in lags]
    assert lags[int(np.argmax(xc))] == 0


def test_bandpass_works_along_any_axis():
    x = np.stack([sine(10), sine(40)])
    along_last = bandpass(x, FS, 8, 13)
    along_first = bandpass(x.T, FS, 8, 13, axis=0).T
    assert np.allclose(along_last, along_first)
    assert amplitude(along_last[1]) < 0.02


def test_notch_removes_the_mains_and_keeps_alpha():
    mixed = sine(50) + sine(10)
    out = notch(mixed, FS, 50)
    assert amplitude(out) == pytest.approx(1.0, abs=0.05)
    assert amplitude(notch(sine(50), FS, 50)) < 0.05


def test_lowpass():
    assert amplitude(lowpass(sine(2), FS, 10)) == pytest.approx(1.0, abs=0.05)
    assert amplitude(lowpass(sine(60), FS, 10)) < 0.01


@pytest.mark.parametrize(
    "call",
    [
        lambda: bandpass(sine(10), FS, 13, 8),
        lambda: bandpass(sine(10), FS, 0, 8),
        lambda: bandpass(sine(10), FS, 8, 300),
        lambda: lowpass(sine(10), FS, 300),
        lambda: notch(sine(10), FS, 0),
    ],
)
def test_invalid_cutoffs_are_rejected(call):
    with pytest.raises(ValueError):
        call()


def test_bandpass_recording_filters_data_and_eog_and_keeps_everything_else(clean_rec):
    out = ee.bandpass_recording(clean_rec, 8.0, 13.0)
    assert np.array_equal(out.data, ee.bandpass(clean_rec.data, clean_rec.fs, 8.0, 13.0))
    assert np.array_equal(out.eog, ee.bandpass(clean_rec.eog, clean_rec.fs, 8.0, 13.0))
    assert (out.fs, out.ch_names, out.subject) == (
        clean_rec.fs,
        clean_rec.ch_names,
        clean_rec.subject,
    )
    assert np.array_equal(out.events, clean_rec.events)
    assert out.event_labels == clean_rec.event_labels
    assert out.truth.keys() == clean_rec.truth.keys()
    assert not np.array_equal(out.data, clean_rec.data)  # and the original is untouched


def test_bandpass_recording_without_an_eog_trace(clean_rec):
    bare = dataclasses.replace(clean_rec, eog=None)
    assert ee.bandpass_recording(bare, 1.0, 40.0).eog is None
