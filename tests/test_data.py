import numpy as np
import pytest

import eegerds as ee


def make_epochs(n_epochs=4, n_ch=3, n_times=100, fs=100.0, tmin=-0.5, labels=None):
    rng = np.random.default_rng(0)
    labels = labels or ["left/high", "left/low", "right/high", "right/low"][:n_epochs]
    return ee.Epochs(
        data=rng.standard_normal((n_epochs, n_ch, n_times)),
        fs=fs,
        tmin=tmin,
        ch_names=("C3", "Cz", "C4")[:n_ch],
        labels=tuple(labels),
        subject="S1",
    )


def test_geometry_and_times():
    ep = make_epochs()
    assert (ep.n_epochs, ep.n_channels, ep.n_times) == (4, 3, 100)
    assert len(ep) == 4
    assert ep.times[0] == -0.5 and ep.times[-1] == pytest.approx(0.49)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"data": np.zeros((3, 10))}, "n_epochs, n_channels, n_times"),
        ({"fs": 0}, "fs must be positive"),
        ({"ch_names": ("A", "B")}, "channel names"),
        ({"ch_names": ("A", "A", "B")}, "unique"),
        ({"labels": ("a", "b")}, "labels"),
    ],
)
def test_epochs_validation(kwargs, message):
    base = {
        "data": np.zeros((4, 3, 10)),
        "fs": 100.0,
        "tmin": 0.0,
        "ch_names": ("A", "B", "C"),
        "labels": ("a", "b", "c", "d"),
    }
    with pytest.raises(ValueError, match=message):
        ee.Epochs(**{**base, **kwargs})


def test_sample_range_is_inclusive_and_checks_the_interval():
    ep = make_epochs()  # times -0.50 ... 0.49 at 100 Hz
    assert ep.sample_range(-0.5, -0.4) == (0, 10)
    assert ep.sample_range(0.0, 0.2) == (50, 70)
    for bad, message in [((0.0, 0.0), "empty"), ((-1.0, 0.0), "outside"), ((0.0, 0.6), "outside")]:
        with pytest.raises(ValueError, match=message):
            ep.sample_range(*bad)
    with pytest.raises(ValueError, match="fewer than two samples"):
        ep.sample_range(0.0, 0.004)


def test_select_uses_slash_separated_tags():
    ep = make_epochs()
    assert ep.select("left").labels == ("left/high", "left/low")
    assert ep.select("high").labels == ("left/high", "right/high")
    assert ep.select("left", "high").labels == ("left/high",)
    assert ep.select("lef").n_epochs == 0  # whole tags only, not substrings
    with pytest.raises(ValueError, match="at least one tag"):
        ep.select()


def test_keep_pick_and_with_data():
    ep = make_epochs()
    kept = ep.keep(np.array([True, False, True, False]))
    assert kept.labels == ("left/high", "right/high")
    assert np.array_equal(kept.data, ep.data[[0, 2]])
    with pytest.raises(ValueError, match="mask"):
        ep.keep(np.array([True, False]))
    picked = ep.pick(["C4", "C3"])
    assert picked.ch_names == ("C4", "C3")
    assert np.array_equal(picked.data[:, 0], ep.data[:, 2])
    with pytest.raises(KeyError, match="Pz"):
        ep.pick(["Pz"])
    assert ep.with_data(ep.data * 2).data[0, 0, 0] == 2 * ep.data[0, 0, 0]


def test_baseline_correction_zeroes_the_interval_mean():
    ep = make_epochs().baseline_corrected((-0.5, 0.0))
    i0, i1 = ep.sample_range(-0.5, 0.0)
    assert np.allclose(ep.data[:, :, i0 : i1 + 1].mean(axis=2), 0.0)


def make_recording(n=2000, fs=100.0):
    rng = np.random.default_rng(1)
    events = np.array([[300, 1], [800, 2], [1990, 1], [10, 1]])
    return ee.Recording(
        data=rng.standard_normal((2, n)),
        fs=fs,
        ch_names=("A", "B"),
        events=events,
        event_labels={1: "left", 2: "right"},
        subject="S9",
    )


def test_epoch_cuts_around_events_and_drops_those_that_do_not_fit():
    rec = make_recording()
    ep = ee.epoch(rec, -1.0, 2.0)
    assert ep.data.shape == (2, 2, 300)  # events at 1990 and 10 do not fit
    assert ep.labels == ("left", "right") and ep.tmin == -1.0 and ep.subject == "S9"
    assert np.array_equal(ep.data[0], rec.data[:, 200:500])
    based = ee.epoch(rec, -1.0, 2.0, baseline=(-1.0, 0.0))
    assert np.allclose(based.data[:, :, :101].mean(axis=2), 0.0)


def test_epoch_errors():
    rec = make_recording()
    with pytest.raises(ValueError, match="tmin must be smaller"):
        ee.epoch(rec, 1.0, 1.0)
    with pytest.raises(ValueError, match="no event"):
        ee.epoch(rec, -50.0, 50.0)
    with pytest.raises(ValueError, match="two samples"):
        ee.epoch(rec, 0.0, 0.001)


def test_recording_validation():
    ok = dict(
        data=np.zeros((2, 50)), fs=10.0, ch_names=("A", "B"), events=np.array([[5, 1]]),
        event_labels={1: "x"},
    )  # fmt: skip
    rec = ee.Recording(**ok)
    assert rec.n_samples == 50 and rec.duration == 5.0
    with pytest.raises(ValueError, match="without a label"):
        ee.Recording(**{**ok, "events": np.array([[5, 7]])})
    with pytest.raises(ValueError, match="eog"):
        ee.Recording(**{**ok, "eog": np.zeros(7)})
    with pytest.raises(ValueError, match="channel names"):
        ee.Recording(**{**ok, "ch_names": ("A",)})
    with pytest.raises(ValueError, match="n_channels, n_samples"):
        ee.Recording(**{**ok, "data": np.zeros(50)})
    swapped = rec.with_data(np.ones((2, 50)), eog=np.zeros(50))
    assert swapped.eog is not None and rec.eog is None
