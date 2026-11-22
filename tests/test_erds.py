import numpy as np
import pytest

import eegerds as ee
from eegerds.erds import window_centres, windowed_mean
from eegerds.filters import bandpass

FS = 250.0


def band_noise(n_epochs, n_ch, fs=FS, tmin=-2.0, tmax=6.0, f0=10.5, seed=0):
    """Unit-variance noise confined to ~9-12 Hz, random phase in every epoch.

    It is filtered as one continuous signal and then cut into epochs, so it has no edge
    transients of its own (filtering each epoch separately would put them in the test data).
    """
    rng = np.random.default_rng(seed)
    n = round((tmax - tmin) * fs)
    long = bandpass(rng.standard_normal((n_ch, n_epochs * n)), fs, f0 - 1.5, f0 + 1.5, order=4)
    x = long.reshape(n_ch, n_epochs, n).transpose(1, 0, 2)
    return x / x.std(), tmin + np.arange(n) / fs


def make(x, tmin=-2.0, fs=FS, labels=None, ch=None):
    n_epochs, n_ch, _ = x.shape
    return ee.Epochs(
        data=x, fs=fs, tmin=tmin, ch_names=tuple(ch or [f"Ch{i}" for i in range(n_ch)]),
        labels=tuple(labels or ["a"] * n_epochs), subject="S",
    )  # fmt: skip


def modulated(power_factor, start=1.0, stop=3.0, n_epochs=60, seed=0):
    x, t = band_noise(n_epochs, 1, seed=seed)
    gain = np.where((t >= start) & (t < stop), power_factor, 1.0)
    return make(x * np.sqrt(gain))


def interior(curve, start=1.3, stop=2.7):
    return (curve.times > start) & (curve.times < stop)


# ------------------------------------------------------------------ the estimate itself
@pytest.mark.parametrize("factor", [0.5, 0.25, 2.0])
def test_recovers_a_known_power_change_with_the_right_sign(factor):
    """ERD/ERS = factor - 1. Measured spread of one estimate: about 12 % of the ratio."""
    ratios = []
    for seed in range(8):
        curve = ee.erds_curve(modulated(factor, seed=seed))
        ratios.append(1 + curve.values[0][interior(curve)].mean() / 100)
        assert abs(curve.values[0][curve.times < 0.0].mean()) < 25.0  # before the change: ~0
    assert np.mean(ratios) == pytest.approx(factor, rel=0.12)
    assert (np.mean(ratios) < 1) == (factor < 1)  # ERD is negative, ERS positive


def test_no_change_gives_no_erds_on_average():
    means = [ee.erds_curve(modulated(1.0, seed=s)).values[0].mean() for s in range(30)]
    assert abs(np.mean(means)) < 8.0  # measured bias: about +3 points
    assert np.std(means) < 12.0  # a single estimate is noisy (about 7 points) but not wild


def test_padding_keeps_the_edge_transient_small():
    """Per-epoch band-passing distorts the epoch ends; the padding keeps that below ~8 %."""
    x, t = band_noise(150, 1, seed=1)
    ep = make(x)
    power = ee.sample_power(ep, ee.ErdsConfig())[0]
    truth = bandpass(x.transpose(1, 0, 2).reshape(1, -1), FS, 8, 13).reshape(1, 150, -1)
    truth = truth.transpose(1, 0, 2).var(axis=0, ddof=1)[0]
    ratio = power / truth
    assert abs(ratio[t > 5.5].mean() - 1) < 0.08
    assert abs(ratio[t < -1.5].mean() - 1) < 0.10
    assert abs(ratio[(t > -1.0) & (t < 5.0)].mean() - 1) < 0.02


def test_scale_invariance():
    ep = modulated(0.5)
    big = ep.with_data(ep.data * 1e3)
    assert np.allclose(ee.erds_curve(ep).values, ee.erds_curve(big).values)


def test_changing_the_reference_rescales_every_window_by_one_constant():
    ep = modulated(0.5)
    a = ee.erds_curve(ep, ee.ErdsConfig(reference=(-2.0, -1.0)))
    b = ee.erds_curve(ep, ee.ErdsConfig(reference=(-1.0, 0.0)))
    ratio = (1 + a.values / 100) / (1 + b.values / 100)  # equals R_b / R_a for every window
    assert np.allclose(ratio, ratio[0, 0])


def test_the_two_methods_agree_on_induced_activity_and_differ_on_evoked_activity():
    x, t = band_noise(60, 1, seed=3)
    burst = np.where((t >= 1.0) & (t < 2.0), 2.0 * np.sin(2 * np.pi * 10.0 * t), 0.0)
    evoked = make(x + burst[None, None, :])  # the same burst, in phase, in every epoch
    plain = make(x)
    iv = ee.ErdsConfig(method="intertrial_variance")
    pw = ee.ErdsConfig(method="power")
    inside = interior(ee.erds_curve(plain), 1.3, 1.7)
    assert ee.erds_curve(evoked, iv).values[0][inside].mean() == pytest.approx(
        ee.erds_curve(plain, iv).values[0][inside].mean(), abs=12.0
    )  # phase-locked energy is invisible to the inter-trial variance...
    assert ee.erds_curve(evoked, pw).values[0][inside].mean() > 100.0  # ...but not to power
    assert ee.erds_curve(plain, pw).values[0][inside].mean() == pytest.approx(
        ee.erds_curve(plain, iv).values[0][inside].mean(), abs=10.0
    )  # and they agree when there is no evoked part


def test_prefiltered_skips_the_band_pass():
    ep = modulated(0.5, n_epochs=8)
    cfg = ee.ErdsConfig(prefiltered=True)
    assert np.allclose(ee.sample_power(ep, cfg), ep.data.var(axis=0, ddof=1))
    assert not np.allclose(ee.sample_power(ep, ee.ErdsConfig()), ee.sample_power(ep, cfg))


# ------------------------------------------------------------------ windows
def test_window_geometry_matches_a_brute_force_loop():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 1000))
    for window, overlap in [(0.5, 0.5), (0.5, 0.0), (0.3, 0.75)]:
        starts, means = windowed_mean(x, 250.0, window, overlap)
        length = round(window * 250.0)
        step = max(1, round(length * (1 - overlap)))
        assert np.array_equal(starts, np.arange(0, 1000 - length + 1, step))
        brute = np.array([[x[c, s : s + length].mean() for s in starts] for c in range(2)])
        assert np.allclose(means, brute)


def test_window_centres_and_counts():
    ep = make(np.zeros((2, 1, 2000)))
    assert len(window_centres(ep, 0.5, 0.5)) == 31 and len(window_centres(ep, 0.5, 0.0)) == 16
    centres = window_centres(ep, 0.5, 0.0)
    assert centres[0] == pytest.approx(-2.0 + (125 - 1) / 2 / FS)
    assert np.allclose(np.diff(centres), 0.5)


def test_window_errors():
    with pytest.raises(ValueError, match="longer than the epoch"):
        windowed_mean(np.zeros(10), 10.0, 5.0, 0.5)
    with pytest.raises(ValueError, match="shorter than two samples"):
        windowed_mean(np.zeros(100), 10.0, 0.05, 0.5)


# ------------------------------------------------------------------ smoothing and groups
def test_smoothing_is_a_valid_moving_average_without_edge_bias():
    ep = modulated(1.0, n_epochs=10)
    curve = ee.erds_curve(ep)
    times, values = curve.smoothed(3)
    assert values.shape[-1] == curve.values.shape[-1] - 2 and len(times) == len(values[0])
    assert values[0, 0] == pytest.approx(curve.values[0, :3].mean())
    assert times[0] == pytest.approx(curve.times[:3].mean())
    flat = ee.ErdsCurve(curve.times, np.full((1, 31), 7.0), ("A",), 10, curve.config)
    assert np.allclose(flat.smoothed(3)[1], 7.0)  # a constant stays constant, edges included
    with pytest.raises(ValueError, match="exceeds"):
        curve.smoothed(100)
    with pytest.raises(ValueError, match="at least 1"):
        curve.smoothed(0)


def test_group_shapes_mean_and_sem():
    eps = [modulated(0.5, seed=s) for s in range(4)]
    group = ee.erds_group(eps)
    assert group.values.shape == (4, 1, 31) and group.n_subjects == 4
    assert np.allclose(group.mean(), group.values.mean(axis=0))
    assert np.allclose(group.sem(), group.values.std(axis=0, ddof=1) / 2)
    assert group.channel("Ch0").shape == (4, 31)
    with pytest.raises(KeyError):
        group.channel("Nope")
    one = ee.erds_group(eps[:1])
    assert np.isnan(one.sem()).all()
    times, mean = group.smoothed_mean(3)
    assert mean.shape == (1, 29) and len(times) == 29


def test_group_rejects_mismatched_subjects_and_empty_input():
    a, b = modulated(0.5), make(band_noise(60, 2)[0])
    with pytest.raises(ValueError, match="differ"):
        ee.erds_group([a, b])
    with pytest.raises(ValueError, match="no subjects"):
        ee.erds_group([])


# ------------------------------------------------------------------ input checking
def test_errors_on_unusable_input():
    ep = modulated(0.5, n_epochs=2)
    one = ep.keep(np.array([True, False]))
    with pytest.raises(ValueError, match="at least 2 epochs"):
        ee.erds_curve(one)
    assert ee.erds_curve(one, ee.ErdsConfig(method="power")).values.shape[0] == 1
    with pytest.raises(ValueError, match="outside the epoch"):
        ee.erds_curve(ep, ee.ErdsConfig(reference=(-5.0, -4.0)))
    bad = ep.data.copy()
    bad[0, 0, 5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ee.erds_curve(ep.with_data(bad))
    flat = ep.with_data(np.zeros_like(ep.data))
    with pytest.raises(ValueError, match="reference power is zero"):
        ee.erds_curve(flat, ee.ErdsConfig(prefiltered=True))
    with pytest.raises(ValueError, match="no epochs"):
        ee.erds_curve(ep.select("nothing like this"), ee.ErdsConfig(method="power"))


def test_unknown_channel_names_the_available_ones():
    curve = ee.erds_curve(modulated(0.5, n_epochs=4))
    assert curve.channel(curve.ch_names[0]).shape == curve.times.shape
    with pytest.raises(KeyError, match="not found.*available"):
        curve.channel("Nope")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"band": (13.0, 8.0)},
        {"overlap": 1.0},
        {"overlap": -0.1},
        {"window": 0.0},
        {"method": "wavelet"},
        {"reference": (-1.0, -2.0)},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        ee.ErdsConfig(**kwargs)


def test_prefiltered_workflow_is_exact_and_broadband_data_dilute_the_erd(clean_rec):
    """Filter the continuous signal in the band, epoch, ``prefiltered=True``; not the other way."""
    band = (8.0, 13.0)
    j = list(clean_rec.ch_names).index("C4")  # contralateral to the left hand
    plain = ee.epoch(clean_rec, -2.0, 6.0, baseline=(-2.0, 0.0)).select("left")
    in_band = ee.epoch(
        ee.bandpass_recording(clean_rec, *band), -2.0, 6.0, baseline=(-2.0, 0.0)
    ).select("left")

    exact = ee.erds_curve(in_band, ee.ErdsConfig(band=band, prefiltered=True))
    padded = ee.erds_curve(plain, ee.ErdsConfig(band=band))
    broadband = ee.erds_curve(plain, ee.ErdsConfig(band=band, prefiltered=True))  # the misuse

    expected = ee.expected_erds(
        clean_rec.truth["config"], "left", "C4", "mu", exact.times, clean_rec.truth["depth_scale"]
    )
    inside = (exact.times > 0.8) & (exact.times < 2.7)
    target = expected[inside].mean()
    assert exact.values[j][inside].mean() == pytest.approx(target, abs=8.0)
    assert padded.values[j][inside].mean() == pytest.approx(target, abs=8.0)
    assert abs(broadband.values[j][inside].mean()) < 0.5 * abs(target)  # less than half as deep
