import numpy as np
import pytest

import eegerds as ee
from eegerds.erds import windowed_mean
from eegerds.synth import BANDS, DEFAULT_CHANNELS

from .conftest import CLEAN


def test_same_seed_same_recording_different_seed_different():
    a = ee.simulate_recording(ee.SimConfig(n_trials=2), seed=5)
    b = ee.simulate_recording(ee.SimConfig(n_trials=2), seed=5)
    c = ee.simulate_recording(ee.SimConfig(n_trials=2), seed=6)
    assert np.array_equal(a.data, b.data) and np.array_equal(a.events, b.events)
    assert not np.array_equal(a.data, c.data)


def test_geometry_events_and_units(clean_rec):
    cfg = CLEAN
    n_trials = cfg.n_trials * len(cfg.conditions)
    assert clean_rec.data.shape == (len(DEFAULT_CHANNELS), n_trials * round(cfg.span * cfg.fs))
    assert clean_rec.events.shape == (n_trials, 2)
    assert np.all(np.diff(clean_rec.events[:, 0]) == round(cfg.span * cfg.fs))
    assert set(clean_rec.event_labels.values()) == set(cfg.conditions)
    counts = np.bincount(clean_rec.events[:, 1])[1:]
    assert np.all(counts == cfg.n_trials)  # balanced design
    assert 1e-6 < clean_rec.data.std() < 1e-4  # volts, microvolt scale
    assert clean_rec.eog is not None and clean_rec.eog.shape == (clean_rec.n_samples,)


def test_group_is_reproducible_and_subjects_differ():
    cfg = ee.SimConfig(n_trials=2)
    g1 = ee.simulate_group(3, cfg, seed=1)
    g2 = ee.simulate_group(3, cfg, seed=1)
    assert [r.subject for r in g1] == ["S01", "S02", "S03"]
    assert all(np.array_equal(x.data, y.data) for x, y in zip(g1, g2, strict=True))
    scales = [r.truth["depth_scale"] for r in g1]
    assert len(set(scales)) == 3
    with pytest.raises(ValueError, match="at least 1"):
        ee.simulate_group(0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"fs": 50.0}, "fs"),
        ({"n_trials": 0}, "n_trials"),
        ({"epoch": (1.0, 5.0)}, "straddle"),
        ({"mu_depth": 1.0}, "mu_depth"),
        ({"conditions": ("high",)}, "left"),
        ({"erd": (0.5, 9.0)}, "inside the epoch"),
        ({"artifact_rate": 1.5}, "artifact_rate"),
        ({"epoch": (-1.0, 2.5), "erd": (0.5, 2.0), "ers": (2.0, 2.5)}, "4 s"),
    ],
)
def test_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ee.SimConfig(**kwargs)


def test_expected_erds_has_the_designed_shape():
    cfg = ee.SimConfig()
    t = np.arange(-2, 6, 0.01)
    contra = ee.expected_erds(cfg, "left", "C4", "mu", t)
    ipsi = ee.expected_erds(cfg, "left", "C3", "mu", t)
    occipital = ee.expected_erds(cfg, "left", "Oz", "mu", t)
    assert np.allclose(contra[t < 0.0], 0.0, atol=1e-6)  # nothing before the ERD window
    inside = (t > 1.0) & (t < 2.5)
    assert contra[inside].mean() < ipsi[inside].mean() < 0  # contralateral is deeper
    assert ipsi[inside].mean() == pytest.approx(cfg.ipsi_ratio * contra[inside].mean(), rel=0.1)
    assert np.abs(occipital).max() < 1.0  # the control rhythm does not react
    beta = ee.expected_erds(cfg, "left", "C4", "beta", t)
    assert beta[(t > 3.5) & (t < 4.0)].mean() > 50  # rebound after the ERD window
    assert beta[inside].mean() < 0
    # mirror symmetry between hemispheres and sides
    assert np.allclose(contra, ee.expected_erds(cfg, "right", "C3", "mu", t))
    # linear in the subject's depth scale
    assert np.allclose(ee.expected_erds(cfg, "left", "C4", "mu", t, depth_scale=0.5), contra / 2)


@pytest.mark.parametrize(
    "args",
    [("up", "C3", "mu"), ("left", "Qz", "mu"), ("left", "C3", "gamma")],
)
def test_expected_erds_rejects_bad_arguments(args):
    side, channel, band = args
    with pytest.raises((ValueError, KeyError)):
        ee.expected_erds(ee.SimConfig(), side, channel, band, np.arange(0, 1, 0.1))


def test_simulated_erd_matches_the_analytic_expectation(clean_rec, clean_epochs):
    """The simulator and the estimator agree to within sampling noise (25 epochs per side)."""
    depth = clean_rec.truth["depth_scale"]
    cfg = ee.ErdsConfig(band=BANDS["mu"])
    for side in ("left", "right"):
        group = clean_epochs.select(side)
        curve = ee.erds_curve(group, cfg)
        for channel in ("C3", "C4"):
            expected = ee.expected_erds(CLEAN, side, channel, "mu", group.times, depth)
            expected = windowed_mean(expected, group.fs, cfg.window, cfg.overlap)[1]
            inside = (curve.times > 0.75) & (curve.times < 2.75)
            assert curve.channel(channel)[inside].mean() == pytest.approx(
                expected[inside].mean(), abs=9.0
            )


def test_beta_rebound_is_recovered(clean_rec, clean_epochs):
    depth = clean_rec.truth["depth_scale"]
    cfg = ee.ErdsConfig(band=BANDS["beta"])
    group = clean_epochs.select("left")
    curve = ee.erds_curve(group, cfg)
    expected = ee.expected_erds(CLEAN, "left", "C4", "beta", group.times, depth)
    expected = windowed_mean(expected, group.fs, cfg.window, cfg.overlap)[1]
    rebound = (curve.times > 3.25) & (curve.times < 4.25)
    assert curve.channel("C4")[rebound].mean() == pytest.approx(expected[rebound].mean(), abs=15)
    assert curve.channel("C4")[rebound].mean() > 40


def test_occipital_control_rhythm_does_not_react(clean_epochs):
    curve = ee.erds_curve(clean_epochs, ee.ErdsConfig())
    inside = (curve.times > 0.75) & (curve.times < 2.75)
    assert abs(curve.channel("Oz")[inside].mean()) < 12  # noise only
    assert curve.channel("C3")[inside].mean() < -15  # the sensorimotor rhythm does


def test_blinks_follow_the_designed_topography():
    rec = ee.simulate_recording(ee.SimConfig(n_trials=6, artifact_rate=0.0), seed=2)
    blink = rec.truth["blink_signal"]
    assert len(rec.truth["blink_starts"]) > 5
    r = {ch: np.corrcoef(rec.data[i], blink)[0, 1] for i, ch in enumerate(rec.ch_names)}
    assert r["Fp1"] > 0.5 and r["Fp2"] > 0.5
    assert r["Fp1"] > r["Fz"] > abs(r["Oz"])
    assert rec.eog is not None and np.corrcoef(rec.eog, blink)[0, 1] > 0.99
    off = ee.simulate_recording(ee.SimConfig(n_trials=2, blink_rate=0.0, artifact_rate=0.0), seed=2)
    assert len(off.truth["blink_starts"]) == 0 and np.all(off.truth["blink_signal"] == 0)


def test_artifacts_are_recorded_in_the_truth():
    none = ee.simulate_recording(ee.SimConfig(n_trials=4, artifact_rate=0.0), seed=1)
    every = ee.simulate_recording(ee.SimConfig(n_trials=4, artifact_rate=1.0), seed=1)
    assert none.truth["artifact_trials"] == {}
    assert len(every.truth["artifact_trials"]) == 16
    assert set(every.truth["artifact_trials"].values()) <= {"burst", "sway", "pop"}


def test_p300_is_larger_in_high_conditions(clean_epochs):
    high = clean_epochs.select("high").data.mean(axis=0)
    low = clean_epochs.select("low").data.mean(axis=0)
    cz = clean_epochs.ch_names.index("Cz")
    window = (clean_epochs.times > 0.28) & (clean_epochs.times < 0.36)
    assert high[cz, window].mean() > 1.2 * low[cz, window].mean() > 0
