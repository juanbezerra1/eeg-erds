import numpy as np
import pytest

import eegerds as ee
from eegerds.filters import bandpass, notch
from eegerds.ica import _joint_diagonalise


def three_sources(n=20000, fs=250.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    ar = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(1, n):
        ar[i] = 0.95 * ar[i - 1] + e[i]
    s = np.vstack(
        [
            np.sin(2 * np.pi * 3 * t + 0.5),
            np.sin(2 * np.pi * 11 * t) * np.sin(2 * np.pi * 0.2 * t),
            ar,
        ]
    )
    return (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True)


def best_match(est, true):
    """For each true source, the best |correlation| with any estimated component."""
    c = np.abs(np.corrcoef(est, true)[: est.shape[0], est.shape[0] :])
    return c.max(axis=0)


def test_sobi_recovers_sources_with_different_autocorrelation():
    s = three_sources()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 3)) @ s + 0.01 * rng.standard_normal((6, s.shape[1]))
    res = ee.sobi(x, n_components=3)
    assert res.converged
    assert best_match(res.sources, s).min() > 0.99


def test_components_are_unit_variance_uncorrelated_and_consistent():
    s = three_sources()
    x = np.random.default_rng(2).standard_normal((5, 3)) @ s
    res = ee.sobi(x, n_components=3)
    assert np.allclose(np.cov(res.sources), np.eye(3), atol=1e-6)
    assert np.allclose(res.unmixing @ (x - res.mean[:, None]), res.sources, atol=1e-8)
    assert np.allclose(res.mean[:, None] + res.mixing @ res.sources, x, atol=1e-8)  # rank 3: exact
    energy = (res.mixing**2).sum(axis=0)
    assert np.all(np.diff(energy) <= 1e-12)  # sorted by variance carried


def test_sobi_is_deterministic():
    x = np.random.default_rng(3).standard_normal((4, 3)) @ three_sources(8000)
    a, b = ee.sobi(x, n_components=3), ee.sobi(x, n_components=3)
    assert np.array_equal(a.sources, b.sources)


def test_joint_diagonalisation_recovers_the_common_rotation():
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    mats = np.stack([q @ np.diag(rng.standard_normal(5)) @ q.T for _ in range(8)])
    v, sweeps, converged = _joint_diagonalise(mats.copy(), tol=1e-10, max_sweeps=100)
    assert converged and sweeps < 100
    overlap = np.abs(v.T @ q)  # a permutation matrix, up to signs
    assert np.allclose(np.sort(overlap, axis=1)[:, -1], 1.0, atol=1e-8)
    assert np.allclose(overlap.sum(axis=0), 1.0, atol=1e-8)


def test_joint_diagonalisation_says_when_it_ran_out_of_sweeps():
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    mats = np.stack([q @ np.diag(rng.standard_normal(5)) @ q.T for _ in range(8)])
    _, sweeps, converged = _joint_diagonalise(mats.copy(), tol=1e-12, max_sweeps=1)
    assert sweeps == 1 and not converged


@pytest.mark.parametrize(
    "x, kwargs, message",
    [
        (np.zeros(50), {}, "n_channels, n_samples"),
        (np.zeros((4, 20)), {}, "too few samples"),
        (np.full((2, 200), np.nan), {}, "NaN"),
        (np.zeros((2, 200)), {}, "constant"),
        (np.random.default_rng(0).standard_normal((3, 400)), {"lags": [0]}, "lags"),
        (np.random.default_rng(0).standard_normal((3, 400)), {"lags": [999]}, "lags"),
        (np.random.default_rng(0).standard_normal((3, 400)), {"n_components": 0}, "n_components"),
    ],
)
def test_sobi_input_checks(x, kwargs, message):
    with pytest.raises(ValueError, match=message):
        ee.sobi(x, **kwargs)


def filtered_blinky(seed=3):
    rec = ee.simulate_recording(ee.SimConfig(n_trials=8, artifact_rate=0.0), seed=seed)
    x = bandpass(notch(rec.data, rec.fs, 50.0), rec.fs, 1.0, 45.0, order=2)
    return rec, x


def test_blink_component_is_found_and_removed_without_touching_distant_channels():
    rec, x = filtered_blinky()
    blink = rec.truth["blink_signal"]
    assert rec.eog is not None
    out = ee.remove_eog_components(x, bandpass(rec.eog, rec.fs, 1.0, 45.0, order=2))
    assert out.removed == (int(np.argmax(np.abs(out.correlations))),)
    assert np.abs(out.correlations[out.removed[0]]) > 0.8
    names = list(rec.ch_names)
    fp1, c3, oz = (names.index(c) for c in ("Fp1", "C3", "Oz"))
    before = abs(np.corrcoef(x[fp1], blink)[0, 1])
    after = abs(np.corrcoef(out.cleaned[fp1], blink)[0, 1])
    assert before > 0.5 and after < 0.1
    for j in (c3, oz):  # far from the eyes: essentially unchanged
        assert out.cleaned[j].std() == pytest.approx(x[j].std(), rel=0.03)
    assert out.cleaned.shape == x.shape


def test_a_recording_without_blinks_is_left_alone():
    rec = ee.simulate_recording(ee.SimConfig(n_trials=6, blink_rate=0.0, artifact_rate=0.0), seed=1)
    x = bandpass(rec.data, rec.fs, 1.0, 45.0, order=2)
    out = ee.remove_eog_components(x, rec.eog)  # the EOG trace is just noise here
    assert out.removed == ()
    assert np.abs(out.correlations).max() < 0.3
    assert np.allclose(out.cleaned, x, atol=1e-9 * np.abs(x).max())  # full-rank reconstruction


def test_n_remove_zero_reconstructs_the_input():
    rec, x = filtered_blinky()
    out = ee.remove_eog_components(x, rec.eog, n_remove=0)
    assert out.removed == () and np.allclose(out.cleaned, x, atol=1e-9 * np.abs(x).max())


def test_more_components_can_be_removed():
    rec, x = filtered_blinky()
    one = ee.remove_eog_components(x, rec.eog, n_remove=1, min_abs_r=0.0)
    two = ee.remove_eog_components(x, rec.eog, n_remove=2, min_abs_r=0.0)
    assert len(one.removed) == 1 and len(two.removed) == 2 and one.removed[0] == two.removed[0]


def test_remove_eog_input_checks():
    x = np.random.default_rng(0).standard_normal((3, 600))
    with pytest.raises(ValueError, match="eog must be 1-D"):
        ee.remove_eog_components(x, np.zeros(10))
    with pytest.raises(ValueError, match="eeg must be"):
        ee.remove_eog_components(x[0], np.zeros(600))
    with pytest.raises(ValueError, match="n_remove"):
        ee.remove_eog_components(x, np.zeros(600), n_remove=-1)
    flat = x.copy()
    flat[1] = 3.0
    with pytest.raises(ValueError, match="constant"):
        ee.remove_eog_components(flat, np.ones(600))
