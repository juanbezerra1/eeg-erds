import numpy as np
import pandas as pd
import pytest

import eegerds as ee
from eegerds.erp import mean_amplitude

T = np.arange(-0.2, 0.8, 0.002)


def gauss(centre, width, amp):
    return amp * np.exp(-((T - centre) ** 2) / (2 * width**2))


def erp_trace(n200=0.20, p300=0.32, n400=0.46):
    return gauss(n200, 0.03, -4) + gauss(p300, 0.05, 6) + gauss(n400, 0.06, -3)


def by_name(peaks):
    return {p.name: p for p in peaks}


def test_ordered_peaks_are_found_at_the_right_latencies():
    peaks = by_name(ee.find_peaks_ordered(T, erp_trace()))
    assert peaks["N200"].latency == pytest.approx(0.20, abs=0.012)
    assert peaks["P300"].latency == pytest.approx(0.32, abs=0.004)
    assert peaks["N400"].latency == pytest.approx(0.46, abs=0.012)
    assert peaks["N200"].amplitude < 0 < peaks["P300"].amplitude and peaks["N400"].amplitude < 0
    assert all(p.found and not p.at_edge for p in peaks.values())


def test_a_later_component_cannot_come_before_the_one_it_follows():
    trace = erp_trace() + gauss(0.14, 0.02, 20)  # a huge positive bump before the N200
    peaks = by_name(ee.find_peaks_ordered(T, trace))
    assert peaks["P300"].latency > peaks["N200"].latency
    assert peaks["N400"].latency > peaks["P300"].latency


def test_a_trace_that_ends_early_flags_the_cut_window_and_finds_nothing_after_it():
    short = T <= 0.25
    peaks = by_name(ee.find_peaks_ordered(T[short], erp_trace()[short]))
    assert peaks["N200"].found and not peaks["N200"].at_edge
    assert (
        peaks["P300"].found and peaks["P300"].at_edge
    )  # its window was cut by the end of the data
    assert not peaks["N400"].found and np.isnan(peaks["N400"].latency)
    assert np.isnan(peaks["N400"].amplitude)


def test_a_component_that_is_not_found_blocks_the_ones_that_follow_it():
    trace = erp_trace()
    trace[(T > 0.15) & (T < 0.25)] = np.nan  # the N200 cannot be measured ...
    peaks = by_name(ee.find_peaks_ordered(T, trace))
    assert not peaks["N200"].found
    # ... and the P300/N400 windows are defined relative to it, so they are not guessed
    assert not peaks["P300"].found and not peaks["N400"].found


def test_a_peak_on_the_edge_of_its_window_is_flagged():
    ramp = np.linspace(0, 1, len(T))  # no peak: the extremum is at the end of the window
    peaks = by_name(ee.find_peaks_ordered(T, ramp, [ee.PeakSpec("P300", "pos", 0.2, 0.5)]))
    assert peaks["P300"].found and peaks["P300"].at_edge


def test_nan_inside_the_window_means_not_found():
    trace = erp_trace()
    trace[(T > 0.25) & (T < 0.3)] = np.nan
    assert not by_name(ee.find_peaks_ordered(T, trace))["P300"].found


def test_spec_validation():
    with pytest.raises(ValueError, match="polarity"):
        ee.PeakSpec("X", "up", 0.1, 0.2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="start must be smaller"):
        ee.PeakSpec("X", "pos", 0.3, 0.2)
    with pytest.raises(ValueError, match="max_gap"):
        ee.PeakSpec("X", "pos", 0.1, 0.2, after="Y", min_gap=0.1, max_gap=0.05)
    with pytest.raises(ValueError, match="not listed before"):
        ee.find_peaks_ordered(T, erp_trace(), [ee.PeakSpec("P300", "pos", 0.2, 0.5, after="N200")])
    dup = [ee.PeakSpec("A", "pos", 0.1, 0.2), ee.PeakSpec("A", "neg", 0.2, 0.3)]
    with pytest.raises(ValueError, match="duplicate"):
        ee.find_peaks_ordered(T, erp_trace(), dup)
    with pytest.raises(ValueError, match="same length"):
        ee.find_peaks_ordered(T, erp_trace()[:-1])


def test_mean_amplitude():
    assert mean_amplitude(T, np.ones_like(T), 0.3, 0.025) == pytest.approx(1.0)
    ramp = T.copy()
    assert mean_amplitude(T, ramp, 0.3, 0.05) == pytest.approx(0.3, abs=1e-3)
    assert np.isnan(mean_amplitude(T, ramp, 5.0, 0.01))


def test_fixed_latency_amplitudes_are_unbiased_but_per_subject_peaks_are_not():
    """Why amplitudes are measured at the group latency: the max of noise is always positive."""
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((400, len(T)))
    window = (T > 0.25) & (T < 0.45)
    per_subject_peak = noise[:, window].max(axis=1).mean()
    at_fixed_latency = np.array([mean_amplitude(T, row, 0.35, 0.025) for row in noise]).mean()
    assert per_subject_peak > 1.5  # "found" a P300 of 1.5+ SD in pure noise
    assert abs(at_fixed_latency) < 0.1


def make_group(n_per_subject, seed=0):
    rng = np.random.default_rng(seed)
    eps = []
    for s, n in enumerate(n_per_subject):
        data = rng.standard_normal((n, 2, len(T))) * 1e-6
        labels = tuple("high" if i % 2 else "low" for i in range(n))
        eps.append(
            ee.Epochs(data, fs=500.0, tmin=float(T[0]), ch_names=("Cz", "Pz"), labels=labels,
                      subject=f"S{s}")
        )  # fmt: skip
    return eps


def test_group_erp_weights_every_subject_once():
    eps = make_group([2, 20])
    group = ee.group_erp(eps)
    assert group.values.shape == (2, 2, len(T))
    assert np.allclose(group.values[0], eps[0].data.mean(axis=0))
    assert np.allclose(group.grand, group.values.mean(axis=0))  # not the pooled epoch mean
    assert group.channel("Cz").shape == (2, len(T))
    with pytest.raises(KeyError):
        group.channel("Fz")


def test_group_erp_selects_tags_and_applies_baseline():
    eps = make_group([10, 10])
    high = ee.group_erp(eps, ["high"], baseline=(-0.2, 0.0))
    manual = eps[0].select("high").baseline_corrected((-0.2, 0.0)).data.mean(axis=0)
    assert np.allclose(high.values[0], manual)
    with pytest.raises(ValueError, match="no epochs with tags"):
        ee.group_erp(eps, ["medium"])
    with pytest.raises(ValueError, match="no subjects"):
        ee.group_erp([])
    other = make_group([4])[0].pick(["Cz"])
    with pytest.raises(ValueError, match="differ"):
        ee.group_erp([eps[0], other])


def test_find_group_peaks_and_measure_amplitudes():
    rng = np.random.default_rng(1)
    eps = []
    for s in range(5):
        data = rng.standard_normal((12, 2, len(T))) * 1e-6 + erp_trace()[None, None, :] * 1e-6
        eps.append(ee.Epochs(data, 500.0, float(T[0]), ("Cz", "Pz"), ("a",) * 12, f"S{s}"))
    group = ee.group_erp(eps)
    peaks = ee.find_group_peaks(group, ["Cz"])
    assert [p.name for p in peaks["Cz"]] == ["N200", "P300", "N400"]
    table = ee.measure_amplitudes(group, peaks, half_window=0.02)
    assert list(table.columns) == ["subject", "channel", "component", "latency", "amplitude"]
    assert len(table) == 5 * 3
    p300 = table[table["component"] == "P300"]
    assert p300["amplitude"].mean() == pytest.approx(5.9e-6, abs=0.8e-6)
    with pytest.raises(KeyError):
        ee.find_group_peaks(group, ["Fz"])


def test_measure_amplitudes_yields_nan_for_components_that_were_not_found():
    group = ee.GroupErp(T, np.zeros((3, 1, len(T))), ("a", "b", "c"), ("Cz",))
    peaks = {"Cz": [ee.Peak("P300", np.nan, np.nan, False)]}
    table = ee.measure_amplitudes(group, peaks)
    assert isinstance(table, pd.DataFrame) and table["amplitude"].isna().all()


def test_the_simulated_p300_is_recovered(group_epochs, group_results):
    recs, _ = group_results
    group = ee.group_erp(group_epochs, baseline=(-0.2, 0.0))
    peaks = by_name(ee.find_group_peaks(group, ["Cz"])["Cz"])
    truth = ee.SimConfig().erp_latency[1] + float(np.mean([r.truth["latency_shift"] for r in recs]))
    assert peaks["P300"].found and peaks["P300"].latency == pytest.approx(truth, abs=0.04)
    assert peaks["P300"].amplitude > 0 > peaks["N400"].amplitude
