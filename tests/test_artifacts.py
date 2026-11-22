import numpy as np
import pytest

import eegerds as ee
from eegerds.artifacts import REASONS, _percent_to_samples, longest_run

FS = 100.0
N = 200  # 2 s epochs: 5 % = 10 samples, 10 % = 20 samples
UV = 1e-6


def alternating(n_epochs=20, n_channels=1):
    """+-1 uV square wave: every epoch has SD exactly 1 uV, so the default threshold is 3 uV."""
    pattern = np.where(np.arange(N) % 2 == 0, UV, -UV)
    return np.tile(pattern, (n_epochs, n_channels, 1))


def make(data, names=None):
    names = names or tuple(f"Ch{j}" for j in range(data.shape[1]))
    return ee.Epochs(data, FS, 0.0, names, ("a",) * data.shape[0])


def brute_longest_run(row):
    best = current = 0
    for value in row:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


@pytest.mark.parametrize("density", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_longest_run_matches_a_plain_loop(density):
    flags = np.random.default_rng(int(density * 100)).random((4, 3, 57)) < density
    expected = [[brute_longest_run(flags[i, j]) for j in range(3)] for i in range(4)]
    assert np.array_equal(longest_run(flags), expected)


def test_longest_run_edge_cases():
    assert longest_run(np.zeros(10, bool)) == 0
    assert longest_run(np.ones(10, bool)) == 10
    assert longest_run(np.array([1, 0, 0, 0], bool)) == 1
    assert longest_run(np.array([0, 0, 1, 1, 1], bool)) == 3
    assert longest_run(np.array([1, 1, 0, 1, 1, 1, 0, 1], bool)) == 3
    assert longest_run(np.zeros((0, 5), bool)).shape == (0,)


def test_percentages_become_whole_samples_without_float_noise():
    assert _percent_to_samples(200, 5) == 10
    assert _percent_to_samples(200, 10) == 20
    assert _percent_to_samples(101, 5) == 6  # 5.05 samples: more than 5 in a row is "too long"
    assert _percent_to_samples(1000, 0.7) == 7  # 1000 * 0.7 / 100 = 7.000000000000001 in floats
    assert _percent_to_samples(100, 100) == 100


def test_clean_epochs_are_all_kept():
    report = ee.reject_epochs(make(alternating()))
    assert report.keep.all() and report.n_rejected == 0 and report.fraction_rejected == 0
    assert all(r == () for r in report.reasons)
    assert report.bad_channels == {}
    assert report.thresholds == pytest.approx([3 * UV])
    assert report.counts() == {r: 0 for r in REASONS}


@pytest.mark.parametrize("length, rejected", [(10, False), (11, True)])
def test_run_rule_triggers_on_more_than_five_percent_in_a_row(length, rejected):
    data = alternating()
    data[4, 0, 30 : 30 + length] = 5 * UV  # beyond +3 uV, consecutively
    report = ee.reject_epochs(make(data))
    assert (not report.keep[4]) is rejected
    assert report.reasons[4] == (("run",) if rejected else ())
    assert report.keep[np.arange(20) != 4].all()


def test_run_rule_also_sees_negative_deflections():
    data = alternating()
    data[2, 0, 60:75] = -5 * UV
    assert ee.reject_epochs(make(data)).reasons[2] == ("run",)


def test_a_positive_and_a_negative_stretch_do_not_add_up_as_one_run():
    data = alternating()
    data[1, 0, 20:30] = 5 * UV  # 10 samples up ...
    data[1, 0, 30:40] = -5 * UV  # ... then 10 down: neither is longer than 10
    assert ee.reject_epochs(make(data)).keep[1]


@pytest.mark.parametrize("count, rejected", [(20, False), (21, True)])
def test_total_rule_counts_samples_that_are_not_in_a_row(count, rejected):
    data = alternating()
    data[5, 0, np.arange(count) * 3] = 5 * UV  # isolated samples, 3 apart
    report = ee.reject_epochs(make(data))
    assert (not report.keep[5]) is rejected
    assert report.reasons[5] == (("total",) if rejected else ())  # the run rule stays quiet


def test_total_rule_counts_both_signs_separately():
    data = alternating()
    data[6, 0, np.arange(15) * 3] = 5 * UV
    data[6, 0, np.arange(15) * 3 + 1] = -5 * UV  # 30 samples out, but 15 on each side
    assert ee.reject_epochs(make(data)).keep[6]


@pytest.mark.parametrize(
    "value, rejected", [(200 * UV, True), (-200 * UV, True), (140 * UV, False)]
)
def test_amplitude_rule_is_an_absolute_limit(value, rejected):
    data = alternating()
    data[3, 0, 100] = value
    report = ee.reject_epochs(make(data))
    assert (not report.keep[3]) is rejected
    assert report.reasons[3] == (("amplitude",) if rejected else ())


def test_one_epoch_can_break_several_rules_and_counts_say_so():
    data = alternating()
    data[0, 0, 40:70] = 5 * UV  # 30 in a row: breaks both run (>10) and total (>20)
    report = ee.reject_epochs(make(data))
    assert report.reasons[0] == ("run", "total")
    assert report.counts() == {"run": 1, "total": 1, "amplitude": 0, "variance": 0}
    assert report.n_rejected == 1 and report.fraction_rejected == pytest.approx(1 / 20)


def test_thresholds_and_flags_are_per_channel():
    data = alternating(n_channels=2)
    data[:, 1, :] *= 10  # channel 1 is ten times larger: its threshold is 30 uV
    data[7, 1, 30:50] = 5 * UV  # a stretch that is huge for channel 0 but ordinary for channel 1
    data[8, 0, 30:50] = 5 * UV  # the same stretch in channel 0
    data[9, 1, 30:50] = 80 * UV  # and one that is big for channel 1
    report = ee.reject_epochs(make(data, ("Fz", "Cz")))
    assert report.thresholds == pytest.approx([3 * UV, 30 * UV])
    assert report.keep[7]
    assert not report.keep[8] and not report.keep[9]
    assert report.bad_channels == {"Fz": 1, "Cz": 1}


def test_default_threshold_is_a_median_and_survives_a_contaminated_minority():
    data = alternating(21)
    data[11:, 0, :] *= 4  # ten epochs with 4 uV SD, eleven with 1 uV
    report = ee.reject_epochs(make(data))
    assert report.thresholds == pytest.approx([3 * UV])  # median SD is 1 uV
    assert not report.keep[11:].any() and report.keep[:11].all()


def test_reference_epochs_set_the_threshold_from_their_mean():
    data = alternating(20)
    data[10:, 0, :] *= 4  # half the epochs are 4x larger: the median is pulled in between
    default = ee.reject_epochs(make(data))
    assert default.thresholds == pytest.approx([3 * 2.5 * UV]) and default.keep.all()
    ref = ee.reject_epochs(make(data), reference=range(10))
    assert ref.thresholds == pytest.approx([3 * UV])
    assert not ref.keep[10:].any() and ref.keep[:10].all()
    mixed = ee.reject_epochs(make(data), reference=[0, 15])  # mean of 1 and 4 uV
    assert mixed.thresholds == pytest.approx([3 * 2.5 * UV])


@pytest.mark.parametrize("bad", [[], [-1], [20], [0, 99]])
def test_invalid_reference_indices(bad):
    with pytest.raises(ValueError, match="reference"):
        ee.reject_epochs(make(alternating()), reference=bad)


def test_demean_removes_a_constant_offset_before_the_rules_are_applied():
    data = alternating()
    data[3, 0, :] += 100 * UV  # a DC offset, below the 150 uV amplitude limit
    assert ee.reject_epochs(make(data)).keep[3]
    raw = ee.reject_epochs(make(data), ee.RejectionConfig(demean=False))
    assert raw.reasons[3] == ("run", "total")


def test_variance_rule_is_off_by_default_and_finds_what_the_others_cannot():
    data = alternating(20, 2)
    data[7, 1, 50:100] = np.where(np.arange(50) % 2 == 0, 2.2 * UV, -2.2 * UV)  # SD 2.2 uV, thr 3
    base = ee.reject_epochs(make(data))
    assert base.keep.all()  # run/total/amplitude cannot see a burst that never reaches 3 SD
    for ratio, flagged in [(1.5, True), (2.0, True), (2.5, False)]:
        cfg = ee.RejectionConfig(max_window_sd_ratio=ratio)
        report = ee.reject_epochs(make(data), cfg)
        assert (not report.keep[7]) is flagged, ratio
        if flagged:
            assert report.reasons[7] == ("variance",)
            assert report.bad_channels == {"Ch1": 1}
            assert report.keep[np.arange(20) != 7].all()
            assert report.counts()["variance"] == 1


def test_variance_window_length_follows_window_sd_seconds():
    data = alternating(20)
    data[7, 0, 50:60] = 0.0  # a 0.1 s flat stretch: invisible in 0.5 s windows (SD 0.9 uV) ...
    data[7, 0, 60:100] = np.where(np.arange(40) % 2 == 0, 2.5 * UV, -2.5 * UV)
    wide = ee.RejectionConfig(max_window_sd_ratio=2.4, window_sd_seconds=0.5)
    assert ee.reject_epochs(make(data), wide).keep[7]  # window SD is below 2.4 x typical
    narrow = ee.RejectionConfig(max_window_sd_ratio=2.4, window_sd_seconds=0.2)
    assert not ee.reject_epochs(make(data), narrow).keep[7]  # 0.2 s windows isolate the burst


def test_variance_rule_needs_at_least_two_windows():
    short = make(alternating(5)[:, :, :60])
    cfg = ee.RejectionConfig(max_window_sd_ratio=3.0)
    with pytest.raises(ValueError, match="too short"):
        ee.reject_epochs(short, cfg)
    assert ee.reject_epochs(short).keep.all()  # the other rules do not mind


def test_apply_returns_the_kept_epochs_only():
    data = alternating()
    data[3, 0, 100] = 300 * UV
    data[9, 0, 40:70] = 5 * UV
    epochs = make(data)
    report = ee.reject_epochs(epochs)
    kept = report.apply(epochs)
    assert report.n_rejected == 2 and kept.n_epochs == 18
    assert np.array_equal(kept.data, epochs.data[report.keep])
    assert len(kept.labels) == 18 and kept.ch_names == epochs.ch_names


def test_non_finite_values_are_refused():
    for bad in (np.nan, np.inf):
        data = alternating()
        data[0, 0, 0] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            ee.reject_epochs(make(data))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sd_factor": 0}, "sd_factor"),
        ({"max_run_percent": 0}, "percentages"),
        ({"max_total_percent": 101}, "percentages"),
        ({"max_abs": -1e-6}, "max_abs"),
        ({"max_window_sd_ratio": 1.0}, "greater than 1"),
        ({"window_sd_seconds": 0}, "window_sd_seconds"),
    ],
)
def test_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ee.RejectionConfig(**kwargs)


def score(recs, results):
    """Recall on the simulated artefact trials and false-rejection rate on the others."""
    caught = missed = false_pos = clean = 0
    for rec, res in zip(recs, results, strict=True):
        truth = rec.truth["artifact_trials"]
        flagged = ~res.report.keep
        for trial, is_flagged in enumerate(flagged):
            if trial in truth:
                caught += int(is_flagged)
                missed += int(not is_flagged)
            else:
                false_pos += int(is_flagged)
                clean += 1
    return caught / (caught + missed), false_pos / clean


def test_simulated_artefacts_are_caught_without_rejecting_clean_trials():
    recs = ee.simulate_group(8, ee.SimConfig(n_trials=10, artifact_rate=0.25), seed=21)
    three = ee.process_group(recs, ee.PipelineConfig(rejection=ee.RejectionConfig()))
    four = ee.process_group(
        recs, ee.PipelineConfig(rejection=ee.RejectionConfig(max_window_sd_ratio=3.0))
    )
    recall3, fp3 = score(recs, three)
    recall4, fp4 = score(recs, four)
    assert recall4 >= 0.9 and fp4 <= 0.02
    assert recall3 < recall4  # the three original rules miss the noise-like bursts
    assert fp3 <= 0.02
