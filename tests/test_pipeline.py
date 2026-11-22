import dataclasses

import numpy as np
import pytest

import eegerds as ee
from eegerds.pipeline import blink_residual, clean_recording

SMALL = ee.SimConfig(n_trials=5)


@pytest.fixture(scope="module")
def rec():
    return ee.simulate_recording(SMALL, seed=3, subject="S01")


def amplitude_at(x, fs, hz):
    spectrum = np.abs(np.fft.rfft(x, axis=-1)) * 2 / x.shape[-1]
    return spectrum[..., np.argmin(np.abs(np.fft.rfftfreq(x.shape[-1], 1 / fs) - hz))].mean()


def test_defaults_describe_the_documented_chain():
    cfg = ee.PipelineConfig()
    assert cfg.band == (1.0, 45.0) and cfg.notch_hz == 50.0 and cfg.remove_blinks
    assert cfg.epoch == (-2.0, 6.0) and cfg.baseline == (-2.0, 0.0)
    assert cfg.rejection == ee.RejectionConfig()


def test_notch_removes_line_noise_and_nothing_is_changed_in_place(rec):
    before = rec.data.copy()
    cleaned, removal = clean_recording(rec, ee.PipelineConfig(remove_blinks=False))
    assert removal is None
    assert amplitude_at(rec.data, rec.fs, 50.0) > 2e-6  # 3 uV of mains in the simulation
    assert amplitude_at(cleaned.data, rec.fs, 50.0) < 0.05e-6
    assert np.array_equal(rec.data, before)  # the input recording is untouched
    assert cleaned.subject == rec.subject and cleaned.ch_names == rec.ch_names


def test_all_steps_can_be_switched_off(rec):
    cfg = ee.PipelineConfig(band=None, notch_hz=None, remove_blinks=False, rejection=None)
    cleaned, removal = clean_recording(rec, cfg)
    assert removal is None and np.array_equal(cleaned.data, rec.data)
    result = ee.process_recording(rec, cfg)
    assert result.report is None and result.eog is None and result.epochs is result.epochs_all


def test_blink_removal_needs_an_eog_trace(rec):
    with pytest.raises(ValueError, match="EOG"):
        clean_recording(dataclasses.replace(rec, eog=None))
    no_eog = dataclasses.replace(rec, eog=None)
    assert clean_recording(no_eog, ee.PipelineConfig(remove_blinks=False))[1] is None


def test_process_recording_returns_every_intermediate_result(rec):
    result = ee.process_recording(rec)
    assert result.subject == "S01" and result.epochs_all.n_epochs == 20  # 4 conditions x 5
    assert result.report is not None and len(result.report.keep) == 20
    assert np.array_equal(result.epochs.data, result.report.apply(result.epochs_all).data)
    assert result.epochs.n_epochs == int(result.report.keep.sum()) >= 1
    assert result.eog is not None and result.eog.removed == (0,)
    assert result.cleaned.data.shape == rec.data.shape
    assert result.epochs_all.tmin == -2.0 and result.epochs_all.n_times == 2000  # 8 s at 250 Hz


def test_baseline_is_applied_and_optional(rec):
    with_base = ee.process_recording(rec, ee.PipelineConfig(rejection=None)).epochs_all
    i0, i1 = with_base.sample_range(-2.0, 0.0)  # the baseline interval, end points included
    assert np.abs(with_base.data[:, :, i0 : i1 + 1].mean(axis=2)).max() < 1e-12
    without = ee.process_recording(rec, ee.PipelineConfig(rejection=None, baseline=None)).epochs_all
    assert np.abs(without.data[:, :, i0 : i1 + 1].mean(axis=2)).max() > 1e-9
    assert not np.array_equal(with_base.data, without.data)


def test_epoch_window_follows_the_configuration(rec):
    cfg = ee.PipelineConfig(epoch=(-1.0, 3.0), baseline=(-1.0, 0.0), rejection=None)
    epochs = ee.process_recording(rec, cfg).epochs_all
    assert epochs.tmin == -1.0 and epochs.n_times == 1000


def test_an_impossible_amplitude_limit_rejects_everything_and_says_so(rec):
    cfg = ee.PipelineConfig(rejection=ee.RejectionConfig(max_abs=1e-9))
    with pytest.raises(ValueError, match="every epoch was rejected"):
        ee.process_recording(rec, cfg)


def test_process_group_keeps_the_order_and_refuses_nothing(group_results):
    recs, results = group_results
    assert [r.subject for r in results] == [rec.subject for rec in recs]
    assert all(r.epochs.n_epochs >= 20 for r in results)  # of 32, a few rejected
    with pytest.raises(ValueError, match="no recordings"):
        ee.process_group([])


def test_blinks_are_removed_from_frontal_channels(group_results):
    recs, results = group_results
    plain = ee.PipelineConfig(remove_blinks=False, rejection=None)
    unfixed = [ee.process_recording(r, plain) for r in recs]
    blink = [r.truth["blink_signal"] for r in recs]
    after = [abs(blink_residual(res, b)) for res, b in zip(results, blink, strict=True)]
    before = [abs(blink_residual(res, b)) for res, b in zip(unfixed, blink, strict=True)]
    assert all(b > 0.4 for b in before)
    assert all(a < 0.3 and a < b for a, b in zip(after, before, strict=True))
    assert np.mean(after) < 0.1
    far_from_the_eyes = abs(blink_residual(results[0], blink[0], "Oz"))
    assert far_from_the_eyes < 0.1


def test_group_pipeline_recovers_the_simulated_p300(group_epochs, group_results):
    recs, _ = group_results
    peaks = ee.find_group_peaks(ee.group_erp(group_epochs, baseline=(-0.2, 0.0)), ["Cz"])["Cz"]
    p300 = {p.name: p for p in peaks}["P300"]
    true = ee.SimConfig().erp_latency[1] + float(np.mean([r.truth["latency_shift"] for r in recs]))
    assert p300.found and p300.latency == pytest.approx(true, abs=0.04)
