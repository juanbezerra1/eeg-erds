"""Shared fixtures. Simulations are cached per session: they are the slowest part of the tests."""

from __future__ import annotations

import pytest

import eegerds as ee

#: Clean simulation: no blinks, artefacts or line noise, so tests can compare with exact truth.
CLEAN = ee.SimConfig(n_trials=25, blink_rate=0.0, artifact_rate=0.0, line_uv=0.0)


@pytest.fixture(scope="session")
def clean_rec() -> ee.Recording:
    return ee.simulate_recording(CLEAN, seed=11)


@pytest.fixture(scope="session")
def clean_epochs(clean_rec: ee.Recording) -> ee.Epochs:
    return ee.epoch(clean_rec, -2.0, 6.0, baseline=(-2.0, 0.0))


@pytest.fixture(scope="session")
def group_results() -> tuple[list[ee.Recording], list[ee.SubjectResult]]:
    """Six default subjects (blinks and artefacts on) taken through the whole pipeline."""
    from eegerds.artifacts import RejectionConfig

    recs = ee.simulate_group(6, ee.SimConfig(n_trials=8), seed=4)
    cfg = ee.PipelineConfig(rejection=RejectionConfig(max_window_sd_ratio=3.0))
    return recs, ee.process_group(recs, cfg)


@pytest.fixture(scope="session")
def group_epochs(group_results) -> list[ee.Epochs]:
    return [r.epochs for r in group_results[1]]


@pytest.fixture(scope="session")
def quick_demo(tmp_path_factory: pytest.TempPathFactory):
    from eegerds.demo import run_demo

    return run_demo(tmp_path_factory.mktemp("demo"), quick=True, seed=2)
