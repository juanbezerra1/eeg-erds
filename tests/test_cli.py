import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import eegerds as ee
from eegerds import cli
from eegerds.cli import app

runner = CliRunner()


def run(*args):
    return runner.invoke(app, [str(a) for a in args])


@pytest.fixture(scope="module")
def epoch_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("epochs")
    result = run("simulate", out, "--subjects", 5, "--trials", 5, "--seed", 3)
    assert result.exit_code == 0, result.output
    return out


@pytest.fixture(scope="module")
def files(epoch_dir):
    return sorted(epoch_dir.glob("*.npz"))


def test_version_and_help():
    result = run("--version")
    assert result.exit_code == 0 and result.output.strip() == f"eegerds {ee.__version__}"
    helped = run("--help")
    assert helped.exit_code == 0
    for command in ("demo", "simulate", "erds", "peaks", "features"):
        assert command in helped.output
    assert "Usage" in run().output  # no arguments: show the help


def test_python_dash_m_runs_the_same_command_line():
    done = subprocess.run(
        [sys.executable, "-I", "-m", "eegerds", "--version"],
        capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip
    assert done.returncode == 0 and done.stdout.strip() == f"eegerds {ee.__version__}"


def test_fif_files_need_mne_and_the_message_says_how_to_get_it(tmp_path, monkeypatch):
    fif = tmp_path / "S01-epo.fif"
    fif.write_bytes(b"")
    monkeypatch.setitem(sys.modules, "mne", None)  # makes ``import mne`` fail
    result = run("erds", fif, fif, "--tag-a", "left", "--tag-b", "right")
    assert result.exit_code != 0
    assert "pip install" in result.output and "mne" in result.output


def test_simulate_writes_one_clean_epochs_file_per_subject(epoch_dir, files):
    assert [f.name for f in files] == [f"S0{i}.npz" for i in range(1, 6)]
    first = ee.load_epochs(files[0])
    assert first.subject == "S01" and first.n_channels == 12 and first.n_epochs >= 10
    assert set(t for label in first.labels for t in label.split("/")) == {
        "left",
        "right",
        "low",
        "high",
    }


def test_erds_compares_two_conditions_and_writes_a_table(files, tmp_path):
    out = tmp_path / "w.csv"
    result = run("erds", *files, "--tag-a", "left", "--tag-b", "right", "--out", out)
    assert result.exit_code == 0, result.output
    assert (
        "windows significant" in result.output
        and "Smallest attainable p with 5 subjects" in result.output
    )
    table = pd.read_csv(out)
    assert set(table["series"]) == {"C3", "C4"} and (table["n"] == 5).all()
    assert table["p_floor"].iloc[0] == pytest.approx(2 / 2**5)
    assert {"mean_a", "mean_b", "p_adj", "effect", "significant"} <= set(table.columns)


def test_erds_options_reach_the_analysis(files, tmp_path):
    out = tmp_path / "beta.csv"
    result = run("erds", *files, "--tag-a", "left", "--tag-b", "right", "--band", 13, 30,
                 "--channel", "Cz", "--window", 1.0, "--method", "signflip_maxt", "--alpha", 0.1,
                 "--out", out)  # fmt: skip
    assert result.exit_code == 0, result.output
    table = pd.read_csv(out)
    assert set(table["series"]) == {"Cz"} and table["p_raw"].isna().all()
    assert (table["t_stop"] - table["t_start"]).round(6).eq(1.0).all()
    assert "signflip_maxt" in result.output and "alpha=0.1" in result.output


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--channel", "Fz9"], "not found"),
        (["--method", "bogus"], "method must be one of"),
        (["--tag-a", "nothing"], "no epochs tagged"),
    ],
)
def test_erds_reports_problems_without_a_traceback(files, tmp_path, extra, message):
    args = [
        "erds",
        *files,
        "--tag-a",
        "left",
        "--tag-b",
        "right",
        "--out",
        tmp_path / "x.csv",
        *extra,
    ]
    result = run(*args)
    assert result.exit_code == 2 and "error:" in result.output and message in result.output
    assert "Traceback" not in result.output and not (tmp_path / "x.csv").exists()


def test_bad_input_files_are_explained(tmp_path):
    missing = run("features", tmp_path / "none.npz", "--class", "left")
    assert missing.exit_code == 2 and "does not exist" in missing.output
    wrong = tmp_path / "data.csv"
    wrong.write_text("a,b\n1,2\n")
    result = run("features", wrong, "--class", "left")
    assert result.exit_code == 2 and "expected a .npz or .fif" in result.output
    foreign = tmp_path / "foreign.npz"
    np.savez(foreign, x=np.arange(3))
    result = run("features", foreign, "--class", "left")
    assert result.exit_code == 2 and "not an eegerds epochs file" in result.output


def test_peaks_pooled_and_by_condition(files, tmp_path):
    out = tmp_path / "p.csv"
    result = run("peaks", *files, "--channel", "Cz", "--channel", "Pz", "--out", out)
    assert result.exit_code == 0, result.output
    pooled = pd.read_csv(out)
    assert list(pooled.columns) == [
        "channel",
        "component",
        "latency",
        "amplitude",
        "found",
        "at_edge",
    ]
    assert len(pooled) == 6 and "Cz:" in result.output and "P300" in result.output
    p300 = pooled[(pooled["channel"] == "Cz") & (pooled["component"] == "P300")].iloc[0]
    assert 0.25 < p300["latency"] < 0.40 and p300["amplitude"] > 0

    out2 = tmp_path / "p2.csv"
    result = run("peaks", *files, "--tag-a", "high", "--tag-b", "low", "--out", out2)
    assert result.exit_code == 0, result.output
    compared = pd.read_csv(out2)
    assert {"mean_a", "mean_b", "p_adj", "p_floor"} <= set(compared.columns) and len(compared) == 3


def test_peaks_wants_both_tags_or_neither(files, tmp_path):
    result = run("peaks", *files, "--tag-a", "high", "--out", tmp_path / "x.csv")
    assert result.exit_code == 2 and "go together" in result.output


def test_features_follow_the_csv_contract(files, tmp_path):
    out = tmp_path / "f.csv"
    result = run("features", *files, "--class", "left", "--class", "right",
                 "--channel", "C3", "--channel", "C4", "--out", out)  # fmt: skip
    assert result.exit_code == 0, result.output
    table = pd.read_csv(out)
    assert list(table.columns[:3]) == ["subject", "level", "trial"]
    assert set(table["level"]) == {"left", "right"} and table["subject"].nunique() == 5
    assert all(c.split("_")[0] in {"C3", "C4"} for c in table.columns[3:])
    assert table.shape[1] - 3 == 2 * 2 * 5  # 2 channels x 2 bands x 5 one-second windows
    assert f"{len(table)} rows" in result.output and "warning" not in result.output
    assert np.isfinite(table.iloc[:, 3:].to_numpy()).all()


def test_the_command_line_reads_fif_files(tmp_path):
    mne = pytest.importorskip("mne")
    for i in range(3):
        rng = np.random.default_rng(i)
        data = rng.standard_normal((8, 3, 2000)) * 1e-6
        labels = ("left/low", "right/low") * 4
        epochs = ee.Epochs(data, 250.0, -2.0, ("C3", "Cz", "C4"), labels, "")
        converted = ee.to_mne(epochs)
        assert isinstance(converted, mne.EpochsArray)
        converted.save(tmp_path / f"P{i}-epo.fif", verbose="ERROR")
    paths = sorted(tmp_path.glob("*.fif"))
    out = tmp_path / "f.csv"
    result = run("features", *paths, "--class", "left", "--class", "right", "--out", out)
    assert result.exit_code == 0, result.output
    table = pd.read_csv(out)
    assert set(table["subject"]) == {"P0-epo", "P1-epo", "P2-epo"}  # named after the files
    assert set(table["level"]) == {"left", "right"}


def test_demo_command_forwards_its_options(monkeypatch, quick_demo, tmp_path):
    seen = {}

    def fake(out, **kwargs):
        seen.update(out=out, **kwargs)
        return quick_demo

    monkeypatch.setattr(cli, "run_demo", fake)
    result = run(
        "demo", "--out", tmp_path / "d", "--subjects", 4, "--trials", 3, "--seed", 9, "--quick"
    )
    assert result.exit_code == 0, result.output
    assert seen == {"out": tmp_path / "d", "n_subjects": 4, "n_trials": 3, "seed": 9, "quick": True}
    assert "# eeg-erds demo (synthetic data)" in result.output and "Wrote" in result.output
