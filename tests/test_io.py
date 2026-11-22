import numpy as np
import pytest

import eegerds as ee


def small_epochs(n=5, subject="S9"):
    rng = np.random.default_rng(0)
    return ee.Epochs(
        rng.standard_normal((n, 3, 100)) * 1e-6, 100.0, -0.5, ("Fz", "Cz", "Pz"),
        tuple(["left/low", "right/high"] * n)[:n], subject,
    )  # fmt: skip


def test_npz_roundtrip_is_exact(tmp_path):
    epochs = small_epochs()
    path = ee.save_epochs(tmp_path / "sub" / "s9", epochs)  # the folder and suffix are created
    assert path.suffix == ".npz" and path.exists()
    back = ee.load_epochs(path)
    assert np.array_equal(back.data, epochs.data) and back.data.dtype == epochs.data.dtype
    assert (back.fs, back.tmin, back.ch_names, back.labels, back.subject) == (
        epochs.fs, epochs.tmin, epochs.ch_names, epochs.labels, epochs.subject,
    )  # fmt: skip


def test_an_empty_subject_survives_the_roundtrip(tmp_path):
    back = ee.load_epochs(ee.save_epochs(tmp_path / "x.npz", small_epochs(subject="")))
    assert back.subject == ""


def test_loading_never_unpickles(tmp_path):
    path = tmp_path / "evil.npz"
    np.savez(path, data=np.array([object()], dtype=object), fs=1.0, tmin=0.0,
             ch_names=np.array(["a"]), labels=np.array(["x"]))  # fmt: skip
    with pytest.raises(ValueError, match="pickle"):
        ee.load_epochs(path)


def test_a_foreign_npz_is_refused_with_a_clear_message(tmp_path):
    path = tmp_path / "other.npz"
    np.savez(path, something=np.arange(3))
    with pytest.raises(ValueError, match="not an eegerds epochs file"):
        ee.load_epochs(path)


def test_the_mne_roundtrip_keeps_data_labels_and_timing():
    mne = pytest.importorskip("mne")
    epochs = small_epochs()
    converted = ee.to_mne(epochs)
    assert isinstance(converted, mne.EpochsArray)
    assert converted.info["sfreq"] == 100.0 and converted.tmin == -0.5
    back = ee.from_mne(converted, subject="S9")
    assert np.allclose(back.data, epochs.data) and back.ch_names == epochs.ch_names
    assert back.labels == epochs.labels and back.fs == 100.0 and back.tmin == -0.5
    assert back.subject == "S9"


def test_from_mne_keeps_only_eeg_channels_by_default():
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(1)
    info = mne.create_info(["Cz", "Pz", "EOG1"], 100.0, ["eeg", "eeg", "eog"])
    events = np.array([[0, 0, 1], [200, 0, 2], [400, 0, 1]])
    raw = mne.EpochsArray(rng.standard_normal((3, 3, 100)) * 1e-6, info, events=events,
                          tmin=0.0, event_id={"a": 1, "b": 2}, verbose="ERROR")  # fmt: skip
    out = ee.from_mne(raw)
    assert out.ch_names == ("Cz", "Pz") and out.labels == ("a", "b", "a")
    with_eog = ee.from_mne(raw, picks=["Cz", "EOG1"])
    assert with_eog.ch_names == ("Cz", "EOG1")
