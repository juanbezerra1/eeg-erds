import numpy as np
import pandas as pd
import pytest

import eegerds as ee
from eegerds.features import DEFAULT_BANDS, default_windows

FS = 250.0
#: Mirrors ``subjectwise.RESERVED_COLUMNS``: names that are bookkeeping, never features.
RESERVED = {
    "subject", "session", "trial", "segment_id", "kind", "t_start", "t_stop", "label", "level", "x",
}  # fmt: skip


def sine_epochs(n_epochs=6, tmin=-2.0, tmax=4.0, labels=None, subject="S1", seed=0):
    """Three channels with a 10.5 Hz rhythm: amplitude x2 after 0 s, unchanged, x0.5 after 0 s."""
    rng = np.random.default_rng(seed)
    t = tmin + np.arange(round((tmax - tmin) * FS)) / FS
    step = (t >= 0).astype(float)
    gains = np.array([1 + step, np.ones_like(t), 1 - 0.5 * step])  # (3, n_times)
    phase = rng.uniform(0, 2 * np.pi, (n_epochs, 1, 1))
    x = gains[None] * np.sin(2 * np.pi * 10.5 * t[None, None, :] + phase) * 1e-6
    return ee.Epochs(
        x, FS, tmin, ("A", "B", "C"), tuple(labels or ["a"] * n_epochs), subject
    )  # fmt: skip


def test_default_windows():
    epochs = sine_epochs()
    wins = default_windows(epochs, width=1.0, start=0.0)
    assert wins[0] == (0.0, 1.0) and all(b - a == 1.0 for a, b in wins)
    assert wins[-1][1] <= epochs.times[-1] + 1e-9  # never past the last sample
    assert len(default_windows(epochs, width=0.5, start=1.0)) == 5
    with pytest.raises(ValueError, match="too short"):
        default_windows(epochs, width=9.0)


def test_log_power_ratios_match_the_known_amplitude_steps():
    f = ee.trial_features(
        sine_epochs(), bands={"mu": (8.0, 13.0)}, windows=[(1.0, 2.0)], reference=(-1.5, -0.5)
    )
    assert f["A_mu_1-2s"].to_numpy() == pytest.approx(np.log(4.0), abs=0.05)  # amplitude x2
    assert f["B_mu_1-2s"].to_numpy() == pytest.approx(0.0, abs=0.05)
    assert f["C_mu_1-2s"].to_numpy() == pytest.approx(np.log(0.25), abs=0.05)  # amplitude x0.5


def test_the_contract_columns_and_names():
    epochs = sine_epochs(labels=["left/low", "right/high"] * 3)
    f = ee.trial_features(epochs)
    assert list(f.columns[:3]) == ["subject", "level", "trial"]
    features = [c for c in f.columns if c not in RESERVED]
    expected = [
        f"{ch}_{band}_{a}-{a + 1}s" for band in DEFAULT_BANDS for a in range(3) for ch in "ABC"
    ]
    assert features == expected  # band-major, then window, then channel
    assert len(f) == 6 and set(f["subject"]) == {"S1"}
    assert list(f["level"]) == ["left/low", "right/high"] * 3  # the whole label by default
    assert np.array_equal(f["trial"], np.arange(6))
    assert not set(features) & RESERVED
    assert f[features].dtypes.eq(float).all() and np.isfinite(f[features].to_numpy()).all()


def test_classes_reduce_composite_labels_and_drop_the_others():
    labels = ["left/low", "right/high", "rest", "left/high", "right/low", "rest"]
    f = ee.trial_features(sine_epochs(labels=labels), classes=("left", "right"))
    assert list(f["level"]) == ["left", "right", "left", "right"]
    assert list(f["trial"]) == [0, 1, 3, 4]  # keeps the epoch's own index
    assert list(ee.trial_features(sine_epochs(labels=labels), classes=("high",))["level"]) == [
        "high",
        "high",
    ]
    with pytest.raises(ValueError, match="more than one"):
        ee.trial_features(sine_epochs(labels=["left/right"] * 6), classes=("left", "right"))


def test_label_column_can_be_renamed_and_subject_has_a_fallback():
    f = ee.trial_features(sine_epochs(subject=""), label_column="y")
    assert "y" in f.columns and "level" not in f.columns and set(f["subject"]) == {"S"}


def test_features_table_stacks_subjects():
    subjects = [sine_epochs(n_epochs=4, subject=f"S{i}", seed=i) for i in range(3)]
    table = ee.features_table(subjects, classes=("a",), bands={"mu": (8.0, 13.0)})
    assert len(table) == 12 and list(table["subject"].unique()) == ["S0", "S1", "S2"]
    trials = {name: list(group["trial"]) for name, group in table.groupby("subject")}
    assert trials == {"S0": [0, 1, 2, 3], "S1": [0, 1, 2, 3], "S2": [0, 1, 2, 3]}
    with pytest.raises(ValueError, match="no subjects"):
        ee.features_table([])


def test_filtering_each_epoch_does_not_bias_the_reference():
    """Edge padding: a reference that starts at the first sample must not read ~5 % too low."""
    rng = np.random.default_rng(3)
    n = round(8 * FS)
    noise = ee.bandpass(rng.standard_normal((1, 60 * n)), FS, 8.0, 13.0)  # continuous, then cut
    x = noise.reshape(1, 60, n).transpose(1, 0, 2)
    epochs = ee.Epochs(x, FS, -2.0, ("A",), ("a",) * 60, "S")
    f = ee.trial_features(
        epochs, bands={"mu": (8.0, 13.0)}, windows=[(2.0, 3.0)], reference=(-2.0, -1.0)
    )
    # same band, unchanged power: mean log ratio is ~0 (a -5 % edge error would give +0.05)
    assert abs(f["A_mu_2-3s"].mean()) < 0.06


def test_input_checks():
    epochs = sine_epochs()
    with pytest.raises(ValueError, match="must not be empty"):
        ee.trial_features(epochs, bands={})
    with pytest.raises(ValueError, match="must not be empty"):
        ee.trial_features(epochs, windows=[])
    with pytest.raises(ValueError):  # reference outside the epoch
        ee.trial_features(epochs, reference=(-9.0, -8.0))
    flat = sine_epochs()
    flat.data[2, 1, :] = 0.0  # one channel flat-lined in one epoch: its power ratio is undefined
    with pytest.raises(ValueError, match="reference power"):
        ee.trial_features(flat)


def test_simulated_features_separate_the_sides_on_the_contralateral_channels(group_epochs):
    table = ee.features_table(group_epochs, classes=("left", "right"), bands={"mu": (8.0, 13.0)},
                              windows=[(1.0, 2.0)])  # fmt: skip
    col = "C4_mu_1-2s"
    by_side = table.groupby("level")[col].mean()
    assert by_side["left"] < by_side["right"] - 0.15  # left hand: C4 power falls more
    assert table[col].notna().all()
    assert isinstance(table, pd.DataFrame)
