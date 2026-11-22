import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import eegerds as ee
from eegerds.stats import METHODS

TIMES = np.array([0.25, 0.75, 1.25])
COLUMNS = [
    "series", "time", "t_start", "t_stop", "n", "mean_a", "mean_b",
    "statistic", "p_raw", "p_adj", "effect", "p_floor", "significant",
]  # fmt: skip


# ---------------------------------------------------------------- Benjamini-Hochberg


def test_bh_known_example():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(ee.fdr_bh(p), [0.02, 0.04, 0.04, 0.02])


def test_bh_ignores_nan_and_keeps_the_shape():
    p = np.array([[0.01, np.nan], [0.04, 0.5]])
    out = ee.fdr_bh(p)
    assert out.shape == p.shape and np.isnan(out[0, 1])
    assert np.allclose([out[0, 0], out[1, 0], out[1, 1]], [0.03, 0.06, 0.5])  # m = 3, not 4


def test_bh_degenerate_inputs():
    assert np.isnan(ee.fdr_bh(np.array([np.nan, np.nan]))).all()
    assert ee.fdr_bh(np.array([])).size == 0
    assert ee.fdr_bh(np.array([0.2])) == pytest.approx(0.2)


def test_bh_properties():
    p = np.random.default_rng(1).random(40)
    adjusted = ee.fdr_bh(p)
    assert np.all(adjusted >= p - 1e-15) and np.all(adjusted <= 1)
    assert np.all(np.diff(adjusted[np.argsort(p)]) >= -1e-15)  # never reorders the p-values


@pytest.mark.skipif(not hasattr(stats, "false_discovery_control"), reason="needs scipy >= 1.11")
def test_bh_agrees_with_scipy():
    p = np.random.default_rng(0).random(50) ** 2
    assert np.allclose(ee.fdr_bh(p), stats.false_discovery_control(p, method="bh"))


# ---------------------------------------------------------------- paired Wilcoxon


def test_wilcoxon_matches_scipy_and_the_rank_biserial_effect():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((10, 3)) + [0.0, 0.8, 2.0]
    b = rng.standard_normal((10, 3))
    stat, p, effect = ee.paired_wilcoxon(a, b)
    for j in range(3):
        d = a[:, j] - b[:, j]
        ref = stats.wilcoxon(d)
        assert stat[j] == pytest.approx(ref.statistic) and p[j] == pytest.approx(ref.pvalue)
        ranks = stats.rankdata(np.abs(d))
        assert effect[j] == pytest.approx((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())
    assert effect[2] > effect[0]


def test_wilcoxon_perfect_separation_reaches_the_exact_floor():
    a = np.arange(1.0, 9.0)[:, None]
    stat, p, effect = ee.paired_wilcoxon(a, np.zeros_like(a))
    assert p[0] == pytest.approx(2 / 2**8) and effect[0] == 1.0 and stat[0] == 0.0
    _, p_swapped, effect_swapped = ee.paired_wilcoxon(np.zeros_like(a), a)
    assert p_swapped[0] == pytest.approx(2 / 2**8) and effect_swapped[0] == -1.0


def test_wilcoxon_columns_without_usable_pairs():
    a = np.array([[1.0, 0.0, np.nan]] * 4) + np.arange(4)[:, None] * [1.0, 0.0, 0.0]
    b = np.array([[0.0, 0.0, np.nan]] * 4)
    stat, p, effect = ee.paired_wilcoxon(a, b)
    assert (stat[1], p[1], effect[1]) == (0.0, 1.0, 0.0)  # every difference is zero
    assert np.isnan([stat[2], p[2], effect[2]]).all()  # nothing valid at all


def test_wilcoxon_drops_a_subject_only_from_the_column_where_it_is_missing():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((9, 2)), rng.standard_normal((9, 2))
    holed = a.copy()
    holed[0, 0] = np.nan
    p_holed = ee.paired_wilcoxon(holed, b)[1]
    assert p_holed[0] == pytest.approx(ee.paired_wilcoxon(a[1:, :1], b[1:, :1])[1][0])
    assert p_holed[1] == pytest.approx(ee.paired_wilcoxon(a, b)[1][1])


def test_wilcoxon_accepts_vectors_checks_shapes_and_stays_quiet_with_ties():
    out = ee.paired_wilcoxon(np.arange(1.0, 9.0), np.zeros(8))
    assert all(x.shape == (1,) for x in out)
    with pytest.raises(ValueError, match="same shape"):
        ee.paired_wilcoxon(np.zeros((5, 2)), np.zeros((5, 3)))
    tied = np.array([[1.0], [1.0], [2.0], [2.0], [3.0], [-1.0], [3.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ee.paired_wilcoxon(tied, np.zeros_like(tied))


# ---------------------------------------------------------------- sign-flip max-T


def brute_force_maxt(d):
    """Every sign pattern, one scipy t-test at a time."""
    t_obs = stats.ttest_1samp(d, 0.0).statistic
    max_abs = np.array(
        [
            np.abs(stats.ttest_1samp(d * np.array(s)[:, None], 0.0).statistic).max()
            for s in itertools.product([-1.0, 1.0], repeat=d.shape[0])
        ]
    )
    return t_obs, np.array([(max_abs >= abs(t) * (1 - 1e-12)).mean() for t in t_obs])


def test_signflip_matches_brute_force_enumeration():
    d = np.random.default_rng(4).standard_normal((6, 3)) + [0.0, 0.5, 1.5]
    t, p = ee.signflip_maxt(d)
    t_ref, p_ref = brute_force_maxt(d)
    assert np.allclose(t, t_ref) and np.allclose(p, p_ref)


def test_signflip_smallest_possible_p_is_two_over_two_to_the_n():
    d = np.abs(np.random.default_rng(5).standard_normal(6)) + 0.1  # six positive differences
    t, p = ee.signflip_maxt(d)  # a vector is one test
    assert t.shape == p.shape == (1,) and p[0] == pytest.approx(2 / 2**6)
    two = np.column_stack([d, d * 0.5])
    assert ee.signflip_maxt(two)[1].min() == pytest.approx(2 / 2**6)


def test_signflip_adjusts_for_the_number_of_tests():
    rng = np.random.default_rng(6)
    d = rng.standard_normal((8, 12))
    d[:, 0] += 1.0
    single = ee.signflip_maxt(d[:, :1])[1][0]
    family = ee.signflip_maxt(d)[1][0]
    assert family >= single  # looking at more tests can only make each p larger


@pytest.mark.parametrize("c", [0.3, 1 / 3, 123.456, -0.7, 1e-6])
@pytest.mark.parametrize("n", [3, 7, 12])
def test_signflip_with_identical_differences_is_numerically_stable(c, n):
    """Zero variance is rounding noise; p = 0 would be impossible for a permutation test."""
    p = ee.signflip_maxt(np.full((n, 1), c))[1][0]
    assert p == pytest.approx(2 / 2**n)


def test_signflip_of_all_zero_differences_is_one():
    t, p = ee.signflip_maxt(np.zeros((5, 2)))
    assert np.array_equal(t, [0.0, 0.0]) and np.array_equal(p, [1.0, 1.0])


def test_signflip_random_branch_is_reproducible_and_close_to_the_exact_answer():
    d = np.random.default_rng(7).standard_normal((9, 4)) + [0.0, 0.3, 0.8, 1.2]
    exact = ee.signflip_maxt(d)[1]
    a = ee.signflip_maxt(d, n_perm=20000, max_exhaustive=3, seed=1)[1]
    b = ee.signflip_maxt(d, n_perm=20000, max_exhaustive=3, seed=1)[1]
    c = ee.signflip_maxt(d, n_perm=20000, max_exhaustive=3, seed=2)[1]
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.abs(a - exact).max() < 0.02 and np.abs(c - exact).max() < 0.02
    assert (a >= 1 / 20001).all()  # (1 + count) / (1 + n_perm): never exactly zero


def test_signflip_input_checks():
    with pytest.raises(ValueError, match="at least 2"):
        ee.signflip_maxt(np.ones((1, 3)))
    with pytest.raises(ValueError, match="NaN"):
        ee.signflip_maxt(np.array([[1.0], [np.nan], [2.0]]))


# ---------------------------------------------------------------- tables


def cube(n_sub=8, n_series=2, n_win=3, seed=0):
    rng = np.random.default_rng(seed)
    shape = (n_sub, n_series, n_win)
    return rng.standard_normal(shape), rng.standard_normal(shape)


def test_compare_series_layout_and_values():
    a, b = cube()
    a[:, 0, 1] = b[:, 0, 1] + np.arange(1, 9)  # series 0, window 1: all 8 subjects higher in a
    t = ee.compare_series(a, b, ["C3", "C4"], TIMES, window=0.5, method="wilcoxon")
    assert list(t.columns) == COLUMNS and len(t) == 6
    assert list(t["series"]) == ["C3"] * 3 + ["C4"] * 3
    assert np.allclose(t["time"], np.tile(TIMES, 2))
    assert np.allclose(t["t_start"], t["time"] - 0.25) and np.allclose(
        t["t_stop"], t["time"] + 0.25
    )
    assert (t["n"] == 8).all()
    row = t.iloc[1]
    assert row["p_raw"] == pytest.approx(2 / 2**8) and row["p_adj"] == row["p_raw"]
    assert row["effect"] == 1.0 and row["p_floor"] == pytest.approx(2 / 2**8)
    assert row["mean_a"] == pytest.approx(a[:, 0, 1].mean())
    assert row["mean_b"] == pytest.approx(b[:, 0, 1].mean())
    assert bool(row["significant"]) and not t["significant"].drop(index=1).any()


def test_with_six_subjects_even_a_perfect_effect_cannot_survive_fdr_over_three_tests():
    a, b = cube(6, 1, 3, seed=1)
    a[:, 0, 0] = b[:, 0, 0] + np.arange(1, 7)  # every subject higher: the smallest p there is
    a[:, 0, 1:] = b[:, 0, 1:] + np.array([1, -2, 3, -4, 5, -6])[:, None]  # no consistent effect
    t = ee.compare_series(a, b, ["Cz"], TIMES, 0.5)  # default: wilcoxon_fdr
    assert t["p_raw"].iloc[0] == pytest.approx(2 / 2**6)
    assert t["p_adj"].iloc[0] == pytest.approx(3 * 2 / 2**6)  # BH over 3 tests: 0.094 > 0.05
    assert not t["significant"].any()
    assert (t["p_floor"] == 2 / 2**6).all()  # the table says why: this is as small as p gets


def test_missing_values_shrink_n_and_raise_the_floor_in_that_cell_only():
    a, b = cube(seed=2)
    a[3, 1, 2] = np.nan
    t = ee.compare_series(a, b, ["C3", "C4"], TIMES, 0.5, method="wilcoxon")
    cell = t.iloc[5]
    keep = np.arange(8) != 3
    assert cell["n"] == 7 and cell["p_floor"] == pytest.approx(2**-6)
    assert cell["mean_a"] == pytest.approx(a[keep, 1, 2].mean())
    assert cell["mean_b"] == pytest.approx(b[keep, 1, 2].mean())
    assert (t["n"].drop(index=5) == 8).all()


def test_wilcoxon_methods_differ_only_in_the_adjustment():
    a, b = cube(seed=3)
    a[:, 0, 0] += 1.5
    raw = ee.compare_series(a, b, ["C3", "C4"], TIMES, 0.5, method="wilcoxon")
    fdr = ee.compare_series(a, b, ["C3", "C4"], TIMES, 0.5, method="wilcoxon_fdr")
    assert np.array_equal(raw["p_raw"], fdr["p_raw"]) and np.array_equal(raw["p_adj"], raw["p_raw"])
    assert np.allclose(fdr["p_adj"], ee.fdr_bh(raw["p_raw"].to_numpy()))
    assert (fdr["p_adj"] >= fdr["p_raw"]).all()


def test_signflip_method_gives_adjusted_p_only():
    a, b = cube(seed=4)
    a[:, 0, 0] += 3.0
    t = ee.compare_series(a, b, ["C3", "C4"], TIMES, 0.5, method="signflip_maxt")
    diff = (a - b).reshape(8, -1)
    _, p_adj = ee.signflip_maxt(diff)
    assert t["p_raw"].isna().all()
    assert np.allclose(t["p_adj"], p_adj)
    assert np.allclose(t["statistic"], stats.ttest_1samp(diff, 0.0).statistic)
    assert bool(t["significant"].iloc[0])
    a[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="complete data"):
        ee.compare_series(a, b, ["C3", "C4"], TIMES, 0.5, method="signflip_maxt")


@pytest.mark.parametrize(
    "a_shape, b_shape, names, times, message",
    [
        ((8, 2, 3), (8, 2, 4), ["x", "y"], TIMES, "both be"),
        ((8, 6), (8, 6), ["x"], TIMES, "both be"),
        ((8, 2, 3), (8, 2, 3), ["x"], TIMES, "do not match"),
        ((8, 2, 3), (8, 2, 3), ["x", "y"], TIMES[:2], "do not match"),
    ],
)
def test_compare_series_argument_checks(a_shape, b_shape, names, times, message):
    with pytest.raises(ValueError, match=message):
        ee.compare_series(np.zeros(a_shape), np.zeros(b_shape), names, times, 0.5)


def test_unknown_method_is_refused_everywhere():
    a, b = cube()
    assert set(METHODS) == {"wilcoxon_fdr", "wilcoxon", "signflip_maxt"}
    with pytest.raises(ValueError, match="method must be one of"):
        ee.compare_series(a, b, ["x", "y"], TIMES, 0.5, method="t-test")  # type: ignore[arg-type]
    amp = pd.DataFrame({"subject": ["S0"], "channel": ["Cz"], "component": ["P300"],
                        "latency": [0.3], "amplitude": [1.0]})  # fmt: skip
    with pytest.raises(ValueError, match="method must be one of"):
        ee.compare_amplitudes(amp, amp, method="bonferroni")  # type: ignore[arg-type]


def test_family_wise_error_under_the_null_is_controlled_only_by_the_corrections():
    """Ten windows, eight subjects, no effect anywhere: how often does *something* come out?"""
    rng = np.random.default_rng(7)
    n_rep, k = 300, 10
    hits = dict.fromkeys(("wilcoxon", "wilcoxon_fdr", "signflip_maxt"), 0)
    for _ in range(n_rep):
        a, b = rng.standard_normal((8, 1, k)), rng.standard_normal((8, 1, k))
        for method in hits:
            t = ee.compare_series(a, b, ["X"], np.arange(k) * 0.5, 0.5, method=method)
            hits[method] += bool(t["significant"].any())
    rate = {m: h / n_rep for m, h in hits.items()}
    assert rate["wilcoxon"] > 0.2  # uncorrected: roughly one false alarm in three experiments
    assert rate["wilcoxon_fdr"] < 0.09 and rate["signflip_maxt"] < 0.09  # nominal 0.05


# ---------------------------------------------------------------- GroupErds / amplitudes


def make_group(values, subjects=None, channels=("C3", "C4")):
    n_sub, _, n_win = values.shape
    return ee.GroupErds(
        times=np.arange(n_win) * 0.25 + 0.25,
        values=values,
        subjects=subjects or tuple(f"S{i}" for i in range(n_sub)),
        ch_names=channels,
        config=ee.ErdsConfig(window=0.5),
    )


def test_compare_conditions_follows_the_requested_channel_order():
    va, vb = cube(8, 2, 4, seed=5)
    va[:, 1, 0] = vb[:, 1, 0] + np.arange(1, 9)  # C4, first window
    t = ee.compare_conditions(make_group(va), make_group(vb), ["C4", "C3"], method="wilcoxon")
    assert list(t["series"].unique()) == ["C4", "C3"] and len(t) == 8
    first = t.iloc[0]
    assert first["p_raw"] == pytest.approx(2 / 2**8) and first["effect"] == 1.0
    assert first["t_stop"] - first["t_start"] == pytest.approx(0.5)
    assert first["mean_a"] == pytest.approx(va[:, 1, 0].mean())


def test_compare_conditions_requires_the_same_subjects_and_axes():
    va, vb = cube(8, 2, 4, seed=5)
    names = tuple(f"S{i}" for i in range(8))
    with pytest.raises(ValueError, match="same subjects"):
        ee.compare_conditions(make_group(va), make_group(vb, names[::-1]), ["C3"])
    with pytest.raises(ValueError, match="share channels"):
        ee.compare_conditions(make_group(va), make_group(vb[:, :, :3]), ["C3"])
    with pytest.raises(KeyError):
        ee.compare_conditions(make_group(va), make_group(vb), ["Fz"])


def test_compare_channels_labels_rows_after_the_pair():
    values, _ = cube(8, 2, 4, seed=6)
    t = ee.compare_channels(make_group(values), [("C3", "C4")], method="wilcoxon")
    assert set(t["series"]) == {"C3 - C4"} and len(t) == 4
    assert np.allclose(t["mean_a"] - t["mean_b"], (values[:, 0] - values[:, 1]).mean(axis=0))


def amplitude_table(values, subjects, components=("N200", "P300"), channel="Cz"):
    latency = {"N200": 0.20, "P300": 0.32}
    rows = [
        {"subject": s, "channel": channel, "component": c, "latency": latency[c], "amplitude": v}
        for s, row in zip(subjects, values, strict=True)
        for c, v in zip(components, row, strict=True)
    ]
    return pd.DataFrame(rows)


def test_compare_amplitudes_pairs_by_subject_name_not_by_row_order():
    rng = np.random.default_rng(6)
    subjects = [f"S{i}" for i in range(8)]
    va, vb = rng.standard_normal((8, 2)), rng.standard_normal((8, 2))
    va[:, 1] = vb[:, 1] + np.arange(1, 9)
    a, b = amplitude_table(va, subjects), amplitude_table(vb, subjects)
    ordered = ee.compare_amplitudes(a, b, method="wilcoxon")
    shuffled = ee.compare_amplitudes(a, b.sample(frac=1, random_state=0), method="wilcoxon")
    pd.testing.assert_frame_equal(ordered, shuffled)
    assert list(ordered.columns) == [
        "channel", "component", "latency", "n", "mean_a", "mean_b",
        "statistic", "p_raw", "p_adj", "effect", "p_floor", "significant",
    ]  # fmt: skip
    p300 = ordered[ordered["component"] == "P300"].iloc[0]
    assert p300["p_raw"] == pytest.approx(2 / 2**8) and p300["latency"] == pytest.approx(0.32)
    assert p300["n"] == 8 and p300["effect"] == 1.0


def test_compare_amplitudes_checks_that_the_tables_match_and_handles_signflip():
    rng = np.random.default_rng(8)
    subjects = [f"S{i}" for i in range(8)]
    va, vb = rng.standard_normal((8, 2)) + [0.0, 2.0], rng.standard_normal((8, 2))
    a, b = amplitude_table(va, subjects), amplitude_table(vb, subjects)
    with pytest.raises(ValueError, match="same subjects, channels and components"):
        ee.compare_amplitudes(a, b[b["subject"] != "S3"])
    t = ee.compare_amplitudes(a, b, method="signflip_maxt")
    _, p_adj = ee.signflip_maxt(va - vb)
    assert t["p_raw"].isna().all() and np.allclose(t["p_adj"], p_adj)
    a.loc[0, "amplitude"] = np.nan
    with pytest.raises(ValueError, match="complete data"):
        ee.compare_amplitudes(a, b, method="signflip_maxt")
    holed = ee.compare_amplitudes(a, b, method="wilcoxon")
    assert holed["n"].tolist() == [7, 8]


def test_simulated_contralateral_erd_is_visible_in_the_comparison(group_epochs):
    cfg = ee.ErdsConfig(band=(8, 13), window=0.5, overlap=0.0)
    left = ee.erds_group([e.select("left") for e in group_epochs], cfg)
    right = ee.erds_group([e.select("right") for e in group_epochs], cfg)
    t = ee.compare_conditions(left, right, ["C3", "C4"], method="wilcoxon")
    in_erd = t[(t["time"] > 1.0) & (t["time"] < 3.0)]
    diff = in_erd["mean_a"] - in_erd["mean_b"]  # left minus right, in percentage points
    assert diff[in_erd["series"] == "C4"].mean() < -15  # moving the left hand: C4 desynchronises
    assert diff[in_erd["series"] == "C3"].mean() > 15
    assert (t["n"] == len(group_epochs)).all()
