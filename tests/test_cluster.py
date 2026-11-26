"""The cluster-based permutation test (Maris and Oostenveld, 2007) across time windows."""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import eegerds as ee
from eegerds import stats as st

TIMES = np.arange(8) * 0.5 + 0.25


def reference_clusters(row, threshold):
    """``(first, last, mass)`` of the runs of equal-signed ``|t| > threshold``, one plain loop."""
    sign = [1 if v > threshold else -1 if v < -threshold else 0 for v in row]
    found, i = [], 0
    for key, group in itertools.groupby(sign):
        length = len(list(group))
        if key:
            found.append((i, i + length - 1, float(sum(abs(v) for v in row[i : i + length]))))
        i += length
    return found


def brute_force(d, threshold):
    """Every one of the ``2**n`` sign patterns, one scipy t-test at a time."""
    n, n_series, _ = d.shape

    def largest(x):
        t = stats.ttest_1samp(x, 0.0).statistic
        return max(
            (m for s in range(n_series) for _, _, m in reference_clusters(t[s], threshold)),
            default=0.0,
        )

    t_obs = stats.ttest_1samp(d, 0.0).statistic
    null = np.array(
        [largest(d * np.array(s)[:, None, None]) for s in itertools.product([-1.0, 1.0], repeat=n)]
    )
    p = np.ones(t_obs.shape)
    for s in range(n_series):
        for first, last, mass in reference_clusters(t_obs[s], threshold):
            p[s, first : last + 1] = (null >= mass * (1 - 1e-12)).mean()
    return t_obs, p


# ----------------------------------------------------------------------- finding the clusters
T_ROW = np.array([0.0, 3.0, 3.5, 0.0, -4.0, -3.0, 2.1, 0.0])


def test_clusters_are_runs_of_neighbouring_windows_beyond_the_threshold_in_one_direction():
    assert st.find_clusters(T_ROW, 2.0) == [(1, 2, 6.5), (4, 5, 7.0), (6, 6, 2.1)]
    assert st.find_clusters(T_ROW, 3.2) == [(2, 2, 3.5), (4, 4, 4.0)]  # the 3.0s no longer qualify
    assert st.find_clusters(T_ROW, 10.0) == []


def test_a_positive_run_next_to_a_negative_one_is_two_clusters_not_one():
    assert st.find_clusters(np.array([3.0, 3.0, -3.0, -3.0]), 2.0) == [(0, 1, 6.0), (2, 3, 6.0)]


def test_the_threshold_is_strict_and_a_cluster_may_touch_either_end_of_the_series():
    assert st.find_clusters(np.array([2.0, 2.0]), 2.0) == []
    assert st.find_clusters(np.array([2.5, 0.0, 2.5, 2.5]), 2.0) == [(0, 0, 2.5), (2, 3, 5.0)]
    assert st.find_clusters(np.array([]), 2.0) == []


def test_the_vectorised_largest_cluster_agrees_with_the_plain_loop():
    rng = np.random.default_rng(0)
    t = rng.standard_normal((50, 3, 20)) * 2
    largest = st._largest_cluster(t, 1.5)
    assert largest.shape == (50, 3)
    for i, s in itertools.product(range(50), range(3)):
        masses = [m for _, _, m in reference_clusters(list(t[i, s]), 1.5)]
        assert largest[i, s] == pytest.approx(max(masses, default=0.0))


def test_the_largest_cluster_of_a_series_with_nothing_beyond_the_threshold_is_zero():
    assert st._largest_cluster(np.array([[0.1, -0.2, 0.3]]), 2.0).tolist() == [0.0]


# ----------------------------------------------------------------------- the permutation test
def test_the_exact_test_matches_a_brute_force_enumeration_of_every_sign_pattern():
    rng = np.random.default_rng(1)
    d = rng.standard_normal((6, 2, 7)) + np.array([0.0, 0.0, 1.4, 1.6, 1.5, 0.0, 0.0])
    threshold = stats.t.ppf(0.975, 5)
    t, p, cluster, mass = ee.signflip_cluster(d, threshold)
    t_ref, p_ref = brute_force(d, threshold)
    assert np.allclose(t, t_ref) and np.allclose(p, p_ref)
    assert (cluster > 0).any(), "the data were meant to hold a cluster"
    assert np.all(mass[cluster == 0] == 0) and np.all(p[cluster == 0] == 1)
    for s in range(2):
        for first, last, size in reference_clusters(t[s], threshold):
            assert len(set(cluster[s, first : last + 1])) == 1
            assert np.allclose(mass[s, first : last + 1], size)


def test_clusters_are_numbered_in_series_and_time_order_and_share_one_p():
    wiggle = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]) * 0.1  # mean exactly 0: t = 0 there
    d = np.repeat(wiggle[:, None, None], 12, axis=1).reshape(6, 2, 6).copy()
    d[:, 0, 1:3] += np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0])[:, None]  # positive, 2 windows
    d[:, 0, 4] -= 1.0  # negative, 1 window, separated from the first by a quiet window
    d[:, 1, 0:2] += 1.0  # second series, positive
    t, p, cluster, _ = ee.signflip_cluster(d, 2.0)
    assert cluster.tolist() == [[0, 1, 1, 0, 2, 0], [3, 3, 0, 0, 0, 0]]
    assert p[0, 1] == p[0, 2] and p[1, 0] == p[1, 1]
    assert np.all(t[cluster == 0] == pytest.approx(0.0, abs=1e-12))


@pytest.mark.parametrize("n", [3, 6, 9])
def test_the_smallest_attainable_p_is_two_over_two_to_the_n(n):
    d = np.ones((n, 1, 5)) + np.arange(n)[:, None, None] * 0.01  # everyone is higher everywhere
    _, p, cluster, _ = ee.signflip_cluster(d, 2.0)
    assert np.all(cluster == 1) and np.allclose(p, 2 / 2**n)


def test_a_window_whose_differences_are_all_identical_does_not_break_the_sums():
    d = np.zeros((6, 1, 4))
    d[:, 0, 1:3] = 2.0  # every subject: exactly the same difference, so zero spread
    t, p, cluster, mass = ee.signflip_cluster(d, 2.0)
    assert np.isfinite(t).all() and np.isfinite(mass).all() and np.isfinite(p).all()
    assert cluster.tolist() == [[0, 1, 1, 0]]
    assert np.allclose(p[0, 1:3], 2 / 2**6)  # never 0: a permutation test cannot give that


def test_a_chunked_run_gives_the_same_answer_as_one_chunk(monkeypatch):
    d = np.random.default_rng(3).standard_normal((8, 3, 10)) + 0.4
    whole = ee.signflip_cluster(d, 1.5)
    monkeypatch.setattr(st, "_CHUNK_CELLS", 100)  # a few relabellings at a time
    chunked = ee.signflip_cluster(d, 1.5)
    for a, b in zip(whole, chunked, strict=True):
        assert np.array_equal(a, b)


def test_beyond_the_exhaustive_limit_the_random_patterns_are_reproducible_and_never_give_zero():
    d = np.random.default_rng(4).standard_normal((14, 1, 6)) + 1.0
    a = ee.signflip_cluster(d, 2.0, n_perm=500, seed=7)
    b = ee.signflip_cluster(d, 2.0, n_perm=500, seed=7)
    assert np.array_equal(a[1], b[1])
    assert a[1].min() >= 1 / 501
    exact = ee.signflip_cluster(d, 2.0, max_exhaustive=14)  # all 16384 patterns
    assert np.abs(exact[1] - a[1]).max() < 0.1  # the random estimate is close to the exact value


def test_input_checks():
    with pytest.raises(ValueError, match="n_subjects, n_series, n_windows"):
        ee.signflip_cluster(np.zeros((6, 4)), 2.0)
    with pytest.raises(ValueError, match="at least 2 subjects"):
        ee.signflip_cluster(np.zeros((1, 1, 4)), 2.0)
    bad = np.zeros((6, 1, 4))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ee.signflip_cluster(bad, 2.0)


# ----------------------------------------------------------------------- calibration and power
def smooth_noise(rng, n_subjects, n_windows, ar=0.7):
    """Noise whose neighbouring windows are correlated, as ERD/ERS windows are."""
    white = rng.standard_normal((n_subjects, n_windows))
    out = np.empty_like(white)
    out[:, 0] = white[:, 0]
    for j in range(1, n_windows):
        out[:, j] = ar * out[:, j - 1] + np.sqrt(1 - ar**2) * white[:, j]
    return out


def detection_rates(effect, seed, n_rep, methods=("cluster", "signflip_maxt")):
    """Share of experiments (8 subjects, 16 windows, correlated noise plus ``effect`` in every
    subject) in which each method calls at least one window significant."""
    rng = np.random.default_rng(seed)
    hits = dict.fromkeys(methods, 0)
    for _ in range(n_rep):
        d = smooth_noise(rng, 8, 16) + effect
        zeros = np.zeros_like(d)
        for method in methods:
            table = ee.compare_series(d[:, None, :], zeros[:, None, :], ["x"], np.arange(16), 0.5,
                                      method=method)  # fmt: skip
            hits[method] += bool(table["significant"].any())
    return {method: count / n_rep for method, count in hits.items()}


def test_with_no_effect_the_family_wise_error_stays_near_the_nominal_five_percent():
    rate = detection_rates(0.0, seed=11, n_rep=400, methods=("cluster", "wilcoxon"))
    assert rate["cluster"] < 0.09  # nominal 0.05; a binomial standard error is 0.011
    assert rate["wilcoxon"] > 0.2  # the same data, tested window by window, are not safe


def test_a_smooth_effect_is_found_more_often_by_the_cluster_test_than_by_max_t():
    """The reason to offer it: a weak effect spread over neighbouring windows adds up."""
    effect = np.zeros(16)
    effect[5:11] = 1.3 * np.hanning(8)[1:-1]  # six windows, peaking at 1.2 standard deviations
    rate = detection_rates(effect, seed=12, n_rep=150)
    assert rate["cluster"] > rate["signflip_maxt"] + 0.05


def test_a_one_window_effect_is_found_more_often_by_max_t_than_by_the_cluster_test():
    """The price: a sharp peak adds up to nothing, and the null holds wider clusters."""
    effect = np.zeros(16)
    effect[8] = 2.0
    rate = detection_rates(effect, seed=13, n_rep=150)
    assert rate["signflip_maxt"] > rate["cluster"] + 0.2


# ----------------------------------------------------------------------- the comparison tables
BASE_COLUMNS = [
    "series", "time", "t_start", "t_stop", "n", "mean_a", "mean_b",
    "statistic", "p_raw", "p_adj", "effect", "p_floor", "significant",
]  # fmt: skip


def effect_cube(n_sub=8):
    """Series "C3" differs in windows 2-4. Elsewhere the conditions differ only by noise whose
    mean is exactly 0 in every window (so t = 0 and no cluster can form there by chance)."""
    noise = np.random.default_rng(0).standard_normal((n_sub, 2, 8)) * 0.1
    noise -= noise.mean(axis=0)
    a, b = noise.copy(), np.zeros_like(noise)
    a[:, 0, 2:5] += 2.0 + np.linspace(-0.3, 0.3, n_sub)[:, None]
    return a, b


def make_group(values, channels=("C3", "C4")):
    n_sub, _, n_win = values.shape
    return ee.GroupErds(
        times=np.arange(n_win) * 0.25 + 0.25,
        values=values,
        subjects=tuple(f"S{i}" for i in range(n_sub)),
        ch_names=channels,
        config=ee.ErdsConfig(window=0.5),
    )


def test_the_cluster_table_has_the_usual_columns_plus_the_cluster_and_its_mass():
    a, b = effect_cube()
    table = ee.compare_series(a, b, ["C3", "C4"], TIMES, 0.5, method="cluster")
    assert list(table.columns) == [*BASE_COLUMNS, "cluster", "cluster_mass"]
    assert len(table) == 16 and table["p_raw"].isna().all()
    found = table[table["cluster"] > 0]
    assert set(found["series"]) == {"C3"} and list(found["time"]) == list(TIMES[2:5])
    assert found["significant"].all() and (found["p_adj"] == found["p_floor"]).all()
    assert found["cluster_mass"].nunique() == 1 and found["cluster_mass"].iloc[0] > 3 * 2.3
    rest = table[table["cluster"] == 0]
    assert (rest["p_adj"] == 1).all() and not rest["significant"].any()
    assert (rest["cluster_mass"] == 0).all()
    t_ref = stats.ttest_rel(a.reshape(8, -1), b.reshape(8, -1)).statistic
    assert np.allclose(table["statistic"], t_ref)


def test_the_other_methods_keep_the_columns_they_always_had():
    a, b = effect_cube()
    for method in ("wilcoxon_fdr", "wilcoxon", "signflip_maxt"):
        table = ee.compare_series(a, b, ["C3", "C4"], TIMES, 0.5, method=method)
        assert list(table.columns) == BASE_COLUMNS


def test_the_cluster_threshold_decides_which_windows_can_join_a_cluster():
    a, b = effect_cube()
    a[:, 0, 5] += 0.12  # next to the cluster, with a t that is large but far from huge
    t5 = abs(stats.ttest_rel(a[:, 0, 5], b[:, 0, 5]).statistic)
    assert stats.t.ppf(0.975, 7) < t5 < stats.t.ppf(0.9995, 7), "the test data are off"

    def joined(cluster_p):
        table = ee.compare_series(a, b, ["C3", "C4"], TIMES, 0.5, method="cluster",
                                  cluster_p=cluster_p)  # fmt: skip
        return list(table.loc[table["cluster"] > 0, "time"])

    assert joined(0.05) == list(TIMES[2:6])  # t5 beyond the two-sided 5 % threshold
    assert joined(0.001) == list(TIMES[2:5])  # but not beyond the 0.1 % one
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="cluster_p"):
            ee.compare_series(a, b, ["C3", "C4"], TIMES, 0.5, method="cluster", cluster_p=bad)


def test_alpha_decides_which_clusters_are_called_significant():
    a, b = effect_cube(n_sub=5)  # five subjects: the floor is 2/2**5 = 0.0625
    table = ee.compare_series(a, b, ["C3", "C4"], TIMES, 0.5, method="cluster", alpha=0.05)
    assert not table["significant"].any() and (table["p_floor"] == 2 / 2**5).all()
    table = ee.compare_series(a, b, ["C3", "C4"], TIMES, 0.5, method="cluster", alpha=0.1)
    assert table["significant"].sum() == 3


def test_missing_values_and_one_subject_are_refused():
    a, b = effect_cube()
    holed = a.copy()
    holed[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="cluster needs complete data"):
        ee.compare_series(holed, b, ["C3", "C4"], TIMES, 0.5, method="cluster")
    with pytest.raises(ValueError, match="at least 2 subjects"):
        ee.compare_series(a[:1], b[:1], ["C3", "C4"], TIMES, 0.5, method="cluster")


def test_amplitudes_have_no_neighbouring_windows_so_the_cluster_method_is_refused():
    amp = pd.DataFrame({"subject": ["S0"], "channel": ["Cz"], "component": ["P300"],
                        "latency": [0.3], "amplitude": [1.0]})  # fmt: skip
    with pytest.raises(ValueError, match="neighbouring time windows"):
        ee.compare_amplitudes(amp, amp, method="cluster")


def test_the_group_comparisons_accept_the_cluster_method():
    a, b = effect_cube()
    table = ee.compare_conditions(make_group(a), make_group(b), ["C4", "C3"], method="cluster")
    assert list(table["series"].unique()) == ["C4", "C3"]  # the requested order
    assert {"cluster", "cluster_mass"} <= set(table.columns)
    assert list(table.loc[table["cluster"] > 0, "series"].unique()) == ["C3"]
    pairs = ee.compare_channels(make_group(a), [("C3", "C4")], method="cluster", cluster_p=0.01)
    assert set(pairs["series"]) == {"C3 - C4"} and "cluster" in pairs.columns
    assert "cluster" in st.METHODS
