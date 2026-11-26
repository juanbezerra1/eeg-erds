import matplotlib.text
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

import eegerds as ee
from eegerds import viz
from eegerds.viz import SERIES, plot_blink_removal, plot_erds, plot_erp, plot_fwer, save_figure


@pytest.fixture(scope="module")
def rows(group_epochs):
    cfg = ee.ErdsConfig(band=(8.0, 13.0))
    return {
        "mu, 8-13 Hz": {
            "left trials": ee.erds_group([e.select("left") for e in group_epochs], cfg),
            "right trials": ee.erds_group([e.select("right") for e in group_epochs], cfg),
        }
    }


@pytest.fixture(scope="module")
def groups(group_epochs):
    return {
        "high": ee.group_erp(group_epochs, ["high"], baseline=(-0.2, 0.0)),
        "low": ee.group_erp(group_epochs, ["low"], baseline=(-0.2, 0.0)),
    }


def significance_table(*windows):
    """A ``compare_conditions``-like table with the given (channel, t_start, t_stop) significant."""
    return pd.DataFrame(
        [{"series": c, "t_start": a, "t_stop": b, "significant": True} for c, a, b in windows]
        + [{"series": "C3", "t_start": 9.0, "t_stop": 9.5, "significant": False}]
    )


def series_lines(ax):
    return [ln for ln in ax.lines if to_hex(ln.get_color()) in SERIES]


def text_colours(fig):
    return {to_hex(t.get_color()) for t in fig.findobj(matplotlib.text.Text) if t.get_text()}


def assert_text_never_wears_a_series_colour(fig):
    assert not text_colours(fig) & set(SERIES)


def test_erds_figure_layout_legend_and_ink(rows):
    fig = plot_erds(rows, ["C3", "C4"])
    assert isinstance(fig, Figure) and len(fig.axes) == 2
    assert all(len(series_lines(ax)) == 2 for ax in fig.axes)  # one line per condition
    legend = fig.legends[0]
    assert [t.get_text() for t in legend.get_texts()] == [
        "left trials",
        "right trials",
        "reference interval",
    ]
    assert_text_never_wears_a_series_colour(fig)
    assert [ax.get_title(loc="left") for ax in fig.axes] == ["C3", "C4"]


def test_erds_marks_only_the_significant_windows_of_each_channel(rows):
    table = significance_table(("C3", 1.0, 1.5), ("C3", 1.5, 2.0), ("C4", 2.0, 2.5))
    fig = plot_erds(rows, ["C3", "C4"], {"mu, 8-13 Hz": table})
    marks = [[ln for ln in ax.lines if ln.get_linewidth() == 3] for ax in fig.axes]
    assert [len(m) for m in marks] == [2, 1]
    assert [t.get_text() for t in fig.legends[0].get_texts()][-1].startswith("p < 0.05")
    assert marks[0][0].get_xdata().tolist() == [1.0, 1.5]
    assert_text_never_wears_a_series_colour(fig)


def test_erds_without_reference_band_and_with_a_custom_title(rows):
    fig = plot_erds(rows, ["C4"], reference=None, title="My title")
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["left trials", "right trials"]
    assert "My title" in [t.get_text() for t in fig.texts]  # get_suptitle() needs matplotlib 3.8


def test_erds_refuses_what_it_cannot_draw_clearly(rows):
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_erds({}, ["C3"])
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_erds(rows, [])
    four = {"x": {str(i): next(iter(rows.values()))["left trials"] for i in range(4)}}
    with pytest.raises(ValueError, match="at most 3"):
        plot_erds(four, ["C3"])


def test_erp_figure_marks_found_peaks_only(groups):
    peaks = [ee.Peak("N200", 0.2, -4e-6, True), ee.Peak("P300", 0.32, 6e-6, True),
             ee.Peak("N400", float("nan"), float("nan"), False)]  # fmt: skip
    fig = plot_erp(groups, "Cz", peaks, xlim=(-0.1, 0.7), title="Cz ERP")
    (ax,) = fig.axes
    assert (
        len(series_lines(ax)) == 2
        and ax.get_xlim() == (-0.1, 0.7)
        and ax.get_title(loc="left") == "Cz ERP"
    )
    marks = [ln for ln in ax.lines if to_hex(ln.get_color()) == viz.MUTED]  # one line per peak
    assert sorted(round(ln.get_xdata()[0], 2) for ln in marks) == [0.2, 0.32]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["high", "low"]
    assert_text_never_wears_a_series_colour(fig)
    assert plot_erp(groups, "Pz").axes[0].get_title(loc="left") == "Event-related potential at Pz"


def test_erp_handles_a_single_subject_and_too_many_conditions(groups, group_epochs):
    solo = {"all": ee.group_erp(group_epochs[:1], baseline=(-0.2, 0.0))}
    assert len(plot_erp(solo, "Cz").axes) == 1  # no standard error with one subject: still draws
    with pytest.raises(ValueError, match="at most 3"):
        plot_erp({str(i): groups["high"] for i in range(4)}, "Cz")


def test_blink_figure_shows_the_removed_component(group_results):
    recs, results = group_results
    res = results[0]
    before = ee.process_recording(recs[0], ee.PipelineConfig(remove_blinks=False, rejection=None))
    fig = plot_blink_removal(before.cleaned.data[0], res.cleaned.data[0], recs[0].fs, res.eog)
    line_ax, bar_ax = fig.axes
    assert len(series_lines(line_ax)) == 2 and len(bar_ax.patches) == 10
    removed = int(res.eog.removed[0])
    top = np.argsort(-np.abs(res.eog.correlations))[0]
    assert removed == int(top)
    assert to_hex(bar_ax.patches[0].get_facecolor()) == SERIES[1]  # the removed one is highlighted
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["before SOBI", "after SOBI"]
    assert_text_never_wears_a_series_colour(fig)


def test_fwer_figure_is_a_single_series_without_a_legend():
    fig = plot_fwer({"uncorrected": 0.34, "FDR": 0.04, "max-T": 0.05}, n_reps=300, n_tests=10)
    (ax,) = fig.axes
    assert len(ax.patches) == 3 and not fig.legends
    assert ax.get_ylim()[1] > 34
    assert "10 windows, 300 datasets" in ax.get_title(loc="left")
    assert {"34%", "4%", "5%"} <= {t.get_text() for t in ax.texts}
    assert_text_never_wears_a_series_colour(fig)


def test_fwer_labels_keep_a_half_point_instead_of_rounding_it_away():
    fig = plot_fwer({"a": 0.065, "b": 0.4, "c": 0.0}, n_reps=200, n_tests=16)
    assert {"6.5%", "40%", "0%"} <= {t.get_text() for t in fig.axes[0].texts}


def test_save_figure_writes_a_png_of_the_requested_size(tmp_path):
    fig = plot_fwer({"a": 0.1, "b": 0.05}, 10, 3)
    path = save_figure(fig, tmp_path / "deep" / "dir" / "fwer.png", dpi=100)
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert (int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")) == (760, 400)


def test_palette_constants_are_the_documented_ones():
    assert SERIES == ("#2a78d6", "#eb6834", "#1baf7a") and viz.SURFACE == "#fcfcfb"
