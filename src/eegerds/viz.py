"""Static matplotlib figures for the demo and the README.

Figures are built on :class:`matplotlib.figure.Figure` directly (no pyplot), so they work in
headless environments and leave no global state behind. Styling: solid hairline grid, thin marks,
text in ink colours (never in a series colour), categorical hues in a fixed order, a legend
whenever there are two or more series, and the series also named directly on the plot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .erds import GroupErds
from .erp import GroupErp, Peak
from .ica import EogRemoval

__all__ = ["plot_blink_removal", "plot_erds", "plot_erp", "plot_fwer", "save_figure"]

SURFACE = "#fcfcfb"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua: fixed order


def _figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")


def _style_axes(ax, grid_axis: str | None = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=8, length=3, width=0.75)
    ax.grid(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.75)
        ax.set_axisbelow(True)
    ax.xaxis.label.set_color(SECONDARY)
    ax.yaxis.label.set_color(SECONDARY)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=10.5, fontweight="bold", color=PRIMARY)


def save_figure(fig: Figure, path: str | Path, dpi: int = 160) -> Path:
    """Save as PNG (or any format matplotlib supports) on the figure surface colour."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    return path


def _extreme(times: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    i = int(np.argmax(np.abs(values)))
    return float(times[i]), float(values[i])


# ---------------------------------------------------------------------------------------
# ERD/ERS curves
# ---------------------------------------------------------------------------------------
def plot_erds(
    rows: Mapping[str, Mapping[str, GroupErds]],
    channels: Sequence[str],
    significance: Mapping[str, pd.DataFrame] | None = None,
    reference: tuple[float, float] | None = (-2.0, -1.0),
    smooth: int = 3,
    title: str = "ERD/ERS over the sensorimotor cortex",
    alpha_label: str = "p < 0.05, FDR across windows",
) -> Figure:
    """One row per band, one column per channel, one line per condition.

    ``rows`` maps a row title (``"mu, 8-13 Hz"``) to ``{condition label: GroupErds}``. Lines are
    the mean over subjects (moving average over ``smooth`` windows) with a ribbon of one
    standard error. ``significance`` maps the same row titles to tables from
    :func:`eegerds.stats.compare_conditions`; windows with ``significant`` set are marked along
    the bottom of the panel.
    """
    n_rows, n_cols = len(rows), len(channels)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("nothing to plot")
    fig = _figure(3.6 * n_cols + 0.9, 2.6 * n_rows + 1.3)
    axes = fig.subplots(n_rows, n_cols, sharex=True, sharey="row", squeeze=False)
    n_series = max(len(v) for v in rows.values())
    if n_series > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} conditions can be drawn")
    for r, (row_title, groups) in enumerate(rows.items()):
        lows, highs = [], []
        for c, channel in enumerate(channels):
            ax = axes[r, c]
            _style_axes(ax)
            if reference is not None:
                ax.axvspan(*reference, color=BASELINE, alpha=0.35, lw=0)
            ax.axhline(0, color=BASELINE, lw=0.75)
            ax.axvline(0, color=SECONDARY, lw=0.75, ls=(0, (3, 3)))
            for i, (label, group) in enumerate(groups.items()):
                j = list(group.ch_names).index(channel)
                times, mean = group.smoothed_mean(smooth)
                mean = mean[j]
                sem = np.convolve(group.sem()[j], np.ones(smooth) / smooth, "valid")
                ax.fill_between(times, mean - sem, mean + sem, color=SERIES[i], alpha=0.16, lw=0)
                ax.plot(times, mean, color=SERIES[i], lw=1.6, solid_capstyle="round")
                lows.append(float((mean - sem).min()))
                highs.append(float((mean + sem).max()))
                tx, ty = _extreme(times, mean)
                up = ty >= 0
                ax.annotate(
                    label,
                    (tx, ty),
                    xytext=(0, 6 if up else -6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if up else "top",
                    fontsize=8,
                    color=SECONDARY,
                )
            _title(ax, channel if r == 0 else "")
            if c == 0:
                ax.set_ylabel(f"{row_title}\nERD/ERS (%)")
        lo, hi = min(lows), max(highs)
        pad = 0.18 * (hi - lo)
        for c, channel in enumerate(channels):
            ax = axes[r, c]
            ax.set_ylim(lo - pad * 1.6, hi + pad)
            table = None if significance is None else significance.get(row_title)
            if table is not None:
                hits = table[(table["series"] == channel) & table["significant"]]
                for _, w in hits.iterrows():
                    ax.plot(
                        [w["t_start"], w["t_stop"]],
                        [lo - pad * 1.2] * 2,
                        color=SECONDARY,
                        lw=3,
                        solid_capstyle="butt",
                    )
    for c in range(n_cols):
        axes[-1, c].set_xlabel("Time from cue (s)")
    handles: list[Artist] = [Line2D([], [], color=SERIES[i], lw=1.6) for i in range(n_series)]
    labels = list(next(iter(rows.values())).keys())
    if reference is not None:
        handles.append(Patch(facecolor=BASELINE, alpha=0.35, lw=0))
        labels.append("reference interval")
    if significance:
        handles.append(Line2D([], [], color=SECONDARY, lw=3))
        labels.append(alpha_label)
    fig.legend(handles, labels, loc="outside lower center", ncols=len(labels), frameon=False,
               fontsize=8.5, labelcolor=SECONDARY)  # fmt: skip
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=PRIMARY)
    return fig


# ---------------------------------------------------------------------------------------
# ERP
# ---------------------------------------------------------------------------------------
def plot_erp(
    groups: Mapping[str, GroupErp],
    channel: str,
    peaks: Sequence[Peak] = (),
    xlim: tuple[float, float] = (-0.2, 0.8),
    title: str | None = None,
) -> Figure:
    """Grand-average ERP of ``channel`` per condition, with the group-level peaks marked."""
    if len(groups) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} conditions can be drawn")
    fig = _figure(7.6, 4.2)
    ax = fig.subplots()
    _style_axes(ax)
    ax.axhline(0, color=BASELINE, lw=0.75)
    ax.axvline(0, color=SECONDARY, lw=0.75, ls=(0, (3, 3)))
    ymin, ymax = 0.0, 0.0
    for i, (label, group) in enumerate(groups.items()):
        j = list(group.ch_names).index(channel)
        data = group.values[:, j, :] * 1e6
        mean = data.mean(axis=0)
        sem = data.std(axis=0, ddof=1) / np.sqrt(len(data)) if len(data) > 1 else 0 * mean
        sel = (group.times >= xlim[0]) & (group.times <= xlim[1])
        ax.fill_between(group.times[sel], (mean - sem)[sel], (mean + sem)[sel],
                        color=SERIES[i], alpha=0.16, lw=0)  # fmt: skip
        ax.plot(group.times[sel], mean[sel], color=SERIES[i], lw=1.6)
        ymin, ymax = (
            min(ymin, float((mean - sem)[sel].min())),
            max(ymax, float((mean + sem)[sel].max())),
        )
        tx, ty = _extreme(group.times[sel], mean[sel])
        ax.annotate(
            label,
            (tx, ty),
            xytext=(0, 7 if ty >= 0 else -7),
            textcoords="offset points",
            ha="center",
            va="bottom" if ty >= 0 else "top",
            fontsize=8,
            color=SECONDARY,
        )
    span = ymax - ymin
    ax.set_ylim(ymin - 0.2 * span, ymax + 0.25 * span)
    for peak in peaks:
        if not peak.found:
            continue
        ax.axvline(peak.latency, color=MUTED, lw=0.75, ls=(0, (1, 2)))
        ax.annotate(
            f"{peak.name}\n{peak.latency * 1000:.0f} ms",
            (peak.latency, ymax + 0.25 * span),
            xytext=(0, -3),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color=SECONDARY,
        )
    ax.set_xlim(*xlim)
    ax.set_xlabel("Time from cue (s)")
    ax.set_ylabel("Amplitude (µV, positive up)")
    _title(ax, title or f"Event-related potential at {channel}")
    handles = [Line2D([], [], color=SERIES[i], lw=1.6) for i in range(len(groups))]
    fig.legend(handles, list(groups), loc="outside lower center", ncols=len(groups), frameon=False,
               fontsize=8.5, labelcolor=SECONDARY)  # fmt: skip
    return fig


# ---------------------------------------------------------------------------------------
# Blink removal
# ---------------------------------------------------------------------------------------
def plot_blink_removal(
    before: np.ndarray,
    after: np.ndarray,
    fs: float,
    removal: EogRemoval,
    channel: str = "Fp1",
    seconds: tuple[float, float] = (20.0, 50.0),
) -> Figure:
    """A frontal channel before and after SOBI cleaning, and each component's EOG correlation."""
    fig = _figure(10.2, 4.0)
    ax, bx = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2.2, 1.0]})
    _style_axes(ax)
    i0, i1 = int(seconds[0] * fs), int(seconds[1] * fs)
    t = np.arange(i0, i1) / fs
    ax.plot(t, before[i0:i1] * 1e6, color=SERIES[0], lw=1.0)
    ax.plot(t, after[i0:i1] * 1e6, color=SERIES[1], lw=1.0)
    ax.axhline(0, color=BASELINE, lw=0.75)
    peak_t = t[int(np.argmax(before[i0:i1]))]
    ax.annotate("before", (peak_t, before[i0:i1].max() * 1e6), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=8, color=SECONDARY)  # fmt: skip
    ax.annotate("after", (peak_t, after[i0:i1].max() * 1e6), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=8, color=SECONDARY)  # fmt: skip
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    _title(ax, f"{channel}: blinks removed")

    _style_axes(bx)
    r = np.abs(removal.correlations)
    order = np.argsort(-r)[:10]
    colors = [SERIES[1] if int(k) in removal.removed else SERIES[0] for k in order]
    bx.bar(range(len(order)), r[order], color=colors, width=0.7)
    bx.set_xticks(range(len(order)), [str(int(k)) for k in order])
    bx.set_ylim(0, 1.05)
    bx.set_xlabel("Component (sorted by |r|)")
    bx.set_ylabel("|r| with the EOG trace")
    top = int(order[0])
    if top in removal.removed:
        bx.annotate("removed", (0, r[top]), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=SECONDARY)  # fmt: skip
    _title(bx, "Which component is the blink?")
    fig.legend(
        [Line2D([], [], color=SERIES[0], lw=1.6), Line2D([], [], color=SERIES[1], lw=1.6)],
        ["before SOBI", "after SOBI"],
        loc="outside lower center", ncols=2, frameon=False, fontsize=8.5, labelcolor=SECONDARY,
    )  # fmt: skip
    return fig


# ---------------------------------------------------------------------------------------
# Multiple-comparison study
# ---------------------------------------------------------------------------------------
def plot_fwer(
    rates: Mapping[str, float],
    n_reps: int,
    n_tests: int,
    intervals: Mapping[str, tuple[float, float]] | None = None,
) -> Figure:
    """Share of null datasets with at least one 'significant' window, per testing rule.

    ``intervals`` (optional) maps each rule to a ``(low, high)`` confidence interval for its rate,
    drawn as a thin vertical line: with a few hundred datasets the interval is wide, and a bar
    without it suggests more precision than there is.
    """
    fig = _figure(7.6, 4.0)
    ax = fig.subplots()
    _style_axes(ax)
    labels = list(rates)
    values = [rates[k] * 100 for k in labels]
    tops = [intervals[k][1] * 100 if intervals else v for k, v in zip(labels, values, strict=True)]
    ax.bar(range(len(labels)), values, color=SERIES[0], width=0.4)
    for i, (k, v, top) in enumerate(zip(labels, values, tops, strict=True)):
        if intervals:
            low, high = (100 * x for x in intervals[k])
            ax.plot([i, i], [low, high], color=SECONDARY, lw=1.0, solid_capstyle="butt", zorder=3)
        ax.text(
            i,
            max(v, top) + 0.8,
            f"{v:.1f}".removesuffix(".0") + "%",  # 6.5% stays 6.5%, 40.0% reads 40%
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=SECONDARY,
        )
    ax.axhline(5, color=SECONDARY, lw=0.75, ls=(0, (3, 3)), zorder=2)
    ax.text(
        (len(labels) - 1) / 2,
        5.8,
        "5% target",
        ha="center",
        va="bottom",
        fontsize=8,
        color=SECONDARY,
    )
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylim(0, max(tops) * 1.18 + 2)
    ax.set_ylabel("Datasets with a false positive")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    _title(ax, f"False positives when nothing is there ({n_tests} windows, {n_reps} datasets)")
    return fig
