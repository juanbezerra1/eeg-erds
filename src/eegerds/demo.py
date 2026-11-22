"""End-to-end demonstration on synthetic data, with the ground truth printed next to the result.

``run_demo`` simulates a group of subjects, runs the whole pipeline (filters, SOBI blink removal,
epoching, epoch rejection), estimates ERD/ERS and ERPs, tests them across subjects, exports a
trial-level feature table, draws three figures and writes ``report.md``. Everything is synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np

from .artifacts import RejectionConfig, reject_epochs
from .data import Epochs
from .erds import ErdsConfig, erds_group, windowed_mean
from .erp import find_group_peaks, group_erp, measure_amplitudes
from .features import features_table
from .pipeline import PipelineConfig, clean_recording, process_group
from .stats import compare_amplitudes, compare_conditions
from .synth import SimConfig, expected_erds, simulate_group
from .viz import plot_blink_removal, plot_erds, plot_erp, save_figure

__all__ = ["DemoResult", "run_demo"]

BANDS = {"mu, 8-13 Hz": ("mu", (8.0, 13.0)), "beta, 13-30 Hz": ("beta", (13.0, 30.0))}
ERP_CHANNELS = ("Fz", "Cz", "Pz")


@dataclass(frozen=True)
class DemoResult:
    out_dir: Path
    files: dict[str, Path]
    numbers: dict[str, float]


def _window_truth(
    cfg: SimConfig,
    epochs: Epochs,
    side: str,
    channel: str,
    band: str,
    depth: float,
    ecfg: ErdsConfig,
) -> np.ndarray:
    """Expected ERD/ERS per analysis window (same windowing as the estimate)."""
    curve = expected_erds(cfg, side, channel, band, epochs.times, depth)
    return windowed_mean(curve, epochs.fs, ecfg.window, ecfg.overlap)[1]


def run_demo(
    out_dir: str | Path = "eegerds_demo",
    n_subjects: int = 10,
    n_trials: int = 10,
    seed: int = 0,
    quick: bool = False,
) -> DemoResult:
    """Run the demonstration and write its outputs into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if quick:
        n_subjects, n_trials = 6, 5
    cfg = SimConfig(n_trials=n_trials)
    recs = simulate_group(n_subjects, cfg, seed=seed)
    four_rules = RejectionConfig(max_window_sd_ratio=3.0)
    results = process_group(recs, PipelineConfig(rejection=four_rules))
    eps = [r.epochs for r in results]
    nums: dict[str, float] = {"n_subjects": n_subjects, "n_trials_per_condition": n_trials}

    # --- artefact handling against the known truth ----------------------------------
    caught = {"three": 0, "four": 0}
    n_art = wrongly = n_clean = 0
    for res, rec in zip(results, recs, strict=True):
        truth = cast("dict[int, str]", rec.truth["artifact_trials"])
        flags3 = ~reject_epochs(res.epochs_all, RejectionConfig()).keep
        flags4 = ~reject_epochs(res.epochs_all, four_rules).keep
        for k in range(res.epochs_all.n_epochs):
            if k in truth:
                n_art += 1
                caught["three"] += int(flags3[k])
                caught["four"] += int(flags4[k])
            else:
                n_clean += 1
                wrongly += int(flags4[k])
    nums.update(
        artifact_epochs=n_art,
        caught_three_rules=caught["three"],
        caught_four_rules=caught["four"],
        clean_epochs=n_clean,
        clean_rejected_four_rules=wrongly,
        epochs_kept=sum(e.n_epochs for e in eps),
        epochs_total=sum(r.epochs_all.n_epochs for r in results),
    )

    # --- blink removal against the known blink signal -------------------------------
    unfixed = [clean_recording(rec, PipelineConfig(remove_blinks=False))[0] for rec in recs]
    r_before, r_after = [], []
    for res, rec, raw in zip(results, recs, unfixed, strict=True):
        j = rec.ch_names.index("Fp1")
        blink = cast("np.ndarray", rec.truth["blink_signal"])
        r_before.append(np.corrcoef(raw.data[j], blink)[0, 1])
        r_after.append(np.corrcoef(res.cleaned.data[j], blink)[0, 1])
    nums.update(blink_r_before=float(np.mean(r_before)), blink_r_after=float(np.mean(r_after)))
    assert results[0].eog is not None
    fig = plot_blink_removal(
        unfixed[0].data[0], results[0].cleaned.data[0], recs[0].fs, results[0].eog
    )
    files = {"blinks": save_figure(fig, out / "blinks.png")}

    # --- ERD/ERS ----------------------------------------------------------------------
    rows: dict[str, dict[str, object]] = {}
    significance = {}
    depth = float(np.mean([rec.truth["depth_scale"] for rec in recs]))
    for row_title, (band_name, band) in BANDS.items():
        show = ErdsConfig(band=band)
        test = replace(show, overlap=0.0)
        left = erds_group([e.select("left") for e in eps], show)
        right = erds_group([e.select("right") for e in eps], show)
        rows[row_title] = {"left trials": left, "right trials": right}
        a = erds_group([e.select("left") for e in eps], test)
        b = erds_group([e.select("right") for e in eps], test)
        table = compare_conditions(a, b, ["C3", "C4"], method="wilcoxon_fdr")
        significance[row_title] = table
        nums[f"n_significant_windows_{band_name}"] = int(table["significant"].sum())
        nums[f"n_windows_tested_{band_name}"] = len(table)
        window = (0.75, 2.75) if band_name == "mu" else (3.25, 4.25)
        for side, group in (("left", left), ("right", right)):
            for ch in ("C3", "C4"):
                est = group.mean()[list(group.ch_names).index(ch)]
                exp = _window_truth(cfg, eps[0], side, ch, band_name, depth, show)
                pick = (group.times >= window[0]) & (group.times <= window[1])
                nums[f"{band_name}_{side}_{ch}_estimated"] = float(est[pick].mean())
                nums[f"{band_name}_{side}_{ch}_expected"] = float(exp[pick].mean())
    fig = plot_erds(rows, ["C3", "C4"], significance)  # type: ignore[arg-type]
    files["erds"] = save_figure(fig, out / "erds.png")
    next(iter(significance.values())).to_csv(out / "erds_windows_mu.csv", index=False)
    significance["beta, 13-30 Hz"].to_csv(out / "erds_windows_beta.csv", index=False)

    # --- ERP --------------------------------------------------------------------------
    pooled = group_erp(eps, baseline=(-0.2, 0.0))
    peaks = find_group_peaks(pooled, ERP_CHANNELS)
    high = group_erp(eps, ["high"], baseline=(-0.2, 0.0))
    low = group_erp(eps, ["low"], baseline=(-0.2, 0.0))
    amp_high, amp_low = measure_amplitudes(high, peaks), measure_amplitudes(low, peaks)
    erp_stats = compare_amplitudes(amp_high, amp_low, method="wilcoxon_fdr")
    erp_stats.to_csv(out / "erp_components.csv", index=False)
    shift = float(np.mean([rec.truth["latency_shift"] for rec in recs]))
    for ch, plist in peaks.items():
        for pk in plist:
            nums[f"erp_{ch}_{pk.name}_latency_ms"] = pk.latency * 1000 if pk.found else float("nan")
    for k, name in enumerate(("N200", "P300", "N400")):
        nums[f"erp_{name}_true_latency_ms"] = (cfg.erp_latency[k] + shift) * 1000
    row = erp_stats[(erp_stats["channel"] == "Cz") & (erp_stats["component"] == "P300")].iloc[0]
    nums["p300_Cz_high_uV"] = float(row["mean_a"] * 1e6)
    nums["p300_Cz_low_uV"] = float(row["mean_b"] * 1e6)
    nums["p300_Cz_p_adj"] = float(row["p_adj"])
    fig = plot_erp({"high": high, "low": low}, "Cz", peaks["Cz"])
    files["erp"] = save_figure(fig, out / "erp.png")

    # --- trial-level features (the CSV contract) ----------------------------------------
    feats = features_table([e.pick(["C3", "Cz", "C4"]) for e in eps], classes=("left", "right"))
    feats.to_csv(out / "features.csv", index=False)
    files["features"] = out / "features.csv"
    nums["feature_rows"] = len(feats)
    nums["feature_columns"] = feats.shape[1] - 3

    files["report"] = out / "report.md"
    files["report"].write_text(_report(nums, erp_stats, quick), encoding="utf-8")
    return DemoResult(out, files, nums)


def _report(n: dict[str, float], erp_stats, quick: bool) -> str:
    def pct(a: float, b: float) -> str:
        return f"{a:.0f}/{b:.0f} ({100 * a / b:.0f}%)" if b else "n/a"

    lines = [
        "# eeg-erds demo (synthetic data)",
        "",
        f"{n['n_subjects']:.0f} simulated subjects, {n['n_trials_per_condition']:.0f} trials per "
        "condition. Nothing here comes from a real participant.",
    ]
    if quick:
        lines += [
            "",
            "> `--quick` uses very few subjects: with 6 subjects the smallest attainable exact "
            "p-value is 0.031, so correcting over many windows leaves almost nothing significant. "
            "Use it as a smoke test only.",
        ]
    lines += [
        "",
        "## Artefacts",
        "",
        f"* epochs kept: {n['epochs_kept']:.0f} of {n['epochs_total']:.0f}",
        f"* contaminated epochs caught by the three amplitude rules: "
        f"{pct(n['caught_three_rules'], n['artifact_epochs'])}",
        f"* ... and with the optional window-variance rule: "
        f"{pct(n['caught_four_rules'], n['artifact_epochs'])}",
        f"* clean epochs rejected by mistake: "
        f"{pct(n['clean_rejected_four_rules'], n['clean_epochs'])}",
        f"* correlation of Fp1 with the true blink signal: {n['blink_r_before']:.2f} before SOBI, "
        f"{n['blink_r_after']:.2f} after",
        "",
        "## ERD/ERS: estimated vs expected (group mean, %)",
        "",
        "| band | moving side | channel | estimated | expected |",
        "|---|---|---|---:|---:|",
    ]
    for band in ("mu", "beta"):
        for side in ("left", "right"):
            for ch in ("C3", "C4"):
                lines.append(
                    f"| {band} | {side} | {ch} | {n[f'{band}_{side}_{ch}_estimated']:.1f} | "
                    f"{n[f'{band}_{side}_{ch}_expected']:.1f} |"
                )
    lines += [
        "",
        "mu: mean over 0.75-2.75 s (ERD window); beta: mean over 3.25-4.25 s (rebound). Left vs "
        f"right trials: {n['n_significant_windows_mu']:.0f} of {n['n_windows_tested_mu']:.0f} "
        f"mu windows and {n['n_significant_windows_beta']:.0f} of "
        f"{n['n_windows_tested_beta']:.0f} beta windows differ (Wilcoxon, FDR 5%).",
        "",
        "## ERP components (group average, all conditions pooled)",
        "",
        "| channel | component | latency (ms) | high (µV) | low (µV) | p (FDR) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in erp_stats.iterrows():
        lines.append(
            f"| {r['channel']} | {r['component']} | {r['latency'] * 1000:.0f} | "
            f"{r['mean_a'] * 1e6:.2f} | {r['mean_b'] * 1e6:.2f} | {r['p_adj']:.3f} |"
        )
    lines += [
        "",
        f"True P300 latency in this simulation: {n['erp_P300_true_latency_ms']:.0f} ms "
        "(group mean). Amplitudes are means in a ±25 ms window around the pooled peak.",
        "",
        "Files: `erds.png`, `erp.png`, `blinks.png`, `erds_windows_*.csv`, `erp_components.csv`, "
        f"`features.csv` ({n['feature_rows']:.0f} rows, {n['feature_columns']:.0f} features; "
        "feed it to `subjectwise evaluate`).",
        "",
    ]
    return "\n".join(lines)
