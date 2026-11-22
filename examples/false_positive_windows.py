"""How often does a window-by-window test find a difference that is not there?

Simulated experiments in which left and right trials are *identical* at C3 (``ipsi_ratio=1``: both
hemispheres desynchronise equally, so there is nothing to find). Each experiment is run through the
whole pipeline and tested in 16 half-second windows. The share of experiments with at least one
"significant" window should be 5%; it is much larger without a correction.

    python examples/false_positive_windows.py                 # 200 experiments, a few minutes
    python examples/false_positive_windows.py --reps 20       # quick look (+/- 8 points of noise)

The figure goes to ``docs/img/fwer.png``.
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

import eegerds as ee
from eegerds.viz import plot_fwer, save_figure

ROOT = Path(__file__).resolve().parent.parent
METHODS = {
    "no correction": "wilcoxon",
    "FDR (Benjamini-Hochberg)": "wilcoxon_fdr",
    "sign-flip max-T": "signflip_maxt",
}


def replicate(task: tuple[int, int, int]) -> dict[str, int]:
    """One null experiment: how many windows does each rule call significant?"""
    seed, n_subjects, n_trials = task
    recordings = ee.simulate_group(
        n_subjects, ee.SimConfig(n_trials=n_trials, ipsi_ratio=1.0), seed
    )
    config = ee.PipelineConfig(rejection=ee.RejectionConfig(max_window_sd_ratio=3.0))
    epochs = [r.epochs for r in ee.process_group(recordings, config)]
    erds = ee.ErdsConfig(band=(8, 13), window=0.5, overlap=0.0)
    left = ee.erds_group([e.select("left") for e in epochs], erds)
    right = ee.erds_group([e.select("right") for e in epochs], erds)
    counts = {"n_tests": 0}
    for label, method in METHODS.items():
        table = ee.compare_conditions(left, right, ["C3"], method=method)  # type: ignore[arg-type]
        counts[label] = int(table["significant"].sum())
        counts["n_tests"] = len(table)
    return counts


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion."""
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--reps", type=int, default=200, help="simulated experiments")
    parser.add_argument("--subjects", type=int, default=8)
    parser.add_argument("--trials", type=int, default=10, help="trials per condition")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 4))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "img" / "fwer.png")
    parser.add_argument("--csv", type=Path, help="also save the per-experiment counts here")
    args = parser.parse_args(argv)

    tasks = [(args.seed + i, args.subjects, args.trials) for i in range(args.reps)]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            runs = list(pool.map(replicate, tasks, chunksize=1))
    else:
        runs = [replicate(t) for t in tasks]
    frame = pd.DataFrame(runs)
    n_tests = int(frame["n_tests"].iloc[0])

    rows, rates, intervals = [], {}, {}
    for label in METHODS:
        hits = int((frame[label] > 0).sum())
        lo, hi = wilson(hits, args.reps)
        rates[label] = hits / args.reps
        intervals[label] = (lo, hi)
        rows.append(
            {
                "rule": label,
                "experiments with a false positive": f"{hits}/{args.reps}",
                "rate": f"{100 * hits / args.reps:.0f}%",
                "95% interval": f"{100 * lo:.0f}-{100 * hi:.0f}%",
            }
        )
    summary = pd.DataFrame(rows)
    print(f"{args.subjects} subjects, {n_tests} windows per experiment, no real effect:\n")
    print(summary.to_string(index=False))
    save_figure(plot_fwer(rates, args.reps, n_tests, intervals), args.out)
    print(f"\nfigure: {args.out}")
    if args.csv:
        frame.to_csv(args.csv, index_label="experiment")
        print(f"counts: {args.csv}")
    return summary


if __name__ == "__main__":
    main()
