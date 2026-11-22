"""Command-line interface: ``eegerds demo | simulate | erds | peaks | features``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from . import __version__
from .data import Epochs
from .demo import run_demo
from .erds import ErdsConfig, erds_group
from .erp import find_group_peaks, group_erp, measure_amplitudes
from .features import features_table
from .io import from_mne, load_epochs, save_epochs
from .pipeline import PipelineConfig, process_group
from .stats import METHODS, compare_amplitudes, compare_conditions
from .synth import SimConfig, simulate_group

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ERD/ERS, ERP peaks, SOBI blink removal and epoch rejection for epoched EEG.",
)

Paths = Annotated[list[Path], typer.Argument(help="One epochs file per subject (.npz or .fif).")]


def _version(value: bool) -> None:
    if value:
        typer.echo(f"eegerds {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version.")
    ] = False,
) -> None:
    """eeg-erds."""


def _load(path: Path) -> Epochs:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if path.suffix == ".npz":
        epochs = load_epochs(path)
    elif path.suffix == ".fif":
        try:
            import mne
        except ImportError as exc:
            raise ImportError("reading .fif needs MNE-Python: pip install 'eeg-erds[mne]'") from exc
        epochs = from_mne(mne.read_epochs(path, preload=True, verbose="ERROR"))
    else:
        raise ValueError(f"{path}: expected a .npz or .fif file")
    return epochs if epochs.subject else _named(epochs, path.stem)


def _named(epochs: Epochs, name: str) -> Epochs:
    from dataclasses import replace

    return replace(epochs, subject=name)


def _load_all(paths: list[Path]) -> list[Epochs]:
    return [_load(p) for p in paths]


def _select(eps: list[Epochs], tag: str) -> list[Epochs]:
    """Each subject's epochs carrying ``tag``; a tag nobody has (a typo) is an error."""
    chosen = [e.select(tag) for e in eps]
    empty = [e.subject for e in chosen if e.n_epochs == 0]
    if empty:
        raise ValueError(f"no epochs tagged {tag!r} for: {', '.join(empty)}")
    return chosen


def _fail(exc: Exception) -> typer.Exit:
    typer.echo(f"error: {exc}", err=True)
    return typer.Exit(2)


@app.command()
def demo(
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("eegerds_demo"),
    subjects: Annotated[int, typer.Option(min=2, help="Simulated subjects.")] = 10,
    trials: Annotated[int, typer.Option(min=2, help="Trials per condition.")] = 10,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    quick: Annotated[bool, typer.Option(help="Tiny run for smoke tests.")] = False,
) -> None:
    """Run the full pipeline on synthetic data and write figures, tables and a report."""
    result = run_demo(out, n_subjects=subjects, n_trials=trials, seed=seed, quick=quick)
    typer.echo((result.out_dir / "report.md").read_text(encoding="utf-8"))
    typer.echo(f"Wrote {len(result.files)} files to {result.out_dir}/")


@app.command()
def simulate(
    out: Annotated[Path, typer.Argument(help="Directory for S01.npz, S02.npz, ...")],
    subjects: Annotated[int, typer.Option(min=1)] = 10,
    trials: Annotated[int, typer.Option(min=2, help="Trials per condition.")] = 10,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Simulate subjects, run the pipeline and save their clean epochs as .npz files."""
    recs = simulate_group(subjects, SimConfig(n_trials=trials), seed=seed)
    from .artifacts import RejectionConfig

    results = process_group(
        recs, PipelineConfig(rejection=RejectionConfig(max_window_sd_ratio=3.0))
    )
    for res in results:
        save_epochs(out / f"{res.subject}.npz", res.epochs)
    typer.echo(f"Wrote {len(results)} epochs files to {out}/")


@app.command()
def erds(
    files: Paths,
    tag_a: Annotated[str, typer.Option(help="Label tag of the first condition, e.g. left.")],
    tag_b: Annotated[str, typer.Option(help="Label tag of the second condition.")],
    band: Annotated[tuple[float, float], typer.Option(help="Band in Hz.")] = (8.0, 13.0),
    channels: Annotated[list[str], typer.Option("--channel", help="Channel(s) to test.")] = [  # noqa: B006
        "C3",
        "C4",
    ],
    reference: Annotated[tuple[float, float], typer.Option(help="Reference interval, s.")] = (
        -2.0,
        -1.0,
    ),
    window: Annotated[float, typer.Option(help="Window length, s.")] = 0.5,
    method: Annotated[str, typer.Option(help=f"One of {', '.join(METHODS)}.")] = "wilcoxon_fdr",
    alpha: Annotated[float, typer.Option(help="Significance level.")] = 0.05,
    out: Annotated[Path, typer.Option(help="CSV with one row per channel and window.")] = Path(
        "erds_windows.csv"
    ),
) -> None:
    """Compare ERD/ERS between two conditions across subjects, window by window."""
    try:
        eps = _load_all(files)
        cfg = ErdsConfig(band=band, window=window, overlap=0.0, reference=reference)
        a = erds_group(_select(eps, tag_a), cfg)
        b = erds_group(_select(eps, tag_b), cfg)
        table = compare_conditions(a, b, channels, method=method, alpha=alpha)  # type: ignore[arg-type]
    except (ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        raise _fail(exc) from exc
    table.to_csv(out, index=False)
    hits = table[table["significant"]]
    typer.echo(f"{len(hits)} of {len(table)} windows significant ({method}, alpha={alpha}).")
    typer.echo(f"Smallest attainable p with {len(eps)} subjects: {table['p_floor'].iloc[0]:.4f}")
    typer.echo(f"Wrote {out}")


@app.command()
def peaks(
    files: Paths,
    channels: Annotated[list[str], typer.Option("--channel", help="Channel(s).")] = ["Cz"],  # noqa: B006
    tag_a: Annotated[str | None, typer.Option(help="First condition to compare.")] = None,
    tag_b: Annotated[str | None, typer.Option(help="Second condition to compare.")] = None,
    baseline: Annotated[tuple[float, float], typer.Option(help="Baseline interval, s.")] = (
        -0.2,
        0.0,
    ),
    out: Annotated[Path, typer.Option(help="CSV output.")] = Path("erp_components.csv"),
) -> None:
    """N200/P300/N400 of the pooled average; with two tags, also compare their amplitudes."""
    try:
        if (tag_a is None) != (tag_b is None):
            raise ValueError("--tag-a and --tag-b go together: give both or neither")
        eps = _load_all(files)
        pooled = group_erp(eps, baseline=baseline)
        found = find_group_peaks(pooled, channels)
        if tag_a and tag_b:
            table = compare_amplitudes(
                measure_amplitudes(group_erp(eps, [tag_a], baseline), found),
                measure_amplitudes(group_erp(eps, [tag_b], baseline), found),
            )
        else:
            table = pd.DataFrame(
                [
                    (ch, p.name, p.latency, p.amplitude, p.found, p.at_edge)
                    for ch, plist in found.items()
                    for p in plist
                ],
                columns=["channel", "component", "latency", "amplitude", "found", "at_edge"],
            )
    except (ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        raise _fail(exc) from exc
    table.to_csv(out, index=False)
    for ch, plist in found.items():
        text = ", ".join(
            f"{p.name} {p.latency * 1000:.0f} ms" if p.found else f"{p.name} not found"
            for p in plist
        )
        typer.echo(f"{ch}: {text}")
    typer.echo(f"Wrote {out}")


@app.command()
def features(
    files: Paths,
    classes: Annotated[list[str], typer.Option("--class", help="Tags that define the classes.")],
    channels: Annotated[list[str] | None, typer.Option("--channel", help="Keep these.")] = None,
    out: Annotated[Path, typer.Option(help="CSV output.")] = Path("features.csv"),
) -> None:
    """Trial-level ERD/ERS features as ``subject, level, trial, <channel>_<band>_<t0>-<t1>s``."""
    try:
        eps = _load_all(files)
        if channels:
            eps = [e.pick(channels) for e in eps]
        table = features_table(eps, classes=classes)
    except (ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        raise _fail(exc) from exc
    table.to_csv(out, index=False)
    counts = table["level"].value_counts().to_dict()
    typer.echo(f"{len(table)} rows ({counts}), {table.shape[1] - 3} features -> {out}")
