"""The examples and the README snippets must only call things that exist, and still run."""

import ast
import importlib.util
import re
from pathlib import Path

import pytest

import eegerds as ee

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))
READMES = [p for p in (ROOT / "README.md", ROOT / "README.pt-BR.md") if p.exists()]


def _missing_attributes(source: str) -> list[str]:
    tree = ast.parse(source)
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "ee"
    }
    return sorted(name for name in used if not hasattr(ee, name))


def _load_example(name: str):
    spec = importlib.util.spec_from_file_location(
        f"example_{name}", ROOT / "examples" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_scripts_use_only_the_public_api(path):
    assert _missing_attributes(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", READMES, ids=lambda p: p.name)
def test_readme_python_snippets_parse_and_use_the_public_api(path):
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL)
    assert blocks, "the README should show at least one Python snippet"
    for block in blocks:
        assert _missing_attributes(block) == []


@pytest.mark.parametrize("path", READMES, ids=lambda p: p.name)
def test_readme_cli_commands_exist(path):
    from typer.testing import CliRunner

    from eegerds.cli import app

    commands = set(re.findall(r"^eegerds (\w+)", path.read_text(encoding="utf-8"), re.MULTILINE))
    assert {"demo", "simulate", "erds", "peaks", "features"} <= commands
    for command in commands:
        assert CliRunner().invoke(app, [command, "--help"]).exit_code == 0


def test_the_false_positive_counts_in_the_readmes_are_those_of_the_saved_runs():
    import pandas as pd

    runs = pd.read_csv(ROOT / "docs" / "fwer_runs.csv")
    n = len(runs)
    rules = ("no correction", "FDR (Benjamini-Hochberg)", "sign-flip max-T", "cluster permutation")
    for path in READMES:
        text = path.read_text(encoding="utf-8")
        for rule in rules:
            hits = int((runs[rule] > 0).sum())
            assert re.search(rf"\b{hits}\s+(of|de)\s+{n}\b", text), (path.name, rule, hits)


def test_quickstart_runs_and_prints_the_tour(capsys):
    _load_example("quickstart").main()
    out = capsys.readouterr().out
    for expected in (
        "kept",
        "blink component removed",
        "left minus right",
        "P300",
        "feature table",
    ):
        assert expected in out


def test_wilson_interval_matches_known_values():
    wilson = _load_example("false_positive_windows").wilson
    assert wilson(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson(50, 200)
    assert lo < 0.25 < hi and hi - lo < 0.13


def test_false_positive_study_runs_on_a_tiny_design(tmp_path, capsys):
    study = _load_example("false_positive_windows")
    out = tmp_path / "fwer.png"
    table = study.main(["--reps", "2", "--subjects", "4", "--trials", "3", "--jobs", "1",
                        "--out", str(out)])  # fmt: skip
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert list(table["rule"]) == [
        "no correction",
        "FDR (Benjamini-Hochberg)",
        "sign-flip max-T",
        "cluster permutation",
    ]
    assert "4 subjects, 16 windows per experiment" in capsys.readouterr().out
