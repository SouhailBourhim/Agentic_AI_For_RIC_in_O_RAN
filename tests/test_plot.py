"""Smoke tests for the M01 visualizations: the figures and the notebook must keep working."""

import subprocess
import sys
from pathlib import Path

from sim.plot import main

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_plot_script_writes_both_figures(tmp_path: Path) -> None:
    paths = main(out_dir=tmp_path)
    assert [p.name for p in paths] == ["M01-model.png", "M01-spike.png"]
    for p in paths:
        assert p.stat().st_size > 10_000  # a real image, not an empty file


def test_marimo_notebook_runs_top_to_bottom() -> None:
    """Running a marimo notebook as a script executes every cell; any exception fails it."""
    result = subprocess.run(
        [sys.executable, "notebooks/M01_simulator.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
