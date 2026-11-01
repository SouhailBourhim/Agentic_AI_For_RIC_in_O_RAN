"""Smoke tests for the M01 visualizations: the figures and the notebook must keep working."""

import subprocess
import sys
from pathlib import Path

import pytest

from sim.plot import main

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_plot_script_writes_both_figures(tmp_path: Path) -> None:
    paths = main(out_dir=tmp_path)
    assert [p.name for p in paths] == ["M01-model.png", "M01-spike.png"]
    for p in paths:
        assert p.stat().st_size > 10_000  # a real image, not an empty file


@pytest.mark.parametrize("notebook", ["M01_simulator.py", "M02_policy.py"])
def test_marimo_notebook_runs_top_to_bottom(notebook: str) -> None:
    """Running a marimo notebook as a script executes every cell; any exception fails it.

    M02_policy.py stops early (no exception) when no trained model exists, e.g. on a fresh clone.
    """
    result = subprocess.run(
        [sys.executable, f"notebooks/{notebook}"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_verifier_feasibility_map_and_table(tmp_path: Path) -> None:
    """M03: the map is written, and the decision table shows every check at least once."""
    from verifier.plot import decision_table
    from verifier.plot import main as verifier_plot_main

    path = verifier_plot_main(out_dir=tmp_path, step_mbps=2.5)  # coarse grid: fast
    assert path.name == "M03-feasibility.png" and path.stat().st_size > 10_000
    table = decision_table()
    for check in ["ok", "schema", "window", "latency_physics", "budget", "floor", "training_range",
                  "conflict", "step"]:
        assert f" {check} " in table
