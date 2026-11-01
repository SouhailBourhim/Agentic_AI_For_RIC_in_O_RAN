"""Play with the M01 simulator: change the PRB split and the URLLC spike, see the effect.

Run with:  uv run marimo edit notebooks/M01_simulator.py
(or `uv run marimo run notebooks/M01_simulator.py` for a read-only app view)

A marimo notebook is a plain Python file. Cells re-run automatically when a value they
depend on changes, so moving a slider redraws the figure. All the logic lives in sim/;
this notebook only adds sliders on top of sim/plot.py.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    import marimo as mo

    # The notebook lives in notebooks/; put the repo root on the path so `import sim` works.
    sys.path.insert(0, str(mo.notebook_dir().parent))
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # M01 simulator

    Pick a **fixed PRB split** and a **URLLC spike size**. The figure shows throughput vs
    offered load, latency, PRB shares and reward over 30 s, with the equal split on the
    left for comparison. Shares are divided by their sum, as in `sim/cell.py`.
    """)
    return


@app.cell
def _(mo):
    embb = mo.ui.slider(0.0, 1.0, step=0.01, value=0.40, label="eMBB weight of split")
    urllc = mo.ui.slider(0.0, 1.0, step=0.01, value=0.42, label="URLLC weight of split")
    be = mo.ui.slider(0.0, 1.0, step=0.01, value=0.18, label="BE weight of split")
    peak = mo.ui.slider(0.1, 1.0, step=0.05, value=0.40, label="URLLC spike peak (fraction of capacity)")
    mo.vstack([embb, urllc, be, peak])
    return be, embb, peak, urllc


@app.cell
def _(be, embb, peak, urllc):
    import numpy as np

    from sim.cell import shares_from_action
    from sim.plot import plot_spike_comparison, run_fixed_split
    from sim.scenarios import urllc_spike

    traffic, objective = urllc_spike(spike_fraction=peak.value)
    chosen = shares_from_action(np.array([embb.value, urllc.value, be.value]))
    label = f"Chosen split ({chosen[0]:.2f} / {chosen[1]:.2f} / {chosen[2]:.2f})"
    runs = {
        "Equal split (1/3 each)": run_fixed_split(np.full(3, 1 / 3), traffic, objective),
        label: run_fixed_split(chosen, traffic, objective),
    }
    fig = plot_spike_comparison(runs, objective, traffic["urllc"])
    fig
    return (runs,)


@app.cell
def _(mo, runs):
    rows = [
        {"split": name, "mean reward": round(float(run["reward"].mean()), 3),
         "max URLLC latency (ms)": round(float(run["lat_urllc"].max()), 1)}
        for name, run in runs.items()
    ]
    mo.ui.table(rows, selection=None)
    return


if __name__ == "__main__":
    app.run()
