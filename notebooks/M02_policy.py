"""Play with the M02 policy: change the objective, see how the trained policy splits the PRBs.

Run with:  uv run marimo edit notebooks/M02_policy.py
(needs trained models in models/m02/; see docs/learn/M02-ddpg.md)

Same URLLC-spike traffic as M01. You set the objective the LLM would send (weights, URLLC
minimum throughput, latency bound); the left column is the equal split, the right column the
trained policy, both judged by the reward of that objective. Moving a slider re-runs the
episode, so you see goal conditioning directly: the policy's split changes, the equal split
does not.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    import marimo as mo

    # The notebook lives in notebooks/; put the repo root on the path so `import rl` works.
    root = mo.notebook_dir().parent
    sys.path.insert(0, str(root))
    return mo, root


@app.cell
def _(mo):
    mo.md("""
    # M02 goal-conditioned policy

    Choose a **policy**, an **objective** and the **URLLC spike size**. The weights are
    divided by their sum (an `RLObjective` needs weights summing to 1). The smoothing filter
    (`rl/policy.py`) is on by default, as deployed.
    """)
    return


@app.cell
def _(mo, root):
    from mlops.registry import champion

    # Every trained model on disk (M02 rounds, M02b confirmation checkpoints), by name.
    paths = {p.stem: p for p in sorted((root / "models" / "m02").glob("*_seed*.zip"))}
    paths.update({p.stem: p for p in sorted((root / "models" / "m02b" / "confirm").glob("*_best.zip"))})
    deployed = champion()  # (name, file) of the registry's `champion`, None without a registry
    if deployed is not None:
        paths.setdefault(deployed[0], deployed[1])
    names = sorted(paths)
    mo.stop(not names, mo.md("**No trained models in `models/`.** Train first (see the M02 note)."))
    default = deployed[0] if deployed is not None else names[0]
    policy_name = mo.ui.dropdown(names, value=default, label="policy")
    smoothing = mo.ui.checkbox(value=True, label="smoothing filter (alpha = 0.25)")
    w_embb = mo.ui.slider(0.0, 1.0, step=0.05, value=0.2, label="weight eMBB")
    w_urllc = mo.ui.slider(0.0, 1.0, step=0.05, value=0.6, label="weight URLLC")
    w_be = mo.ui.slider(0.0, 1.0, step=0.05, value=0.2, label="weight BE")
    urllc_min = mo.ui.slider(0.0, 0.5, step=0.05, value=0.3, label="URLLC minimum (fraction of capacity)")
    use_bound = mo.ui.checkbox(value=True, label="URLLC latency bound")
    bound_ms = mo.ui.slider(2.0, 50.0, step=1.0, value=10.0, label="bound (ms)")
    peak = mo.ui.slider(0.1, 0.8, step=0.05, value=0.40, label="URLLC spike peak (fraction of capacity)")
    mo.vstack([policy_name, smoothing, w_embb, w_urllc, w_be, urllc_min, mo.hstack([use_bound, bound_ms]), peak])
    return bound_ms, paths, peak, policy_name, smoothing, urllc_min, use_bound, w_be, w_embb, w_urllc


@app.cell
def _(bound_ms, paths, peak, policy_name, smoothing, urllc_min, use_bound, w_be, w_embb, w_urllc):
    import numpy as np

    from rl.baselines import equal_split
    from rl.evaluate import SPIKE_STEPS
    from rl.policy import SMOOTHING_ALPHA, SlicePolicy
    from sim.cell import CellConfig
    from sim.plot import plot_spike_comparison, run_episode
    from sim.scenarios import urllc_spike
    from verifier.objective import RLObjective, SliceValues

    raw = np.array([w_embb.value, w_urllc.value, w_be.value])
    weights = raw / raw.sum() if raw.sum() > 0 else np.full(3, 1 / 3)
    objective = RLObjective(
        weights=SliceValues(embb=weights[0], urllc=weights[1], be=weights[2]),
        min_thr_mbps=SliceValues(urllc=urllc_min.value * CellConfig().capacity_mbps),
        max_latency_ms=bound_ms.value if use_bound.value else None,
    )
    policy = SlicePolicy.load(paths[policy_name.value],
                              alpha=SMOOTHING_ALPHA if smoothing.value else 1.0)
    traffic, _ = urllc_spike(spike_fraction=peak.value)
    runs = {
        "Equal split": run_episode(equal_split, traffic, objective, steps=SPIKE_STEPS),
        f"{policy_name.value}": run_episode(policy, traffic, objective, steps=SPIKE_STEPS),
    }
    fig = plot_spike_comparison(runs, objective, traffic["urllc"], what="— equal split vs trained policy")
    fig
    return objective, runs


@app.cell
def _(mo, objective, runs):
    bound = objective.max_latency_ms
    rows = [
        {"controller": name,
         "mean reward": round(float(run["reward"].mean()), 3),
         "mean URLLC share": round(float(run["share_urllc"].mean()), 3),
         "URLLC latency within bound": "n/a" if bound is None else f"{(run['lat_urllc'] <= bound).mean():.0%}"}
        for name, run in runs.items()
    ]
    mo.ui.table(rows, selection=None)
    return


if __name__ == "__main__":
    app.run()
