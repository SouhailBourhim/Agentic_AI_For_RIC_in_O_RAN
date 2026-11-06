"""Does a late observation make the policy oscillate? The M07b testbed check, on the simulator.

  uv run python testbed/sim_delay_m07b.py

On the testbed the RL split swings (eMBB share 0.22 <-> 0.84, period 10-12 s) during the URLLC
surge, and the KPM report shows a split's effect one tick later than the simulator does. Here the
same policy (the registry's champion, smoothing on, as deployed) runs on the simulator's demo
surge with its KPI records held back by `delay` ticks, so the policy decides on older values;
nothing else changes. If the delay alone makes it swing, the swing should grow with the delay.

Writes testbed/results/m07b/sim_delay.json and docs/learn/img/M07b-sim-delay.png.
"""

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from backends.sim import DelayedBackend, scenario_traffic  # noqa: E402
from rl.runner import Runner, load_policy  # noqa: E402
from sim.cell import CellConfig  # noqa: E402
from sim.plot import LINE_WIDTH, SLICE_COLOURS, SURFACE, TEXT, TEXT_MUTED, _style  # noqa: E402
from sim.scenarios import DEMO_SURGE_START_S  # noqa: E402
from store import db  # noqa: E402

DELAYS = [0, 1, 2, 3]
TICKS = 420
WINDOW = (int(DEMO_SURGE_START_S) + 30, TICKS)  # surge, after its first 30 s


def run(delay: int, policy_file: Path) -> dict:
    cfg = CellConfig()
    backend = DelayedBackend(cfg, scenario_traffic("demo", cfg), seed=0, delay=delay)
    policy, _ = load_policy(policy_file)
    runner = Runner(backend, policy, db.connect(timeout_s=0.0), cfg.dt_s)
    shares = np.array([runner.tick().shares for _ in range(TICKS)])
    window = shares[WINDOW[0]:WINDOW[1], 0]  # eMBB
    x = window - window.mean()
    freqs, power = np.fft.rfftfreq(len(x), 1.0), np.abs(np.fft.rfft(x))
    k = int(np.argmax(power[1:]) + 1)
    return {"delay": delay, "embb_share": shares[:, 0].tolist(), "urllc_share": shares[:, 1].tolist(),
            "surge_embb_min": float(window.min()), "surge_embb_max": float(window.max()),
            "surge_mean_abs_change": float(np.mean(np.abs(np.diff(window)))),
            "surge_period_s": float(1 / freqs[k])}


def main() -> None:
    import os
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    os.environ["RIC_DB"], os.environ["RIC_EVENTS"] = str(tmp / "ric.db"), str(tmp / "events.jsonl")
    _, label = load_policy(None)  # the champion, as the runner loads it
    served = json.loads(Path("var/policy_cache/served.json").read_text())
    policy_file = Path("var/policy_cache") / served["file"]
    results = [run(d, policy_file) for d in DELAYS]
    out = Path("testbed/results/m07b/sim_delay.json")
    out.write_text(json.dumps({"policy": label, "window_s": WINDOW, "runs": results}))
    for r in results:
        print(f"delay {r['delay']} tick(s): eMBB share in the surge {r['surge_embb_min']:.2f}-{r['surge_embb_max']:.2f}, "
              f"mean change {r['surge_mean_abs_change']:.3f} per s, dominant period {r['surge_period_s']:.1f} s")

    fig = Figure(figsize=(11, 6.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(len(results), 1, sharex=True)
    for ax, r in zip(axes, results):
        t = np.arange(len(r["embb_share"]))
        ax.plot(t, r["embb_share"], color=SLICE_COLOURS["embb"], linewidth=LINE_WIDTH, label="eMBB")
        ax.plot(t, r["urllc_share"], color=SLICE_COLOURS["urllc"], linewidth=LINE_WIDTH, label="URLLC")
        ax.axvline(DEMO_SURGE_START_S, color=TEXT_MUTED, linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1)
        _style(ax, f"delay {r['delay']}")
        ax.annotate(f"surge: {r['surge_mean_abs_change'] * 100:.0f} % of the cell moved per s",
                    (0.99, 0.92), xycoords="axes fraction", ha="right", va="top", fontsize=8.5, color=TEXT_MUTED)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left", ncols=2)
    axes[-1].set_xlabel("simulated time, s (URLLC surge from the dashed line)", color=TEXT)
    fig.suptitle(f"Simulator, {label}: PRB share with the KPIs held back by 0-3 ticks", color=TEXT, fontsize=11,
                 x=0.01, ha="left")
    fig.savefig("docs/learn/img/M07b-sim-delay.png", dpi=120)
    print("wrote docs/learn/img/M07b-sim-delay.png")


if __name__ == "__main__":
    main()
