"""Figure for M04a: one intent travelling through the whole plumbing, in a fast simulated run.

Run with:  uv run python -m rl.plot_runner      (about 10 s; writes docs/learn/img/M04a-runner.png)

Same code path as the live demo, in one process and at speed 0: the RL runner on the `demo`
scenario (URLLC surges from 10 % to 40 % of capacity at t=120 s), the deployed champion, and at
t=150 s the intent "garantir 20 Mb/s au slice URLLC avec une latence de 10 ms pendant 5 minutes"
submitted and confirmed (evaluation mode). The objective ends at t=450 s; the runner falls back
to the default objective by itself. A temporary database is used, var/ric.db is not touched.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from agent.tools import AgentContext, Settings
from agent.worker import submit_and_run
from backends.sim import SimBackend, scenario_traffic
from rl.runner import Runner, load_policy
from sim.cell import CellConfig
from sim.plot import SLICE_COLOURS, SLICE_LABELS, SURFACE, TEXT, TEXT_MUTED, _style
from sim.scenarios import DEMO_SURGE_END_S, DEMO_SURGE_START_S
from store import db, kpi
from verifier.objective import SLICES

INTENT = "garantir 20 Mb/s au slice URLLC avec une latence de 10 ms pendant 5 minutes"
SUBMIT_AT_S = 150
RUN_S = 700
OUT = Path("docs/learn/img/M04a-runner.png")


def simulate(policy_file: Path | None = None) -> tuple[dict[str, np.ndarray], dict, str]:
    """Run the scenario; return per-slice KPI arrays, the intent's record and the model label."""
    work = Path(tempfile.mkdtemp(prefix="m04a_"))
    os.environ["RIC_EVENTS"] = str(work / "events.jsonl")
    conn = db.connect(work / "ric.db")
    policy, label = load_policy(policy_file)
    cfg = CellConfig()
    backend = SimBackend(cfg, scenario_traffic("demo", cfg), seed=0, speed=0.0)
    kpi.write_backend_info(conn, backend.info(), label)
    runner = Runner(backend, policy, db.connect(work / "ric.db", timeout_s=0.0), cfg.dt_s)
    runner.run(SUBMIT_AT_S)
    # The rule-based agent, evaluation mode, no observation loop: the same objective every run.
    agent_ctx = AgentContext(conn=conn, settings=Settings(mode="eval", observe_s=0.0, max_revisions=0), llm=None)
    record = submit_and_run(agent_ctx, INTENT)
    runner.run(RUN_S - SUBMIT_AT_S)
    rows = conn.execute("SELECT ts, slice, dl_thr_mbps, prb_share_alloc, latency_ms, offered_load_mbps, policy_id"
                        " FROM kpi ORDER BY ts").fetchall()
    data = {s: np.array([[r["ts"], r["dl_thr_mbps"], r["prb_share_alloc"], r["latency_ms"], r["offered_load_mbps"],
                          r["policy_id"] or 0] for r in rows if r["slice"] == s]) for s in SLICES}
    return data, {"state": record.state, "objective": record.objective}, label


def plot(data: dict[str, np.ndarray], record: dict, label: str) -> Figure:
    objective = record["objective"]
    start, end = objective.valid_from, objective.valid_until
    fig = Figure(figsize=(12, 9), facecolor=SURFACE, layout="constrained")
    ax_share, ax_lat, ax_thr = fig.subplots(3, 1, sharex=True)
    for s in SLICES:
        t, thr, share, lat, load = (data[s][:, i] for i in range(5))
        ax_share.plot(t, share, color=SLICE_COLOURS[s], linewidth=1.6, label=SLICE_LABELS[s])
        ax_lat.plot(t, lat, color=SLICE_COLOURS[s], linewidth=1.6, label=SLICE_LABELS[s])
        ax_thr.plot(t, thr, color=SLICE_COLOURS[s], linewidth=1.6, label=f"{SLICE_LABELS[s]} throughput")
        ax_thr.plot(t, load, color=SLICE_COLOURS[s], linewidth=1.0, linestyle=(0, (4, 3)))
    ax_lat.axhline(objective.max_latency_ms, color=TEXT_MUTED, linewidth=1.0, linestyle=(0, (2, 3)))
    ax_lat.annotate(f"URLLC bound {objective.max_latency_ms:g} ms", (RUN_S, objective.max_latency_ms),
                    xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=8, color=TEXT_MUTED)
    ax_thr.axhline(objective.min_thr_mbps.urllc, color=SLICE_COLOURS["urllc"], linewidth=1.0, linestyle=(0, (2, 3)))
    ax_lat.set_yscale("log")
    for ax in (ax_share, ax_lat, ax_thr):
        ax.axvspan(DEMO_SURGE_START_S, DEMO_SURGE_END_S, color="#efeee9", zorder=0)
        ax.axvspan(start, end, color="#e8e4fb", zorder=0)
    _style(ax_share, "PRB share")
    _style(ax_lat, "latency (ms, log)")
    _style(ax_thr, "Mb/s (dashed: offered load)")
    ax_thr.set_xlabel("backend clock (s)", color=TEXT_MUTED, fontsize=9)
    handles, labels = ax_share.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=9)
    y_top = ax_share.get_ylim()[1]
    ax_share.text((DEMO_SURGE_START_S + start) / 2, y_top, "URLLC surge\n(default objective)", ha="center",
                  va="top", fontsize=8, color=TEXT_MUTED)
    ax_share.text((start + end) / 2, y_top, f"intent applied: a1_policy segment, t={start:g}-{end:g} s",
                  ha="center", va="top", fontsize=8, color=TEXT)
    fig.suptitle(f"One intent through the plumbing: \"{INTENT}\" ({label})", color=TEXT, fontsize=10.5)
    return fig


def main() -> None:
    data, record, label = simulate()
    assert record["state"] == "applied", record
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plot(data, record, label).savefig(OUT, dpi=110)
    print(f"wrote {OUT}")
    for name, (a, b) in {"surge, default": (130, SUBMIT_AT_S), "surge, intent": (200, 440),
                         "surge, after expiry": (460, DEMO_SURGE_END_S)}.items():
        window = {s: data[s][(data[s][:, 0] > a) & (data[s][:, 0] <= b)] for s in SLICES}
        print(f"{name:20s} " + "  ".join(
            f"{s}: share {w[:, 2].mean():.2f} thr {w[:, 1].mean():5.1f} lat {np.median(w[:, 3]):8.1f} ms"
            for s, w in window.items()))


if __name__ == "__main__":
    main()
