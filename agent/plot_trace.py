"""Figure for M04b: one intent through the LangGraph agent, node by node, with the real LLM.

Run with:  uv run python -m agent.plot_trace      (needs Ollama with the agent's model; ~1 min)

Same code path as the live demo, in one process: the RL runner on the `demo` scenario (URLLC
surges at t=120 s) with the deployed champion at speed 0, and at t=150 s the agent (LLM, eval
mode, 20 s observation, at most one revision) handles a French intent. A temporary database is
used, var/ric.db is not touched. Writes docs/learn/img/M04b-trace.png and the step trace to
docs/learn/data/M04b-trace.json (the note walks through it).

Top panel: what the agent did, on the wall clock (each bar ends when the node logged its
step). Bottom panel: what the cell did, on the backend clock (URLLC throughput and PRB shares).
At speed 0 the 20 s observation takes almost no wall time; in the live demo it takes 20 s.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from agent.llm import OllamaClient
from agent.tools import AgentContext, Settings
from agent.worker import submit_and_run
from backends.sim import SimBackend, scenario_traffic
from rl.runner import Runner, load_policy
from sim.cell import CellConfig
from sim.plot import SLICE_COLOURS, SLICE_LABELS, SURFACE, TEXT, TEXT_MUTED, _style
from sim.scenarios import DEMO_SURGE_START_S
from store import db, kpi
from store.events import read_events
from verifier.objective import SLICES

INTENT = "Protège l'URLLC : au moins 20 Mb/s et une latence sous 10 ms pendant 5 minutes."
SUBMIT_AT_S = 150
RUN_S = 520
OUT = Path("docs/learn/img/M04b-trace.png")
TRACE_OUT = Path("docs/learn/data/M04b-trace.json")
NODES = ["parse", "plan", "act", "confirm", "observe", "reflect"]
BAR = "#6b6a66"  # neutral: the bars are steps, not slices
BAR_LLM = "#2a78d6"  # nodes that call the LLM


def simulate() -> tuple[dict[str, np.ndarray], list[dict], dict, str]:
    """Run the scenario with the LLM agent; return KPI arrays, the step events, the record, the model."""
    work = Path(tempfile.mkdtemp(prefix="m04b_"))
    os.environ["RIC_EVENTS"] = str(work / "events.jsonl")
    conn = db.connect(work / "ric.db")
    policy, label = load_policy(None)
    cfg = CellConfig()
    backend = SimBackend(cfg, scenario_traffic("demo", cfg), seed=0, speed=0.0)
    kpi.write_backend_info(conn, backend.info(), label)
    runner = Runner(backend, policy, db.connect(work / "ric.db", timeout_s=0.0), cfg.dt_s)
    runner.run(SUBMIT_AT_S)
    llm = OllamaClient()
    settings = Settings(mode="eval", observe_s=20.0, max_revisions=1)
    ctx = AgentContext(conn=conn, settings=settings, llm=llm, wait_backend=lambda s: runner.run(int(s)))
    submitted_wall = time.time()
    record = submit_and_run(ctx, INTENT)
    runner.run(RUN_S - int(backend.now()))
    rows = conn.execute("SELECT ts, slice, dl_thr_mbps, prb_share_alloc FROM kpi ORDER BY ts").fetchall()
    data = {s: np.array([[r["ts"], r["dl_thr_mbps"], r["prb_share_alloc"]] for r in rows if r["slice"] == s])
            for s in SLICES}
    steps = [e for e in read_events() if e["event"] == "agent_step"]
    for e in steps:
        e["t_wall"] = round(e["ts_wall"] - submitted_wall, 2)
    summary = {"model": llm.model, "intent": INTENT, "state": record.state, "reason": record.reason,
               "objective": record.objective.model_dump() if record.objective else None}
    return data, steps, summary, label


def plot(data: dict[str, np.ndarray], steps: list[dict], summary: dict, label: str) -> Figure:
    fig = Figure(figsize=(12, 8.5), facecolor=SURFACE, layout="constrained")
    top, bottom = fig.subplots(2, 1, height_ratios=[1, 1.3])

    # Top: one row per node, a bar from the previous step's end to this step's end (wall clock).
    rows = {n: i for i, n in enumerate(reversed(NODES))}
    start = 0.0
    for e in steps:
        node, end = e["data"]["node"], e["t_wall"]
        llm_node = "latency_s" in e["data"]
        top.barh(rows[node], max(end - start, 0.05), left=start, height=0.55,
                 color=BAR_LLM if llm_node else BAR, edgecolor=SURFACE, linewidth=2)
        top.text(end + 0.1, rows[node], step_label(e["data"]), va="center", fontsize=8, color=TEXT_MUTED)
        start = end
    top.set_yticks(list(rows.values()), list(rows.keys()))
    top.set_xlim(0, start * 1.45 + 0.5)
    _style(top, "")
    top.set_xlabel("wall-clock seconds since the intent was submitted", color=TEXT_MUTED)
    top.set_title(f"Agent steps ({summary['model']}; blue = LLM call): “{summary['intent']}”",
                  loc="left", color=TEXT, fontsize=11)

    # Bottom: the cell on the backend clock.
    for s in SLICES:
        bottom.plot(data[s][:, 0], data[s][:, 2] * 100, color=SLICE_COLOURS[s], linewidth=2,
                    label=f"{SLICE_LABELS[s]} PRB share (%)")
    urllc = data["urllc"]
    bottom.plot(urllc[:, 0], urllc[:, 1], color=SLICE_COLOURS["urllc"], linewidth=1.2, linestyle=(0, (4, 2)),
                label="URLLC throughput (Mb/s)")
    objective = summary["objective"]
    if objective:
        target = objective["min_thr_mbps"]["urllc"]
        bottom.axvspan(objective["valid_from"], objective["valid_until"], color=TEXT_MUTED, alpha=0.07, linewidth=0)
        if target:
            bottom.hlines(target, objective["valid_from"], objective["valid_until"], color=TEXT, linewidth=1,
                          linestyle=":", label=f"URLLC target {target:g} Mb/s")
    bottom.axvline(DEMO_SURGE_START_S, color=TEXT_MUTED, linewidth=1, linestyle="--")
    bottom.text(DEMO_SURGE_START_S + 3, 2, "URLLC surge", color=TEXT_MUTED, fontsize=8)
    observe = [e for e in steps if e["data"]["node"] == "observe"]
    for e in observe:
        bottom.axvline(e["ts_backend"], color=TEXT, linewidth=0.8)
        bottom.text(e["ts_backend"] + 3, 92, "observed", color=TEXT_MUTED, fontsize=8)
    bottom.set_xlim(DEMO_SURGE_START_S - 40, data["urllc"][-1, 0])
    bottom.set_ylim(0, 100)
    _style(bottom, "% of PRBs  /  Mb/s")
    bottom.set_xlabel(f"backend seconds (shaded: objective in force; policy {label})", color=TEXT_MUTED)
    bottom.legend(loc="upper right", fontsize=8, frameon=False, ncols=2)
    return fig


def step_label(d: dict) -> str:
    """A few words per step, for the timeline."""
    if d["node"] == "parse":
        return f"prompt {d.get('prompt')}, {d.get('latency_s')} s"
    if d["node"] == "plan":
        return f"-> {d.get('tool')}, {d.get('latency_s')} s"
    if d["node"] == "act":
        verdict = d.get("verifier")
        if verdict:
            return f"{d['tool']}: verifier {'accept' if verdict['accepted'] else 'reject (' + verdict['check'] + ')'}"
        return str(d.get("tool"))
    if d["node"] == "reflect":
        return f"{d.get('decision')}, {d.get('latency_s')} s"
    if d["node"] == "observe":
        comp = d.get("compliance") or {}
        return "compliance " + ", ".join(f"{k} {v:.0%}" for k, v in comp.items()) if comp else "KPIs read"
    return str(d.get("result", ""))


def main() -> None:
    data, steps, summary, label = simulate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plot(data, steps, summary, label).savefig(OUT, dpi=130, facecolor=SURFACE)
    TRACE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TRACE_OUT.write_text(json.dumps({**summary, "steps": [{"t_wall": e["t_wall"], "backend_t": e["ts_backend"],
                                                           **e["data"]} for e in steps]},
                                    indent=2, ensure_ascii=False, default=str))
    print(f"{summary['state']}: {summary['reason']}")
    for e in steps:
        print(f"  {e['t_wall']:6.2f} s  t={e['ts_backend']:6.1f}  {e['data']['node']:8s} {step_label(e['data'])}")
    print(f"wrote {OUT} and {TRACE_OUT}")


if __name__ == "__main__":
    main()
