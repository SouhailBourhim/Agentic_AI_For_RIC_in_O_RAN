"""Figure for M11a: the runner's safe state, on a simulated cell with three injected faults.

Run with:  uv run python -m rl.plot_safe_state      (a few seconds; writes docs/learn/img/M11a-safe-state.png)
           uv run python -m rl.plot_safe_state --policy-file path/to/policy.zip   (another policy)

The served policy (var/policy_cache, as the runner serves it: alpha 0.10) runs the `demo` scenario (URLLC surge
from t=120 s) on backends.faults.FaultyBackend, which reports:
- stale KPIs over 60-89 s (what FlexRICBackend.health() says when KPM reports stop),
- a store that can't be read over 150-152 s (read_kpis raises: reason 'error'),
- a dead actuator over 200-219 s (the slice-control xApp is down).
Each puts the equal split on the cell at once; three clean ticks later the policy decides again, its filter
restarted from the equal split. A temporary database is used; var/ric.db is not touched.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from backends.faults import FaultyBackend
from rl.policy import SlicePolicy
from rl.runner import SERVED_SMOOTHING_ALPHA, Runner
from sim.plot import SLICE_COLOURS, SLICE_LABELS, SURFACE, TEXT, TEXT_MUTED, _style
from sim.scenarios import DEMO_SURGE_START_S
from store import db, kpi
from verifier.objective import SLICES

RUN_S = 300
STALE = range(60, 90)
READ_ERROR = range(150, 153)
DOWN = range(200, 220)
SAFE_TINT = "#fbe3dc"
OUT = Path("docs/learn/img/M11a-safe-state.png")


def served_policy() -> tuple[Path, str]:
    """The cached champion and its label, without asking MLflow (rl.runner.load_policy would)."""
    cache = Path("var/policy_cache")
    served = json.loads((cache / "served.json").read_text())
    return cache / served["file"], served["label"]


def simulate(policy_file: Path, label: str) -> dict:
    """One faulty run. Returns per tick the time the split was applied, the split, the mode and the reason,
    and per slice the KPI rows (time, throughput, offered load)."""
    work = Path(tempfile.mkdtemp(prefix="m11a_"))
    os.environ["RIC_EVENTS"] = str(work / "events.jsonl")
    conn = db.connect(work / "ric.db")
    backend = FaultyBackend("demo", stale=STALE, down=DOWN, read_error=READ_ERROR)
    kpi.write_backend_info(conn, backend.info(), label)
    runner = Runner(backend, SlicePolicy.load(policy_file, SERVED_SMOOTHING_ALPHA), db.connect(work / "ric.db", 0.0))
    ticks = []
    for _ in range(RUN_S):
        applied_at = backend.now()
        result = runner.tick()
        ticks.append((applied_at, result.shares, result.mode, runner.reason))
    rows = conn.execute("SELECT ts, slice, dl_thr_mbps, offered_load_mbps FROM kpi ORDER BY ts").fetchall()
    kpis = {s: np.array([[r["ts"], r["dl_thr_mbps"], r["offered_load_mbps"]] for r in rows if r["slice"] == s])
            for s in SLICES}
    return {"ticks": ticks, "kpis": kpis}


def safe_spans(ticks: list) -> list[tuple[float, float, str]]:
    """(start, end, reasons) of each run of safe ticks; a reason that changes inside one run is listed in order."""
    spans: list[tuple[float, float, str]] = []
    start, reasons = None, []
    for t, _, mode, reason in ticks + [(ticks[-1][0] + 1, None, "normal", None)]:
        if mode == "safe":
            if start is None:
                start, reasons = t, []
            if reason not in reasons:
                reasons.append(reason)
        elif start is not None:
            spans.append((start, t, " then ".join(reasons)))
            start = None
    return spans


def plot(run: dict, label: str) -> Figure:
    ticks, kpis = run["ticks"], run["kpis"]
    t = np.array([x[0] for x in ticks])
    shares = np.array([x[1] for x in ticks])
    fig = Figure(figsize=(12, 7), facecolor=SURFACE, layout="constrained")
    ax_share, ax_thr = fig.subplots(2, 1, sharex=True)
    for i, s in enumerate(SLICES):
        ax_share.step(t, shares[:, i], where="post", color=SLICE_COLOURS[s], linewidth=1.6, label=SLICE_LABELS[s])
        ax_thr.plot(kpis[s][:, 0], kpis[s][:, 1], color=SLICE_COLOURS[s], linewidth=1.6)
        ax_thr.plot(kpis[s][:, 0], kpis[s][:, 2], color=SLICE_COLOURS[s], linewidth=1.0, linestyle=(0, (4, 3)))
    ax_share.axhline(1 / 3, color=TEXT_MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
    ax_share.annotate("safe split 1/3", (RUN_S, 1 / 3), xytext=(-4, 4), textcoords="offset points", ha="right",
                      fontsize=8, color=TEXT_MUTED)
    ax_share.set_ylim(0, 1)
    for start, end, reasons in safe_spans(ticks):
        for ax in (ax_share, ax_thr):
            ax.axvspan(start, end, color=SAFE_TINT, zorder=0)
        ax_share.text((start + end) / 2, 0.98, f"safe: {reasons}\n{start:g}-{end:g} s", ha="center", va="top",
                      fontsize=8, color=TEXT)
    for ax in (ax_share, ax_thr):
        ax.axvline(DEMO_SURGE_START_S, color=TEXT_MUTED, linewidth=0.8)
    ax_thr.annotate("URLLC surge starts", (DEMO_SURGE_START_S, ax_thr.get_ylim()[1]), xytext=(4, -10),
                    textcoords="offset points", fontsize=8, color=TEXT_MUTED)
    _style(ax_share, "PRB share applied")
    _style(ax_thr, "Mb/s (dashed: offered load)")
    ax_thr.set_xlabel("backend clock (s)", color=TEXT_MUTED, fontsize=9)
    handles, labels = ax_share.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=9)
    fig.suptitle(f"Runner safe state under injected faults (policy {label}): equal split at once, policy back "
                 f"after 3 clean ticks", color=TEXT, fontsize=10.5)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy-file", type=Path, default=None, help="default: the served champion (cache)")
    args = parser.parse_args()
    policy_file, label = (args.policy_file, args.policy_file.name) if args.policy_file else served_policy()
    run = simulate(policy_file, label)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plot(run, label).savefig(OUT, dpi=110)
    print(f"wrote {OUT}")
    for start, end, reasons in safe_spans(run["ticks"]):
        print(f"safe {start:5.0f}-{end:5.0f} s  ({end - start:3.0f} s)  {reasons}")
    print(f"safe ticks: {sum(x[2] == 'safe' for x in run['ticks'])} of {RUN_S}")


if __name__ == "__main__":
    main()
