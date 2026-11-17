"""The figures of M10d, the demand the gNB measures (run `eval.m10d --calibration` and `--score` first).

  uv run python -m eval.plot_m10d

Writes, from committed data only:
- docs/learn/img/M10d-calibration.png, from testbed/results/m10d/calib_{runs,kpm}.jsonl: per slice, what iperf3
  sent (grey), what PDCP counted (solid) and what the gNB served (dotted), over the calibration run. The overload
  phase is the point: PDCP reads the 30 Mb/s that arrive while URLLC is served about 4.
- docs/learn/img/M10d-s1.png, from docs/learn/data/M10d-s1-runs.csv: S1 intent met and reaction time per executor
  and demand source, one dot per repeat, a bar at the median.
- docs/learn/img/M10d-s1-demand.png, from docs/learn/data/M10d-s1-measured-demand.csv: one S1 run (LLM + formula on
  the offered rate, repeat 3, where the measure was recorded but not used): the three demand sources side by side,
  with what iperf3 really sent: on a slice with a long queue, a new segment's session starts only once its handshake
  has crossed that queue (eMBB 16 s late at 120 s), so the segment's rate (the oracle) is wrong for those seconds
  and PDCP, reading 0, is right.
"""

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from eval.m10d import CALIB_DIR, CELLS, TABLE, reports_by_slice
from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from rl.runner import DEMANDS
from sim.plot import SLICE_COLOURS, SURFACE, TEXT, TEXT_MUTED

DEMAND_CSV = Path("docs/learn/data/M10d-s1-measured-demand.csv")
SLICE_LABELS = {"embb": "eMBB", "urllc": "URLLC", "be": "best effort"}
EXECUTOR_LABELS = {"rl": "LLM + RL", "formula": "LLM + formula"}
EXECUTOR_COLOURS = {"rl": ACCENT, "formula": NEUTRAL}
DEMAND_LABELS = {"offered": "offered\n(iperf3, oracle)", "measured": "measured\n(PDCP volume)",
                 "throughput": "throughput\n(blind)"}
SURGE_S = (120.0, 300.0)  # S1's URLLC surge (eval/testbed_scenarios.SEGMENTS)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def target_line(runs: list[dict], slice_: str, t0: float, end: float) -> tuple[list[float], list[float]]:
    """The iperf3 rate as a step line: the run's rate over its sending window, 0 elsewhere."""
    xs, ys = [0.0], [0.0]
    for run in runs:
        rate = run["rates_mbps"].get(slice_, 0.0)
        start = run["start_wall"] - t0
        xs += [start, start, start + run["seconds"], start + run["seconds"]]
        ys += [0.0, rate, rate, 0.0]
    return xs + [end], ys + [0.0]


def plot_calibration(runs: list[dict], lines: list[str]) -> Figure:
    reports = reports_by_slice(lines)
    t0 = runs[0]["start_wall"] - 5
    end = max(r[0] for rs in reports.values() for r in rs) - t0
    fig = Figure(figsize=(11, 6.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(3, 1, sharex=True)
    for ax, s in zip(axes, ("embb", "urllc", "be")):
        mine = [r for r in reports.get(s, []) if r[0] >= t0]
        t = [r[0] - t0 + 0.5 for r in mine]  # the middle of each 1 s period
        xs, ys = target_line(runs, s, t0, end)
        ax.plot(xs, ys, color=TEXT_MUTED, linewidth=1.2, label="iperf3 rate (sent)")
        ax.plot(t, [r[1] if r[1] is not None else np.nan for r in mine], color=SLICE_COLOURS[s], linewidth=2,
                label="PDCP SDU volume (arrived at the gNB)")
        ax.plot(t, [r[2] if r[2] is not None else np.nan for r in mine], color=SLICE_COLOURS[s], linewidth=1.2,
                linestyle=":", label="KPM throughput (served)")
        titled(ax, SLICE_LABELS[s])
        ax.set_ylabel("Mb/s", color=TEXT_MUTED, fontsize=8.5)
        ax.set_xlim(0, end)
        ax.set_ylim(bottom=0)
    for run in runs:
        start = run["start_wall"] - t0
        for ax in axes:
            ax.axvline(start, color=TEXT_MUTED, linewidth=0.6, linestyle=":")
        label = {"steady": "steady 30 / 10 / 5", "overload": "overload: URLLC 30 at 5 PRBs"}.get(
            run["phase"], f"{run['rates_mbps']['urllc']:g}")
        axes[0].text(start + 1, axes[0].get_ylim()[1] * 0.97, label, fontsize=7.5, color=TEXT, va="top")
    axes[-1].set_xlabel("seconds since the calibration started", color=TEXT_MUTED, fontsize=8.5)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=8.5, labelcolor=TEXT)
    fig.suptitle("M10d calibration: PDCP counts what arrives, not what is served (overload: 30 Mb/s read, ~4 served)",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def plot_s1(rows: list[dict]) -> Figure:
    fig = Figure(figsize=(11, 3.8), facecolor=SURFACE, layout="constrained")
    met_ax, react_ax = fig.subplots(1, 2)
    for ax, key, scale in ((met_ax, "intent_met", 100.0), (react_ax, "reaction_s", 1.0)):
        for x, demand in enumerate(DEMANDS):
            for k, executor in enumerate(("rl", "formula")):
                xpos = x + (k - 0.5) * 0.3
                values = [scale * float(r[key]) for r in rows if r["executor"] == executor and r["demand"] == demand
                          and r["valid"] == "True" and r["invalid_copy"] == "False" and r[key] != ""]
                ax.scatter([xpos] * len(values), values, s=36, color=EXECUTOR_COLOURS[executor], alpha=0.85,
                           edgecolors=SURFACE, linewidths=1.2, zorder=3,
                           label=EXECUTOR_LABELS[executor] if x == 0 else None)
                if values:
                    ax.plot([xpos - 0.1, xpos + 0.1], [np.median(values)] * 2, color=EXECUTOR_COLOURS[executor],
                            linewidth=2.5, zorder=2)
        ax.set_xticks(range(len(DEMANDS)), [DEMAND_LABELS[d] for d in DEMANDS], fontsize=8.5)
        ax.set_xlim(-0.6, len(DEMANDS) - 0.4)
    titled(met_ax, "S1 intent met (URLLC >= 21 Mb/s, < 10 ms), 3 repeats, bar = median")
    met_ax.set_ylabel("intent met, % of the window", color=TEXT_MUTED, fontsize=8.5)
    titled(react_ax, "Reaction time after the surge")
    react_ax.set_ylabel("seconds", color=TEXT_MUTED, fontsize=8.5)
    react_ax.set_ylim(bottom=0)
    met_ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=TEXT)
    fig.suptitle("M10d S1: on the measured demand, the goal is met as on the oracle; the executors react 2-3 s later",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def plot_s1_demand(rows: list[dict]) -> Figure:
    fig = Figure(figsize=(11, 6.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(3, 1, sharex=True)
    for ax, s in zip(axes, ("embb", "urllc", "be")):
        mine = [r for r in rows if r["slice"] == s]
        t = [float(r["t_s"]) for r in mine]
        num = [float(r["measured_load_mbps"]) if r["measured_load_mbps"] not in ("", "None") else np.nan for r in mine]
        ax.axvspan(*SURGE_S, color=SLICE_COLOURS["urllc"], alpha=0.06, linewidth=0)
        ax.plot(t, [float(r["sent_mbps"]) for r in mine], color=TEXT_MUTED, linewidth=1.4, drawstyle="steps-post",
                label="sent (iperf3 sessions as they ran)")
        ax.plot(t, [float(r["offered_load_mbps"]) for r in mine], color=TEXT_MUTED, linewidth=1, linestyle="--",
                drawstyle="steps-post", label="offered (the segment's rate: the oracle)")
        ax.plot(t, num, color=SLICE_COLOURS[s], linewidth=2, label="measured (PDCP SDU volume)")
        ax.plot(t, [float(r["dl_thr_mbps"]) for r in mine], color=SLICE_COLOURS[s], linewidth=1.2, linestyle=":",
                label="throughput (served)")
        zeros = sum(1 for v, tt in zip(num, t) if v < 1 and tt > 5)
        titled(ax, f"{SLICE_LABELS[s]}: {zeros} zero readings after the first 5 s")
        ax.set_ylabel("Mb/s", color=TEXT_MUTED, fontsize=8.5)
        ax.set_xlim(0, 360)
        ax.set_ylim(bottom=0)
    axes[0].text(SURGE_S[0] + 2, axes[0].get_ylim()[1] * 0.97, "URLLC surge (8 -> 31.5 Mb/s)", fontsize=7.5,
                 color=TEXT, va="top")
    axes[-1].set_xlabel("seconds into the S1 run", color=TEXT_MUTED, fontsize=8.5)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False, fontsize=8.5, labelcolor=TEXT)
    fig.suptitle("M10d S1, one run (LLM + formula on the offered rate, repeat 3): PDCP follows what iperf3 really sent; the oracle doesn't",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(line) for line in (CALIB_DIR / "calib_runs.jsonl").read_text().splitlines()]
    kpm = (CALIB_DIR / "calib_kpm.jsonl").read_text().splitlines()
    assert {(e, d) for e, d in CELLS} == {(r["executor"], r["demand"]) for r in read_csv(TABLE)}, "score S1 first"
    figures = (("M10d-calibration.png", plot_calibration(runs, kpm)), ("M10d-s1.png", plot_s1(read_csv(TABLE))),
               ("M10d-s1-demand.png", plot_s1_demand(read_csv(DEMAND_CSV))))
    for name, fig in figures:
        fig.savefig(IMG / name, dpi=150)
        print(IMG / name)


if __name__ == "__main__":
    main()
