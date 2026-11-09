"""Figures for docs/learn/M07b-slice-control.md, from the files in testbed/results/m07b/.

  uv run python testbed/plot_m07b.py

1. M07b-cap.png: each slice's received throughput against the PRBs it was given over E2, with every
   slice overloaded (testbed/measure_m07b.py cap), next to the line the simulator assumes.
2. M07b-isolation.png: a light URLLC next to an overloaded BE, stock scheduler vs capped.
3. M07b-loop.png: the surge scenario on the real cell, one column per run (stock scheduler, RL
   xApp, RL xApp without the offered rates): split, throughput against offered, RLC delay.
4. M07b-intent.png: an operator intent through gateway, agent, verifier and console, applied by
   the RL xApp on the real gNB during the surge (run loop_intent2).
The delay test on the simulator draws its own figure (testbed/sim_delay_m07b.py).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent))  # testbed/, for plot_spike
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from plot_spike import load_run  # noqa: E402
from sim.cell import CellConfig  # noqa: E402
from sim.plot import LINE_WIDTH, SLICE_COLOURS, SLICE_LABELS, SURFACE, TEXT, TEXT_MUTED, _style  # noqa: E402

RESULTS = Path("testbed/results/m07b")
IMG = Path("docs/learn/img")
SLICES = ("embb", "urllc", "be")
N_PRB = 106
CAP_RUNS = ["equal", "s60_30_10", "s10_80_10", "s05_05_90", "s20_20_60", "s80_15_05"]
LOOP_RUNS = [("loop_stock", "Stock scheduler (no control)"), ("loop_rl", "RL xApp, policy v7"),
             ("loop_blind", "RL xApp, offered rates hidden")]


def controls(mode: str) -> dict[str, list[int] | None]:
    """run label -> PRBs per slice sent for it (None = stock scheduler)."""
    out = {}
    for line in (RESULTS / f"{mode}_controls.jsonl").read_text().splitlines():
        c = json.loads(line)
        out[c["run"]] = c["ack"].get("prbs")
    return out


def plot_cap() -> Figure:
    prbs = controls("cap")
    fig = Figure(figsize=(8.5, 5.2), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    xs, ys = [], []
    for run in CAP_RUNS:
        for i, s in enumerate(SLICES):
            got = load_run(RESULTS / f"{run}_{s}.json")["steady_mbps"]
            ax.scatter(prbs[run][i], got, s=46, color=SLICE_COLOURS[s], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=SLICE_LABELS[s] if run == CAP_RUNS[0] else None)
            xs.append(prbs[run][i])
            ys.append(got)
    x = np.linspace(0, N_PRB, 50)
    sim_capacity = CellConfig().capacity_mbps
    ax.plot(x, x / N_PRB * sim_capacity, color=TEXT_MUTED, linestyle="--", linewidth=LINE_WIDTH,
            label=f"simulator: share × {sim_capacity:g} Mb/s")
    slope = float(np.dot(xs, ys) / np.dot(xs, xs))  # least squares through the origin
    ax.plot(x, slope * x, color=TEXT, linewidth=1, label=f"fit: {slope:.3f} Mb/s per PRB (= {slope * N_PRB:.0f} Mb/s cell)")
    stock = [load_run(RESULTS / f"stock_{s}.json")["steady_mbps"] for s in SLICES]
    ax.annotate(f"stock scheduler, same overload: {stock[0]:.1f} / {stock[1]:.1f} / {stock[2]:.1f} Mb/s",
                (0.02, 0.96), xycoords="axes fraction", va="top", fontsize=8.5, color=TEXT_MUTED)
    ax.set_xlim(0, N_PRB + 2)
    ax.set_ylim(0, None)
    ax.set_xlabel("PRBs given to the slice over E2 (of 106)", color=TEXT)
    _style(ax, "received throughput, Mb/s (iperf3, steady)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Every slice overloaded at 80 Mb/s: throughput follows the PRBs given", color=TEXT, fontsize=11, loc="left")
    return fig


def ping_p95(path: Path, after_s: float = 15.0) -> float:
    import re

    rtts = [float(rtt) for seq, rtt in re.findall(r"icmp_seq=(\d+) .*time=([\d.]+) ms", path.read_text())
            if int(seq) * 0.2 >= after_s]
    return float(np.percentile(rtts, 95)) if rtts else float("nan")


def plot_isolation() -> Figure:
    runs = [("iso_stock", "stock scheduler"), ("iso_capped", "capped: eMBB 53, URLLC 27, BE 26 PRBs")]
    fig = Figure(figsize=(10, 4.2), facecolor=SURFACE, layout="constrained")
    ax_thr, ax_rtt = fig.subplots(1, 2, width_ratios=[1.6, 1])
    width = 0.36
    for k, (run, _label) in enumerate(runs):
        vals = [load_run(RESULTS / f"{run}_{s}.json")["steady_mbps"] for s in ("urllc", "be")]
        pos = np.arange(2) + (k - 0.5) * width
        bars = ax_thr.bar(pos, vals, width * 0.92, color=[SLICE_COLOURS["urllc"], SLICE_COLOURS["be"]],
                          alpha=1.0 if k == 0 else 0.55)
        for b, v in zip(bars, vals):
            ax_thr.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8.5, color=TEXT)
        p95 = ping_p95(RESULTS / f"{run}_ping_urllc.txt")
        ax_rtt.bar(k, p95, 0.6, color=SLICE_COLOURS["urllc"], alpha=1.0 if k == 0 else 0.55)
        ax_rtt.annotate(f"{p95:.0f} ms", (k, p95), ha="center", va="bottom", fontsize=8.5, color=TEXT)
    ax_thr.set_xticks([0, 1], ["URLLC (sends 10 Mb/s)", "BE (sends 80 Mb/s)"])
    _style(ax_thr, "received, Mb/s")
    ax_thr.annotate(f"left, solid: {runs[0][1]}\nright, faded: {runs[1][1]}", (0.98, 0.97), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8.5, color=TEXT_MUTED)
    ax_rtt.set_xticks([0, 1], ["stock", "capped"])
    _style(ax_rtt, "URLLC ping RTT p95, ms (steady)")
    fig.suptitle("A light URLLC was already safe on the stock scheduler; the hard cap costs BE the idle eMBB PRBs",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def load_kpi(run: str) -> dict[str, dict[str, np.ndarray]]:
    rows = list(csv.DictReader((RESULTS / f"{run}_kpi.csv").open()))
    t0 = min(float(r["ts"]) for r in rows)
    out = {}
    for s in SLICES:
        mine = [r for r in rows if r["slice"] == s]
        num = lambda k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in mine])  # noqa: B023 (used within this iteration)
        out[s] = {"t": num("ts") - t0, "thr": num("dl_thr_mbps"), "alloc": num("prb_share_alloc"),
                  "used": num("prb_used"), "lat": num("latency_ms"), "load": num("offered_load_mbps")}
    return out


def segment_times(run: str, t0: float) -> list[tuple[float, dict, float]]:
    """(start, rates, end of traffic) per segment, in s from t0. A segment's traffic lasts its
    planned seconds; the next one starts only when every iperf3 run has closed, which waits for
    the slices' queues to drain (testbed/scenario_m07b.py), so there is a gap with no new traffic."""
    events = [json.loads(line) for line in (RESULTS / f"{run}_events.jsonl").read_text().splitlines()]
    return [(e["data"]["wall"] - t0, e["data"]["rates_mbps"], e["data"]["wall"] - t0 + e["data"]["seconds"])
            for e in events if e["event"] == "segment_start"]


def plot_loop() -> Figure:
    runs = [(r, title) for r, title in LOOP_RUNS if (RESULTS / f"{r}_kpi.csv").exists()]
    fig = Figure(figsize=(14, 8.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(3, len(runs), sharex="col", sharey="row", squeeze=False)
    for c, (run, title) in enumerate(runs):
        kpi = load_kpi(run)
        t0 = min(float(r["ts"]) for r in csv.DictReader((RESULTS / f"{run}_kpi.csv").open()))
        segs = segment_times(run, t0)
        for s in SLICES:
            k = kpi[s]
            axes[0][c].plot(k["t"], k["alloc"], color=SLICE_COLOURS[s], linewidth=LINE_WIDTH, label=SLICE_LABELS[s])
            axes[1][c].plot(k["t"], k["thr"], color=SLICE_COLOURS[s], linewidth=LINE_WIDTH)
            axes[2][c].plot(k["t"], k["lat"], color=SLICE_COLOURS[s], linewidth=LINE_WIDTH)
            # what each slice is sent, as steps from the scenario's own log
            for start, rates, end in segs:
                axes[1][c].plot([start, end], [rates[s], rates[s]], color=SLICE_COLOURS[s], linestyle=":", linewidth=1.2)
        for row in axes[:, c]:
            for i, (start, _, end) in enumerate(segs):
                if i > 0:
                    row.axvline(start, color=TEXT_MUTED, linewidth=0.8, linestyle="--")
                gap_end = segs[i + 1][0] if i + 1 < len(segs) else end
                if gap_end > end:
                    row.axvspan(end, gap_end, color=TEXT_MUTED, alpha=0.10, linewidth=0)
        axes[0][c].set_title(title, color=TEXT, fontsize=10, loc="left")
        axes[2][c].set_xlabel("time, s", color=TEXT)
        axes[2][c].set_yscale("symlog", linthresh=10)
        axes[2][c].set_ylim(0, None)
    _style(axes[0][0], "PRB share (1.0 = whole cell,\nstock: no cap)")
    _style(axes[1][0], "throughput, Mb/s\n(dotted: sent)")
    _style(axes[2][0], "RLC queueing delay, ms\n(KPM, log scale)")
    for row in axes:
        for ax in row[1:]:
            _style(ax, "")
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="center left")
    fig.suptitle("URLLC surge (7.5 → 30 Mb/s) on the real cell. Dashed: next segment starts; grey: no new traffic, "
                 "queues draining", color=TEXT,
                 fontsize=11, x=0.01, ha="left")
    return fig


def plot_intent(run: str = "loop_intent2") -> Figure:
    kpi = load_kpi(run)
    t0 = min(float(r["ts"]) for r in csv.DictReader((RESULTS / f"{run}_kpi.csv").open()))
    events = [json.loads(line) for line in (RESULTS / f"{run}_events.jsonl").read_text().splitlines()]
    switches = [(e["ts_wall"] - t0, e["data"]["to"]) for e in events if e["event"] == "policy_switch"]
    segs = segment_times(run, t0)
    fig = Figure(figsize=(11, 7.4), facecolor=SURFACE, layout="constrained")
    ax_share, ax_thr, ax_lat = fig.subplots(3, 1, sharex=True)
    for s in SLICES:
        k = kpi[s]
        ax_share.plot(k["t"], k["alloc"], color=SLICE_COLOURS[s], linewidth=LINE_WIDTH, label=SLICE_LABELS[s])
        ax_thr.plot(k["t"], k["thr"], color=SLICE_COLOURS[s], linewidth=LINE_WIDTH)
        ax_lat.plot(k["t"], k["lat"], color=SLICE_COLOURS[s], linewidth=LINE_WIDTH)
    ax_thr.axhline(20, color=SLICE_COLOURS["urllc"], linestyle=":", linewidth=1.2)
    ax_thr.annotate("intent: URLLC ≥ 20 Mb/s", (switches[0][0] + 3, 21), fontsize=8.5, color=TEXT_MUTED)
    for ax in (ax_share, ax_thr, ax_lat):
        (on, _), (off, _) = switches[0], switches[1]
        ax.axvspan(on, off, color=SLICE_COLOURS["urllc"], alpha=0.07, linewidth=0)
        ax.axvspan(segs[1][0], segs[1][2], ymin=0.97, ymax=1.0, color=TEXT_MUTED, alpha=0.6, linewidth=0)
    ax_share.annotate("grey bar: URLLC surge (sent 30 Mb/s)", (segs[1][0], 1.06), fontsize=8.5, color=TEXT_MUTED,
                      annotation_clip=False)
    ax_share.set_ylim(0, 1.0)
    ax_share.annotate("objective of the intent in force (tinted)", (switches[0][0] + 3, 0.92), fontsize=8.5,
                      color=TEXT_MUTED)
    ax_lat.set_yscale("symlog", linthresh=10)
    ax_lat.set_ylim(0, None)
    _style(ax_share, "PRB share")
    _style(ax_thr, "throughput, Mb/s (KPM)")
    _style(ax_lat, "RLC queueing delay, ms")
    ax_lat.set_xlabel("time, s", color=TEXT)
    ax_share.legend(frameon=False, fontsize=8.5, ncols=3, loc="upper right")
    fig.suptitle("“Garantis 20 Mb/s au slice URLLC pendant 5 minutes.” on the real cell: confirmed in the console "
                 f"at {switches[0][0]:.0f} s, applied by the RL xApp the next tick", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    for name, plot in (("M07b-cap.png", plot_cap), ("M07b-isolation.png", plot_isolation), ("M07b-loop.png", plot_loop),
                       ("M07b-intent.png", plot_intent)):
        plot().savefig(IMG / name, dpi=120)
        print(f"wrote {IMG / name}")


if __name__ == "__main__":
    main()
