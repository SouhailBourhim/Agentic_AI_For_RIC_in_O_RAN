"""Figures for M13, the verifier's offer after a rejection.

  uv run python -m eval.plot_m13

1. docs/learn/img/M13-sim.png: dev and the m13_test split (scored once each), per family: the first-rejected
   intent's delivered share and its met share against what was stated, and the guarantee already in force (F6),
   for reject and offer. Reads the committed docs/learn/data/M13-{dev,test}.json.
2. docs/learn/img/M13-f6.png: the dev F6 instance where the offer gains most: eMBB and best effort throughput over
   time against the stated and the offered targets, both arms. Reads the raw dev runs (eval/results/m13, git-ignored).
Colours: the project's two-label pair (grey = as before, accent = the new behaviour), with a different marker each.
"""

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from eval import m13
from eval.generator import m13_instances
from eval.metrics import load_run
from eval.plot_control import ACCENT, IMG, titled
from sim.plot import GRID, SURFACE, TEXT, TEXT_MUTED

DATA = Path("docs/learn/data")
LABELS = {"reject": "rejected (as before M13)", "offer": "offer taken"}
COLOURS = {"reject": "#b4b2ac", "offer": ACCENT}
MARKERS = {"reject": "D", "offer": "o"}
GROUPS = (("delivered", "f4", "F4b\ndelivered"), ("delivered", "f6", "F6\ndelivered"),
          ("delivered", "all", "all\ndelivered"), ("met", "all", "all\nmet as stated"),
          ("in_force_met", None, "F6 guarantee\nin force, met"))


def plot_sim() -> Figure:
    fig = Figure(figsize=(12.5, 3.9), facecolor=SURFACE, layout="constrained")
    for ax, study in zip(fig.subplots(1, 2, sharey=True), ("dev", "test"), strict=True):
        result = json.loads((DATA / f"M13-{study}.json").read_text())
        s = result["summary"]
        x = np.arange(len(GROUPS))
        for k, label in enumerate(LABELS):
            values = [s[label][key] if family is None else s[label][key][family] for key, family, _ in GROUPS]
            ax.scatter(x + (k - 0.5) * 0.24, values, s=40, color=COLOURS[label], marker=MARKERS[label], zorder=3,
                       label=LABELS[label], edgecolors=SURFACE, linewidths=1.2)
        ax.set_xticks(x, [g[2] for g in GROUPS], fontsize=8)
        ax.set_ylim(0, 1.05)
        v = result["verdict"]
        name = "dev" if study == "dev" else "m13_test (fresh, once)"
        titled(ax, f"{name}: G {v['G_gain']:+.3f} (rule ≥ +0.05), H {v['H_in_force_change']:+.3f} (rule ≥ −0.02)")
    fig.axes[0].set_ylabel("mean share of the window", color=TEXT_MUTED, fontsize=8.5)
    fig.axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    return fig


def most_improved_f6() -> str:
    rows = [r for r in csv.DictReader((DATA / "M13-dev-runs.csv").open()) if r["family"] == "f6"]
    gain = {}
    for r in rows:
        gain[r["instance"]] = gain.get(r["instance"], 0.0) + (1 if r["label"] == "offer" else -1) * float(r["delivered"])
    return max(gain, key=gain.get)


def plot_f6(instance: str | None = None) -> Figure:
    instance = instance or most_improved_f6()
    inst = next(i for i in m13_instances() if i.name == instance)
    first, second = inst.scenario().intents
    start, end = second.window
    offered = json.loads(next(r["offer"] for r in csv.DictReader((DATA / "M13-dev-runs.csv").open())
                              if r["instance"] == instance and r["label"] == "offer"))
    fig = Figure(figsize=(12.5, 5.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(2, 1, sharex=True)
    for ax, s in zip(axes, ("embb", "be"), strict=True):
        for label in LABELS:
            kpis, _ = load_run(m13.work_of(inst, label, "dev"))
            k = kpis[s]
            if label == "reject":
                ax.fill_between(k["t"], 0, k["load"], color=GRID, alpha=0.6, linewidth=0, zorder=0,
                                label="offered load")
            ax.plot(k["t"], k["thr"], color=COLOURS[label], linewidth=1.6 if label == "offer" else 1.2,
                    label=LABELS[label], zorder=3 if label == "offer" else 2)
        ax.hlines(second.targets[s], start, end, color=TEXT, linewidth=1.0, linestyles=(0, (4, 3)),
                  label=f"stated {second.targets[s]:g} Mb/s")
        ax.hlines(offered[s], start, end, color=ACCENT, linewidth=1.0, linestyles=(0, (1, 2)),
                  label=f"offered {offered[s]:g} Mb/s")
        ax.axvspan(start, end, color=GRID, alpha=0.15, zorder=0)
        ax.set_ylabel("Mb/s", color=TEXT_MUTED, fontsize=8.5)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncols=5)
    lead = (f"dev {instance}: URLLC {first.targets['urllc']:g} Mb/s under {first.max_latency_ms:g} ms in force from "
            f"{first.at_s:g} s; eMBB {second.targets['embb']:g} + BE {second.targets['be']:g} Mb/s asked at "
            f"{start:g} s, rejected, offer {offered['embb']:g} + {offered['be']:g}")
    titled(axes[0], f"{lead}. eMBB throughput")
    titled(axes[1], "best effort throughput")
    axes[1].set_xlabel("time (s)", color=TEXT_MUTED, fontsize=8.5)
    return fig


TESTBED_ARMS = {"nooffer": "reject", "offer": "offer"}  # testbed arm -> the colour/marker/label key above


def plot_testbed() -> Figure:
    """Left: S3 intent 2 delivered and intent 1 met, one marker per repeat (committed score).
    Right: throughput per slice over time in the median repeat, both arms, with the monitor's fallback marked."""
    result = json.loads((DATA / "M13-testbed.json").read_text())
    repeats = [k for k in result if k.startswith("repeat")]
    median_repeat = sorted(repeats, key=lambda r: result[r]["offer"]["met_1"] - result[r]["nooffer"]["met_1"])[1]
    fig = Figure(figsize=(12.5, 6.2), facecolor=SURFACE, layout="constrained")
    axes = fig.subplot_mosaic([["sum", "urllc"], ["sum", "embb"], ["sum", "be"]], width_ratios=(1, 2.4))
    ax = axes["sum"]
    measures = (("delivered_2", "intent 2 (URLLC)\ndelivered"), ("met_1", "intent 1 (eMBB, BE)\nmet"))
    for k, (arm, key) in enumerate(TESTBED_ARMS.items()):
        for m, (measure, _) in enumerate(measures):
            values = [result[r][arm][measure] for r in repeats]
            ax.scatter(m + (k - 0.5) * 0.3 + np.linspace(-0.04, 0.04, len(values)), values, s=40, zorder=3,
                       color=COLOURS[key], marker=MARKERS[key], edgecolors=SURFACE, linewidths=1.2,
                       label=LABELS[key] if m == 0 else None)
    ax.set_xticks(range(len(measures)), [m[1] for m in measures], fontsize=8)
    ax.set_xlim(-0.6, len(measures) - 0.4)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("share of the window, 3 repeats", color=TEXT_MUTED, fontsize=8.5)
    v = result["verdict"]
    titled(ax, f"medians: gain {v['median_gain']:+.2f}, intent 1 {v['median_harm']:+.2f} (rule ≥ −0.05)")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    repeat = int(median_repeat.removeprefix("repeat"))
    from eval import m13_testbed  # here, not at the top: it pulls in the testbed control code
    runs = {arm: load_run(m13_testbed.RESULTS / m13_testbed.run_of(arm, repeat).run_id) for arm in TESTBED_ARMS}
    names = {"urllc": "URLLC", "embb": "eMBB", "be": "best effort"}
    targets = {"urllc": "asked 23 under 5 ms, offer 16", "embb": "intent 1: 23", "be": "intent 1: 23"}
    for s in names:
        ax = axes[s]
        for arm, key in TESTBED_ARMS.items():
            k = runs[arm][0][s]
            if arm == "nooffer":
                ax.fill_between(k["t"], 0, k["load"], color=GRID, alpha=0.6, linewidth=0, zorder=0,
                                label="offered load")
            ax.plot(k["t"], k["thr"], color=COLOURS[key], linewidth=1.6 if arm == "offer" else 1.2,
                    label=LABELS[key], zorder=3 if arm == "offer" else 2)
        for e in runs["offer"][1]:
            if e["event"] == "executor_fallback":
                ax.axvspan(e["ts_backend"], 360, color=ACCENT, alpha=0.08, zorder=0,
                           label="offer arm: monitor's fallback\n(URLLC latency)")
        ax.hlines(23 if s != "urllc" else 16, 60 if s != "urllc" else 120, 360, color=TEXT, linewidth=1.0,
                  linestyles=(0, (4, 3)), label=f"target ({targets[s]})")
        ax.set_ylabel("Mb/s", color=TEXT_MUTED, fontsize=8.5)
        titled(ax, f"testbed S3, repeat {repeat} (median). {names[s]} throughput" if s == "urllc"
               else f"{names[s]} throughput")
    axes["embb"].legend(*axes["embb"].get_legend_handles_labels(), frameon=False, fontsize=7, loc="upper right",
                        ncols=1)
    axes["be"].set_xlabel("time (s)", color=TEXT_MUTED, fontsize=8.5)
    return fig


def main() -> None:
    for name, fig in (("M13-sim.png", plot_sim()), ("M13-f6.png", plot_f6()), ("M13-testbed.png", plot_testbed())):
        fig.savefig(IMG / name, dpi=150, facecolor=SURFACE)
        print(IMG / name)


if __name__ == "__main__":
    main()
