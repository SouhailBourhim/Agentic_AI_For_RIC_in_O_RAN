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


def main() -> None:
    for name, fig in (("M13-sim.png", plot_sim()), ("M13-f6.png", plot_f6())):
        fig.savefig(IMG / name, dpi=150, facecolor=SURFACE)
        print(IMG / name)


if __name__ == "__main__":
    main()
