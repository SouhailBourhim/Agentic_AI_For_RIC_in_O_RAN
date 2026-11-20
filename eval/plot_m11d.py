"""Figures for M11d, the shielded RL executor.

  uv run python -m eval.plot_m11d

1. docs/learn/img/M11d-dev.png: mean intent met per family on both cells, five labels (dev, scored once).
2. docs/learn/img/M11d-f3.png: the brief's extra deliverable, one m10 F3 dev run over time: URLLC's offered load and
   share of the cell, and its latency against the bound, one line per executor.
Colours: the dataviz reference palette's categorical slots (orange, violet, magenta) for the three new labels, greys
for the two references, and a different marker per label, so identity never rests on colour alone.
"""

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from eval import m11d
from eval.generator import dev_instances
from eval.m11b import cells
from eval.metrics import load_run
from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from sim.plot import GRID, SURFACE, TEXT, TEXT_MUTED

DATA = Path("docs/learn/data")
LABELS = {"rl": "RL alone", "shield": "shielded RL", "formula_est": "formula alone", "rl_mon": "RL + monitor (served)",
          "shield_mon": "shielded RL + monitor"}
COLOURS = {"rl": "#b4b2ac", "shield": "#eb6834", "formula_est": NEUTRAL, "rl_mon": ACCENT, "shield_mon": "#e87ba4"}
MARKERS = {"rl": "o", "shield": "D", "formula_est": "s", "rl_mon": "o", "shield_mon": "^"}
CELL_TITLES = {"m10": "m10 cell (radio channel), M10c′ policy", "m09": "m09 cell, v7 policy"}


def plot_dev() -> Figure:
    result = json.loads((DATA / "M11d-dev.json").read_text())
    fig = Figure(figsize=(12.5, 4.0), facecolor=SURFACE, layout="constrained")
    for ax, cell in zip(fig.subplots(1, 2, sharey=True), ("m10", "m09"), strict=True):
        means = result[cell]["means"]
        groups = list(means)
        x = np.arange(len(groups))
        for k, label in enumerate(LABELS):
            values = [means[g][label] for g in groups]
            ax.scatter(x + (k - 2) * 0.15, values, s=34, color=COLOURS[label], marker=MARKERS[label], zorder=3,
                       label=LABELS[label], edgecolors=SURFACE, linewidths=1.2)
        ax.set_xticks(x, [g.upper() if g != "all" else "all" for g in groups], fontsize=8.5)
        ax.set_ylim(0.5, 1.05)
        titled(ax, f"{CELL_TITLES[cell]} ({result[cell]['runs']} dev instances)")
    fig.axes[0].set_ylabel("mean intent met (accepted intents)", color=TEXT_MUTED, fontsize=8.5)
    fig.axes[0].legend(frameon=False, fontsize=8, loc="lower left", ncols=2)
    return fig


def most_improved_f3() -> str:
    """The m10 F3 dev instance where the shield gains most over the RL alone (read from the committed runs CSV)."""
    rows = [r for r in csv.DictReader((DATA / "M11d-dev-runs.csv").open())
            if r["cell"] == "m10" and r["family"] == "f3" and r["intent"] == "0" and r["accepted"] == "True"]
    met = {(r["instance"], r["label"]): float(r["intent_met"]) for r in rows}
    instances = sorted({r["instance"] for r in rows})
    return max(instances, key=lambda i: met[(i, "shield")] - met[(i, "rl")])


def plot_f3(instance: str | None = None) -> Figure:
    instance = instance or most_improved_f3()
    cell = cells()["m10"]
    inst = next(i for i in dev_instances(cell["families"]) if i.name == instance)
    intent = inst.scenario().intents[0]
    start, end = intent.window
    fig = Figure(figsize=(12.5, 6.0), facecolor=SURFACE, layout="constrained")
    share_ax, lat_ax = fig.subplots(2, 1, sharex=True)
    load_ax = share_ax.twinx()  # the load is drawn as context in its own unit; shares are what is compared
    for label in ("rl", "shield", "formula_est", "rl_mon"):
        kpis, _ = load_run(m11d.work_of("m10", cell, inst, label))
        u = kpis["urllc"]
        share_ax.plot(u["t"], u["share"], color=COLOURS[label], linewidth=1.6 if label == "shield" else 1.2,
                      label=LABELS[label], zorder=3 if label == "shield" else 2)
        lat_ax.plot(u["t"], np.maximum(u["lat"], 0.1), color=COLOURS[label],
                    linewidth=1.6 if label == "shield" else 1.2, zorder=3 if label == "shield" else 2)
        if label == "rl":
            load_ax.fill_between(u["t"], 0, u["load"], color=GRID, alpha=0.6, linewidth=0, zorder=0)
    load_ax.set_ylabel("URLLC offered load (Mb/s, shaded)", color=TEXT_MUTED, fontsize=8.5)
    load_ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for spine in load_ax.spines.values():
        spine.set_visible(False)
    share_ax.set_zorder(load_ax.get_zorder() + 1)
    share_ax.patch.set_visible(False)
    share_ax.set_ylim(0, 1)
    titled(share_ax, f"m10 {instance} (dev): \"{intent.text}\" URLLC share of the cell")
    share_ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=4)
    lat_ax.axhline(intent.max_latency_ms, color=TEXT, linewidth=1.0, linestyle=(0, (4, 3)))
    lat_ax.annotate(f"bound {intent.max_latency_ms:g} ms", (start, intent.max_latency_ms), xytext=(4, 4),
                    textcoords="offset points", fontsize=7.5, color=TEXT)
    lat_ax.set_yscale("log")
    titled(lat_ax, "URLLC latency (ms, log scale)")
    lat_ax.set_xlabel("time (s)", color=TEXT_MUTED, fontsize=8.5)
    for ax in (share_ax, lat_ax):
        ax.axvspan(start, end, color=GRID, alpha=0.15, zorder=0)
    return fig


def main() -> None:
    for name, fig in (("M11d-dev.png", plot_dev()), ("M11d-f3.png", plot_f3())):
        fig.savefig(IMG / name, dpi=150, facecolor=SURFACE)
        print(IMG / name)


if __name__ == "__main__":
    main()
