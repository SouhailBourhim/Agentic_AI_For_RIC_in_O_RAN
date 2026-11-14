"""The figures of M09e's testbed check (run `eval.m09e --score --final` first).

  uv run python -m eval.plot_m09e

Writes docs/learn/img/M09e-instances.png (per instance and controller: M09d's simulator prediction, its mean as a
hollow ring and its [min, max] widened by 0.15 as a grey bar, against the testbed's two repeats, small dots, and their
mean, a filled dot) and M09e-pairs.png (the 24 pairs LLM + RL minus each other controller: the band the prediction
allows, grey, and the testbed difference, a dot; the verdict written next to it). Reads only the committed tables
(docs/learn/data/M09d-predictions.csv, M09e-runs.csv, M09e-pairs.csv). Same conventions as M08a-M09d: controllers by
position, LLM + RL in purple, everything else grey, no colour that stands for a verdict.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from eval.m09d import PREDICTIONS_CSV, TIE
from eval.m09e import CONTROLLERS, PAIRS_CSV, RUNS_CSV
from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from eval.validate_sim import CELL_TOLERANCE
from sim.plot import GRID, SURFACE, TEXT, TEXT_MUTED

LABELS = {"static": "static", "llm_only_v2": "LLM only v2", "formula": "LLM + formula", "proposed": "LLM + RL"}
FAMILY = {"f1": "F1", "f2": "F2 new goal", "f3": "F3 fast dynamics", "f5": "F5 easy"}
SUBFAMILY = {"f1_19": "F1a URLLC surge + bound", "f1_38": "F1b target raised twice"}  # M09d's sample, as its note names them


def read(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def repeat_means(runs: list[dict]) -> dict[tuple[str, str], list[float]]:
    """Per instance and controller, each repeat's mean intent met (the unit M09e scores), in repeat order."""
    met: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in runs:
        met[(r["instance"], r["controller"], r["repeat"])].append(float(r["intent_met"]))
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (name, controller, _), values in sorted(met.items()):
        out[(name, controller)].append(float(np.mean(values)))
    return out


def plot_instances(predictions: list[dict], runs: list[dict]) -> Figure:
    """Small multiples, one per instance, on one shared 0-1 axis."""
    testbed = repeat_means(runs)
    names = list(dict.fromkeys(p["instance"] for p in predictions))
    fig = Figure(figsize=(13, 6.2), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(2, 4, sharey=True)
    for ax, name in zip(axes.flat, names):
        for x, controller in enumerate(CONTROLLERS):
            p = next(q for q in predictions if q["instance"] == name and q["controller"] == controller)
            colour = ACCENT if controller == "proposed" else NEUTRAL
            ax.vlines(x - 0.12, float(p["low"]), float(p["high"]), color=GRID, linewidth=7, zorder=1)
            ax.scatter(x - 0.12, float(p["mean"]), s=46, facecolors=SURFACE, edgecolors=colour, linewidths=1.6, zorder=3)
            reps = testbed[(name, controller)]
            ax.scatter([x + 0.14] * len(reps), reps, s=12, color=colour, alpha=0.55, linewidths=0, zorder=2)
            ax.scatter(x + 0.14, float(np.mean(reps)), s=46, color=colour, edgecolors=SURFACE, linewidths=1.2, zorder=4)
        titled(ax, f"{name}  ({SUBFAMILY.get(name, FAMILY[name.split('_')[0]])})")
        ax.set_xticks(range(len(CONTROLLERS)), [LABELS[c].replace(" + ", " +\n") for c in CONTROLLERS], fontsize=8)
        ax.set_xlim(-0.6, len(CONTROLLERS) - 0.4)
        ax.set_ylim(-0.03, 1.06)
    for ax in axes[:, 0]:
        ax.set_ylabel("intent met (share of the window)", color=TEXT_MUTED, fontsize=8.5)
    handles = [
        axes[0, 0].scatter([], [], s=46, facecolors=SURFACE, edgecolors=NEUTRAL, linewidths=1.6, label="simulator, M09d mean"),
        axes[0, 0].vlines([], 0, 1, color=GRID, linewidth=7, label=f"simulator [min, max] ± {CELL_TOLERANCE:g}"),
        axes[0, 0].scatter([], [], s=46, color=NEUTRAL, edgecolors=SURFACE, label="testbed, mean of 2 repeats"),
        axes[0, 0].scatter([], [], s=12, color=NEUTRAL, alpha=0.55, label="testbed, each repeat"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=4, frameon=False, fontsize=8.5, labelcolor=TEXT)
    fig.suptitle("M09e: the testbed against M09d's written predictions, per instance (LLM + RL in purple)",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def allowed_band(predicted: str) -> tuple[float, float]:
    """The testbed differences that confirm a prediction (eval.m09e.confirmed), clipped to the drawable range."""
    if predicted == "tie":
        return -CELL_TOLERANCE, CELL_TOLERANCE
    return (TIE, 1.0) if predicted == "+" else (-1.0, -TIE)


def plot_pairs(pairs: list[dict]) -> Figure:
    """One row per pair, grouped by the other controller, instances in order inside each group."""
    others = [c for c in CONTROLLERS if c != "proposed"]
    ordered = [p for other in others for p in pairs if p["other"] == other]
    fig = Figure(figsize=(10, 7.4), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    for y, p in enumerate(reversed(ordered)):
        low, high = allowed_band(p["predicted"])
        ax.hlines(y, low, high, color=GRID, linewidth=7, zorder=1)
        difference = float(p["testbed_difference"])
        ok = p["confirmed"] == "True"
        ax.scatter(difference, y, s=40, color=ACCENT if ok else SURFACE, edgecolors=ACCENT, linewidths=1.6, zorder=3)
        ax.annotate(f"{difference:+.2f}" + ("" if ok else "  MISSED"), (max(difference, high if p["predicted"] != "+" else difference), y),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=TEXT)
    labels = [f"{p['instance']} vs {LABELS[p['other']]}   predicted {p['predicted']}" for p in reversed(ordered)]
    ax.set_yticks(range(len(ordered)), labels, fontsize=8)
    for k in range(1, len(others)):  # a line between the groups
        ax.axhline(len(ordered) - 8 * k - 0.5, color=TEXT_MUTED, linewidth=0.6)
    ax.axvline(0, color=TEXT_MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlim(-0.75, 0.95)
    ax.set_xlabel("testbed: LLM + RL minus the other controller (mean intent met)", color=TEXT_MUTED, fontsize=8.5)
    confirmed = sum(p["confirmed"] == "True" for p in pairs)
    titled(ax, f"Grey: what the prediction allows (tie: within ±{CELL_TOLERANCE:g}; + or −: beyond {TIE:g} that way). "
               f"Filled dot: confirmed ({confirmed} of {len(pairs)})")
    fig.suptitle("M09e: the 24 predicted differences, checked on the testbed", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    figures = {"M09e-instances": plot_instances(read(PREDICTIONS_CSV), read(RUNS_CSV)), "M09e-pairs": plot_pairs(read(PAIRS_CSV))}
    for name, fig in figures.items():
        fig.savefig(IMG / f"{name}.png", dpi=120)
        print(f"wrote {IMG / name}.png")


if __name__ == "__main__":
    main()
