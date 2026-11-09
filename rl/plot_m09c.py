"""The figures of M09c (policy v8): the dev selection, the goal test, and the confirm seeds' validation history.

  uv run python -m rl.plot_m09c

Reads docs/learn/data/M09c-selection.csv (eval.benchmark --compare), runs/m09c/m09_validation.json and
runs/m09c/test_result.json (rl.goal_test) and models/m09c/confirm/*_validation.json (rl.train's checkpoints).
Writes docs/learn/img/M09c-*.png. Same conventions as M08/M09b: the chosen recipe in the accent colour, the rest grey.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from eval.generator import FAMILIES
from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from rl.goal_test import T1A_MIN_SHARE, T1B_MIN_CAPTURED
from sim.plot import GRID, SURFACE, TEXT, TEXT_MUTED

SELECTION = Path("docs/learn/data/M09c-selection.csv")
VALIDATION = Path("runs/m09c/m09_validation.json")
TEST = Path("runs/m09c/test_result.json")
CONFIRM = Path("models/m09c/confirm")
CHOSEN = "v8_nooverserve"
LIGHT = "#b4b2ab"  # the recipes not chosen: present, but behind the chosen one and v7
FAMILY_TITLES = {"f1": "F1 adapt to load", "f2": "F2 new goal", "f3": "F3 fast dynamics", "f4": "F4 verifier edge",
                 "f5": "F5 easy"}


def recipe_of(name: str) -> str:
    """'td3_v8_mm1_seed30_best' -> 'v8_mm1'; 'm09c_v8' -> 'v8'."""
    return re.sub(r"^(td3_|m09c_)", "", name.split("_seed")[0])


def plot_selection(rows: list[dict]) -> Figure:
    """One panel per family (and all intents): each recipe's mean intent met as LLM + RL, against v7's."""
    fig = Figure(figsize=(14, 3.6), facecolor=SURFACE, layout="constrained")
    panels = [("all", "All 265 intents")] + [(f, FAMILY_TITLES[f]) for f in FAMILIES]
    axes = fig.subplots(1, len(panels), sharey=True)
    names = ["v7"] + [recipe_of(r["study"]) for r in rows]
    for ax, (key, title) in zip(axes, panels):
        mine, v7 = ("mean_met", "v7_mean_met") if key == "all" else (f"{key}_met", f"{key}_v7")
        values = [float(rows[0][v7])] + [float(r[mine]) for r in rows]
        colours = [NEUTRAL] + [ACCENT if n == CHOSEN else LIGHT for n in names[1:]]
        y = np.arange(len(names))[::-1]
        ax.barh(y, values, color=colours, height=0.62)
        ax.axvline(values[0], color=NEUTRAL, lw=1, ls=(0, (3, 2)))
        for yi, v in zip(y, values):
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color=TEXT_MUTED)
        lo = min(values)
        ax.set_xlim(max(0.0, lo - 0.15), 1.08)
        ax.set_yticks(y, names)
        titled(ax, title)
        ax.set_xlabel("mean intent met", color=TEXT_MUTED, fontsize=8.5)
    fig.suptitle("LLM + RL on the 200 dev instances, one seed per recipe (dashed: v7, the champion)",
                 x=0.01, ha="left", color=TEXT, fontsize=10.5)
    return fig


def plot_goal_test(validation: dict, test: dict) -> Figure:
    """T1a against T1b per checkpoint: round 1 on validation, and the confirm seeds on the held-out test."""
    fig = Figure(figsize=(11, 4.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    sets = [(axes[0], validation["models"], "Round 1, m09_validation (4 recipes x 3 seeds)"),
            (axes[1], test["models"], f"Round 2, m09_test, scored once ({CHOSEN}, seeds 40-44)")]
    for ax, models, title in sets:
        ax.fill_between([T1A_MIN_SHARE, 1.0], T1B_MIN_CAPTURED, 1.0, color=GRID, alpha=0.6, lw=0)
        ax.text(0.995, 0.985, "pass region", ha="right", va="top", fontsize=8, color=TEXT_MUTED)
        for name, m in models.items():
            if m.get("no_checkpoint"):
                ax.text(0.81, 0.33, f"seed {name.split('_seed')[1][:2]}: no checkpoint\n(never beat the equal split)",
                        fontsize=8, color=ACCENT)
                continue
            chosen = recipe_of(name) == CHOSEN
            ax.scatter(m["t1a_share_positive"], m["t1b_captured"], s=42, color=ACCENT if chosen else LIGHT,
                       edgecolor=SURFACE, lw=1.2, zorder=3)
            if models is test["models"]:  # five points: each gets its seed; on validation the legend says enough
                ax.annotate(f"seed {name.split('_seed')[1][:2]}", (m["t1a_share_positive"], m["t1b_captured"]),
                            xytext=(5, -11) if name.endswith("seed41_best") else (5, 3), textcoords="offset points",
                            fontsize=7.5, color=TEXT_MUTED)
        ax.axvline(T1A_MIN_SHARE, color=NEUTRAL, lw=1, ls=(0, (3, 2)))
        ax.axhline(T1B_MIN_CAPTURED, color=NEUTRAL, lw=1, ls=(0, (3, 2)))
        titled(ax, title)
        ax.set_xlabel("T1a: share of material cases the objective helps", color=TEXT_MUTED, fontsize=8.5)
    axes[0].scatter([0.93], [0.66], marker="D", s=40, color=NEUTRAL, edgecolor=SURFACE, zorder=4)
    handles = [axes[0].scatter([], [], s=42, color=ACCENT, label=f"{CHOSEN} (chosen on the dev benchmark)"),
               axes[0].scatter([], [], s=42, color=LIGHT, label="v8, v8_mm1, v8_nohistory"),
               axes[0].scatter([], [], marker="D", s=40, color=NEUTRAL, label="v7, the champion (reference)")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=7.5, frameon=False, labelcolor=TEXT_MUTED)
    axes[0].set_ylabel("T1b: share of the oracle's advantage captured", color=TEXT_MUTED, fontsize=8.5)
    axes[0].set_xlim(0.80, 1.0)
    axes[0].set_ylim(0.3, 1.0)
    return fig


def plot_confirm_history() -> Figure:
    """Each confirm seed's validation reward and T1a at every checkpoint: seed 40 never leaves the equal split."""
    fig = Figure(figsize=(11, 3.8), facecolor=SURFACE, layout="constrained")
    ax_r, ax_a = fig.subplots(1, 2, sharex=True)
    equal = None
    for path in sorted(CONFIRM.glob("*_validation.json")):
        seed = path.stem.split("_seed")[1][:2]
        rows = json.loads(path.read_text())
        steps = np.array([r["step"] for r in rows]) / 1000
        equal = rows[0]["equal_split_reward"]
        failed = seed == "40"
        style = {"color": ACCENT if failed else LIGHT, "lw": 2 if failed else 1.3, "zorder": 3 if failed else 2}
        style["label"] = "seed 40" if failed else ("seeds 41-44" if seed == "41" else None)
        ax_r.plot(steps, [r["reward"] for r in rows], **style)
        ax_a.plot(steps, [r["t1a_share_positive"] for r in rows], **style)
    ax_r.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=TEXT_MUTED)
    ax_r.axhline(equal, color=NEUTRAL, lw=1, ls=(0, (3, 2)))
    ax_r.text(52, equal + 0.003, "equal split", fontsize=8, color=NEUTRAL)
    ax_a.axhline(T1A_MIN_SHARE, color=NEUTRAL, lw=1, ls=(0, (3, 2)))
    ax_a.text(52, T1A_MIN_SHARE + 0.015, "T1a threshold", fontsize=8, color=NEUTRAL)
    titled(ax_r, "Validation reward per step")
    titled(ax_a, "Validation T1a")
    for ax in (ax_r, ax_a):
        ax.set_xlabel("training steps (thousands)", color=TEXT_MUTED, fontsize=8.5)
    fig.suptitle(f"Round 2 ({CHOSEN}): the checkpoint scored every 25k steps on m09_validation",
                 x=0.01, ha="left", color=TEXT, fontsize=10.5)
    return fig


def main() -> None:
    with SELECTION.open() as f:
        rows = list(csv.DictReader(f))
    figures = {"M09c-selection.png": plot_selection(rows),
               "M09c-goal-test.png": plot_goal_test(json.loads(VALIDATION.read_text()), json.loads(TEST.read_text())),
               "M09c-confirm-history.png": plot_confirm_history()}
    for name, fig in figures.items():
        fig.savefig(IMG / name, dpi=150, facecolor=SURFACE)
        print(f"wrote {IMG / name}")


if __name__ == "__main__":
    main()
