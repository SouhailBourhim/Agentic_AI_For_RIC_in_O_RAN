"""The figures of M09f's dangerous-intent test (run `eval.m09f --score --final` first).

  uv run python -m eval.plot_m09f

Writes docs/learn/img/M09f-stoppers.png (per category and controller, who stopped each item: the LLM asked or
refused, the verifier with a check, or nothing, when something safe was applied) and M09f-blocked.png (every
dangerous proposal, by the verifier check that blocked it). Reads only the committed tables
(docs/learn/data/M09f-items.csv, M09f-proposals.csv). Colours as M08b's interface figures: amber = refused before the
verifier, purple = rejected by the verifier, grey = applied. Whether anything unsafe was applied is written in the
title, not left to a colour: it is the pass rule.
"""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from eval.m09f import CONTROLLERS, ITEMS_CSV, PROPOSALS_CSV
from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from sim.plot import SURFACE, TEXT, TEXT_MUTED

LABELS = {"proposed": "LLM + RL", "llm_only_v2": "LLM only v2", "rules": "rule-based agent"}
CATEGORIES = ["1", "2", "3", "4", "5", "6", "7", "8", "control"]
CATEGORY_LABELS = {"1": "1 starve a slice", "2": "2 targets don't fit", "3": "3 above trained range",
                   "4": "4 impossible latency", "5": "5 bad window", "6": "6 takes a guarantee away",
                   "7": "7 abrupt flip", "8": "8 bypass, injection", "control": "safe controls"}
STOPPERS = [("llm", "the LLM asked or refused", "#eda100"), ("verifier", "the verifier rejected it", ACCENT),
            ("none", "applied (safe)", NEUTRAL)]


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def kind(stopper: str) -> str:
    return "verifier" if stopper.startswith("verifier") else stopper


def plot_stoppers(items: list[dict]) -> Figure:
    """One panel per controller, one bar per category, split by who stopped the item (share of its plays)."""
    controllers = [c for c in CONTROLLERS if any(r["controller"] == c for r in items)]
    fig = Figure(figsize=(13, 4.8), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, len(controllers), sharey=True)
    for ax, controller in zip(axes, controllers):
        mine = [r for r in items if r["controller"] == controller]
        for y, category in enumerate(reversed(CATEGORIES)):
            plays = [r for r in mine if r["category"] == category]
            counts = Counter(kind(r["stopper"]) for r in plays)
            left = 0.0
            for key, label, colour in STOPPERS:
                share = counts[key] / len(plays) if plays else 0.0
                if share:
                    ax.barh(y, share, left=left, height=0.62, color=colour, edgecolor=SURFACE, linewidth=2, label=label)
                    if share >= 0.12:
                        ax.text(left + share / 2, y, f"{counts[key]}", ha="center", va="center", fontsize=8,
                                color=SURFACE if key != "none" else TEXT)
                    left += share
        unsafe = sum(r["unsafe"] == "True" for r in mine)
        titled(ax, f"{LABELS[controller]}: {len(mine)} plays, {unsafe} unsafe applied")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "50 %", "100 %"])
        ax.set_xlabel("share of the category's plays", color=TEXT_MUTED, fontsize=8.5)
    axes[0].set_yticks(range(len(CATEGORIES)), [CATEGORY_LABELS[c] for c in reversed(CATEGORIES)], fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend([unique[label] for _, label, _ in STOPPERS if label in unique],
               [label for _, label, _ in STOPPERS if label in unique],
               loc="outside lower center", ncols=3, frameon=False, fontsize=8.5, labelcolor=TEXT)
    fig.suptitle("M09f: who stopped each dangerous intent (numbers in the bars: plays)", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def plot_blocked(proposals: list[dict]) -> Figure:
    """Every proposal the oracle calls dangerous on its own, by the check that blocked it; LLM + RL and rules."""
    dangerous = [p for p in proposals if p["dangerous"] == "True"]
    checks = sorted({p["check"] for p in dangerous}, key=lambda c: -sum(p["check"] == c for p in dangerous))
    series = [(c, LABELS[c], ACCENT if c == "proposed" else NEUTRAL) for c in ("proposed", "rules")]
    fig = Figure(figsize=(9, 3.6), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    y = np.arange(len(checks))
    for k, (controller, label, colour) in enumerate(series):
        counts = [sum(p["controller"] == controller and p["check"] == c for p in dangerous) for c in checks]
        bars = ax.barh(y[::-1] + (0.2 - 0.4 * k), counts, height=0.38, color=colour, edgecolor=SURFACE, linewidth=1.5, label=label)
        ax.bar_label(bars, labels=[str(n) if n else "" for n in counts], padding=3, fontsize=8, color=TEXT)
    ax.set_yticks(y[::-1], checks, fontsize=8.5)
    ax.set_xlabel("dangerous proposals (U1-U4 on the proposal alone)", color=TEXT_MUTED, fontsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT, loc="lower right")
    shown = {c: sum(p["controller"] == c for p in dangerous) for c in CONTROLLERS}
    blocked = {c: sum(p["controller"] == c and p["accepted"] == "False" for p in dangerous) for c in CONTROLLERS}
    titled(ax, " · ".join(f"{LABELS[c]}: {blocked[c]} of {shown[c]} blocked" for c in CONTROLLERS))
    fig.suptitle("M09f: dangerous proposals, by the verifier check that blocked them", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    figures = {"M09f-stoppers": plot_stoppers(read(ITEMS_CSV)), "M09f-blocked": plot_blocked(read(PROPOSALS_CSV))}
    for name, fig in figures.items():
        fig.savefig(IMG / f"{name}.png", dpi=120)
        print(f"wrote {IMG / name}.png")


if __name__ == "__main__":
    main()
