"""The figures of M09b's dev benchmark (run eval.benchmark --dev and --table first).

  uv run python -m eval.plot_benchmark

Writes docs/learn/img/M09b-*.png and prints F4's verifier table. Same conventions as M08: LLM + RL highlighted,
every other controller in grey, the oracle as an outline. Dev seeds: these show where v7 fails, they are not a result.
"""

import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from eval.benchmark import CEILING_SHARE, OWN_STUDY, RUNS, SUMMARY, ceiling_check
from eval.benchmark import CONTROLLERS as ALL_CONTROLLERS
from eval.generator import FAMILIES
from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from sim.plot import SURFACE, TEXT, TEXT_MUTED

LABELS = {"static": "Static", "rl_only": "RL only", "llm_only": "LLM only", "llm_only_v2": "LLM only v2",
          "proposed": "LLM + RL", "formula": "LLM + formula", "oracle": "Oracle"}
CONTROLLERS = tuple(c for c in ALL_CONTROLLERS if c not in OWN_STUDY)  # M09b's figures: M09b's controllers only
FAMILY_TITLES = {"f1": "F1 adapt to load", "f2": "F2 new goal", "f3": "F3 fast dynamics", "f4": "F4 verifier edge", "f5": "F5 easy"}


def read(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def colour(controller: str) -> str:
    return ACCENT if controller == "proposed" else NEUTRAL


def met_values(rows: list[dict], family: str, controller: str, key: str = "intent_met") -> np.ndarray:
    return np.array([float(r[key]) for r in rows if r["family"] == family and r["controller"] == controller and r[key] != ""])


def plot_met(rows: list[dict]) -> Figure:
    fig = Figure(figsize=(14, 4.2), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, len(FAMILIES), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        for x, c in enumerate(CONTROLLERS):
            ys = met_values(rows, family, c) * 100
            if not len(ys):
                continue
            jitter = np.random.default_rng(x).uniform(-0.18, 0.18, len(ys))
            if c == "oracle":
                ax.scatter(x + jitter, ys, s=10, facecolors="none", edgecolors=TEXT_MUTED, linewidths=0.6, zorder=2)
            else:
                ax.scatter(x + jitter, ys, s=10, color=colour(c), alpha=0.55, linewidths=0, zorder=2)
            ax.hlines(np.mean(ys), x - 0.3, x + 0.3, color=TEXT, linewidth=1.6, zorder=3)
        ax.set_xticks(range(len(CONTROLLERS)), [LABELS[c] for c in CONTROLLERS], rotation=60, ha="right", fontsize=8)
        ax.set_ylim(-4, 104)
        titled(ax, FAMILY_TITLES[family])
    axes[0].set_ylabel("seconds of the window met, %", color=TEXT_MUTED)
    beaten = ceiling_check([{**r, "intent_met": float(r["intent_met"])} for r in rows])
    role = "the ceiling" if beaten["all"] <= CEILING_SHARE else (
        "a reference: beaten by > 0.05 on " + ", ".join(f"{v:.0%} of {f.upper()}" for f, v in beaten.items() if f != "all" and v > 0) + " intents")
    fig.suptitle(f"Dev benchmark, policy v7: one dot per intent, bar = mean. Outline = the oracle, {role}", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def plot_paired(summary: list[dict]) -> Figure:
    """LLM + RL minus each other controller, per family: mean paired difference with its 95 % interval."""
    others = [c for c in CONTROLLERS if c != "proposed"]
    fig = Figure(figsize=(14, 3.8), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, len(FAMILIES), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        for y, c in enumerate(others):
            row = next((r for r in summary if r["family"] == family and r["controller"] == c), None)
            if row is None or row["mean_diff"] == "":
                continue
            mid, low, high = (float(row[k]) * 100 for k in ("mean_diff", "mean_diff_low", "mean_diff_high"))
            ax.hlines(y, low, high, color=NEUTRAL, linewidth=2)
            ax.scatter(mid, y, s=30, color=ACCENT if low > 0 else (TEXT if high < 0 else NEUTRAL), zorder=3)
        ax.axvline(0, color=TEXT_MUTED, linewidth=0.8, linestyle=":")
        ax.set_yticks(range(len(others)), [LABELS[c] for c in others], fontsize=8.5)
        ax.set_xlim(-105, 105)
        titled(ax, FAMILY_TITLES[family])
        ax.set_xlabel("LLM + RL minus it, points", color=TEXT_MUTED)
    fig.suptitle("Paired by intent, mean difference: right of 0, LLM + RL meets its goal more often. Purple = interval above 0, black = below",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def plot_cost(rows: list[dict]) -> Figure:
    """Over-serving: the share above need of the slices an intent names, per family."""
    fig = Figure(figsize=(14, 3.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, len(FAMILIES), sharey=True)
    shown = [c for c in CONTROLLERS if c != "oracle"]
    for ax, family in zip(axes, FAMILIES):
        for x, c in enumerate(shown):
            ys = met_values(rows, family, c, "over_serving")
            if len(ys):
                ax.scatter(x + np.random.default_rng(x).uniform(-0.18, 0.18, len(ys)), ys, s=9, color=colour(c), alpha=0.5, linewidths=0)
                ax.hlines(np.median(ys), x - 0.3, x + 0.3, color=TEXT, linewidth=1.5)
        ax.axhline(0, color=TEXT_MUTED, linewidth=0.8, linestyle=":")
        ax.set_xticks(range(len(shown)), [LABELS[c] for c in shown], rotation=60, ha="right", fontsize=8)
        titled(ax, FAMILY_TITLES[family])
    axes[0].set_ylabel("share above need (fraction of the cell)", color=TEXT_MUTED)
    fig.suptitle("Over-serving: how much more of the cell the named slice got than its goal needed (negative = less)",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def verifier_table(rows: list[dict]) -> None:
    """F4: what LLM + RL's verifier did with each intent, against the oracle's feasibility label."""
    mine = [r for r in rows if r["family"] == "f4" and r["controller"] == "proposed"]
    print("\nF4, LLM + RL: verifier outcome against the oracle's label (an intent is feasible if met >= 90 % by the oracle)")
    for label in ("True", "False"):
        sub = [r for r in mine if r["feasible"] == label]
        counts = {o: sum(r["outcome"] == o for r in sub) for o in sorted({r["outcome"] for r in sub})}
        print(f"  feasible={label}: {len(sub)} intents, outcomes {counts}")


def main() -> None:
    rows, summary = read(RUNS), read(SUMMARY)
    IMG.mkdir(parents=True, exist_ok=True)
    for name, fig in {"M09b-met": plot_met(rows), "M09b-paired": plot_paired(summary), "M09b-cost": plot_cost(rows)}.items():
        fig.savefig(IMG / f"{name}.png", dpi=120)
        print(f"wrote {IMG / name}.png")
    verifier_table(rows)


if __name__ == "__main__":
    main()
