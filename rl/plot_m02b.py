"""Figures for M02b, built from runs/m02b/*.json (nothing typed in by hand).

Run with:  uv run python -m rl.plot_m02b
Writes into docs/learn/img/:
- M02b-ablation.png   validation T1b of every dev-round model, grouped by recipe (one dot per
                      seed), with the M02 policies on the same validation cases for comparison
- M02b-test.png       the held-out test: T1a and T1b of the 5 confirmation seeds against the
                      pass thresholds, and the original M02 gap next to the oracle-valid test
Inputs: validation_dev.json (dev round), validation_m02.json (M02 policies on validation),
test_result.json (held-out test).
"""

import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from rl.goal_test import T1A_MIN_SHARE, T1B_MIN_CAPTURED
from sim.plot import GRID, SURFACE, TEXT, TEXT_MUTED, _style

RESULT_DIR = Path("runs/m02b")
OUT_DIR = Path("docs/learn/img")
RECIPE_COLOUR = "#6250d6"
M02_COLOUR = "#8a8984"
LABELS = {"td3_noreward": "all changes,\ntrained with reward v1", "td3_nofilter": "no filter\nin training",
          "td3_full": "all changes\n(reward v2)", "td3_nocoverage": "no wider\nsurges/load",
          "td3_norelabel": "no objective\nrelabeling", "ddpg": "M02 DDPG", "td3": "M02 TD3", "td3v2": "M02 TD3 v2"}


def recipe_of(name: str) -> str:
    return name.split("_seed")[0]


def plot_ablation(dev: dict, m02: dict) -> Figure:
    models = {**{n: m for n, m in dev["models"].items()}, **m02["models"]}
    recipes = sorted({recipe_of(n) for n in models}, key=lambda r: -np.mean(
        [m["t1b_captured"] for n, m in models.items() if recipe_of(n) == r]))
    fig = Figure(figsize=(11, 4.6), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    for i, recipe in enumerate(recipes):
        pts = [(m["t1b_captured"], m["pass"]) for n, m in models.items() if recipe_of(n) == recipe]
        colour = M02_COLOUR if recipe in ("ddpg", "td3", "td3v2") else RECIPE_COLOUR
        xs = i + np.linspace(-0.12, 0.12, len(pts))
        ax.scatter(xs, [p[0] for p in pts], s=70, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3,
                   marker="o")
        passes = sum(p[1] for p in pts)
        ax.annotate(f"{passes}/{len(pts)} seeds pass", (i, 1.04), ha="center", fontsize=8, color=TEXT_MUTED)
    ax.axhline(T1B_MIN_CAPTURED, color=GRID, linewidth=1.2, linestyle="--")
    ax.annotate(f"T1b threshold {T1B_MIN_CAPTURED}", (len(recipes) - 0.5, T1B_MIN_CAPTURED), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=8, color=TEXT_MUTED)
    ax.set_xticks(range(len(recipes)), [LABELS.get(r, r) for r in recipes], fontsize=8)
    ax.set_ylim(0, 1.12)
    _style(ax, "T1b: share of the oracle's advantage captured (validation)")
    ax.set_title("Dot = one seed. Grey: M02 policies. A seed passes only if T1a, T1b and T2 all pass.",
                 color=TEXT_MUTED, fontsize=9, loc="left")
    fig.suptitle("Which change makes goal conditioning reliable? (development round, validation cases)",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    return fig


def plot_test(test: dict) -> Figure:
    names = list(test["models"])
    fig = Figure(figsize=(11, 4.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 3)
    panels = [("t1a_share_positive", "T1a: cases where telling the objective helps", T1A_MIN_SHARE),
              ("t1b_captured", "T1b: share of the oracle's advantage captured", T1B_MIN_CAPTURED),
              ("original_gap", "original M02 measure: URLLC share gap (not the bar)", 0.10)]
    x = np.arange(len(names))
    for ax, (key, title, threshold) in zip(axes, panels):
        values = [test["models"][n][key] for n in names]
        ax.bar(x, values, color=RECIPE_COLOUR, width=0.6)
        ax.axhline(threshold, color=TEXT_MUTED, linewidth=1.2, linestyle="--")
        for xi, v in zip(x, values):
            ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=TEXT_MUTED)
        ax.set_xticks(x, [n.split("_seed")[1].split("_")[0] for n in names])
        ax.set_xlabel("confirmation seed", color=TEXT_MUTED, fontsize=9)
        ax.set_ylim(0, 1.1)
        _style(ax, "")
        ax.set_title(title, color=TEXT, fontsize=9, loc="left")
    fig.suptitle(f"Held-out test, run once: {sum(m['pass'] for m in test['models'].values())} of "
                 f"{len(names)} seeds pass (dashed: threshold)", color=TEXT, fontsize=12, x=0.01, ha="left")
    return fig


def main() -> None:
    dev = json.loads((RESULT_DIR / "validation_dev.json").read_text())
    m02 = json.loads((RESULT_DIR / "validation_m02.json").read_text())
    test = json.loads((RESULT_DIR / "test_result.json").read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in {"M02b-ablation.png": plot_ablation(dev, m02), "M02b-test.png": plot_test(test)}.items():
        fig.savefig(OUT_DIR / name, dpi=120)
        print(f"wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
