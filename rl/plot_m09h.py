"""The figures of M09h (policy v9): the dev round, the arm choice, and the held-out test.

  uv run python -m rl.plot_m09h

Reads models/m09h/{dev,confirm}/*_validation.json (rl.train's checkpoints), docs/learn/data/M09c-selection.csv
(eval.benchmark --compare), the dev runs in eval/results/m09b (LLM + RL per study, LLM + formula, formula_a10) and
runs/m09h/test_result.json (rl.goal_test). Writes docs/learn/img/M09h-*.png and prints the paired differences the
note quotes, so no number in the note is typed by hand. Same conventions as M09c: the chosen arm in the accent colour.
"""

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from eval.benchmark import STUDY, bootstrap_ci, work_of
from eval.generator import FAMILIES, dev_instances
from eval.metrics import measure_run
from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from rl.goal_test import T1A_MIN_SHARE, T1B_MIN_CAPTURED
from rl.plot_m09c import FAMILY_TITLES, LIGHT
from rl.sweep import median_seeds
from sim.plot import GRID, SURFACE, TEXT, TEXT_MUTED

SELECTION = Path("docs/learn/data/M09c-selection.csv")
DEV = Path("models/m09h/dev")
CONFIRM = Path("models/m09h/confirm")
TEST = Path("runs/m09h/test_result.json")
ARMS = ("v9", "v9_noswitch", "v9_gamma95", "v9_noou", "v9_noapplied", "v9_residual")  # benchmarked first, then out
CHOSEN = "v9_noswitch"
EXPLORE = "m09h_v9_residual_explore"  # exploratory, not selectable (DECISIONS 2026-11-10)


def picked(rows: list[dict]) -> dict | None:
    """The checkpoint a run kept (rl.train's rule for v9): T2 and T1a pass, then highest T1b, then reward."""
    eligible = [r for r in rows if r["t2_pass"] and (r["t1a_pass"] or not r.get("t1a_required"))]
    return max(eligible, key=lambda r: (r["t1b_captured"], r["reward"])) if eligible else None


def history(folder: Path, arm: str, seed: int) -> list[dict]:
    return json.loads((folder / f"td3_{arm}_seed{seed}_validation.json").read_text())


def met_of(controller: str, study: str = STUDY) -> dict[tuple[str, int], tuple[str, float]]:
    """Intent met per (instance, intent) of one controller's finished dev runs, with the family (as eval.benchmark's
    proposed_met, for any controller)."""
    out = {}
    for inst in dev_instances():
        work = work_of(inst, controller, study)
        if (work / "run.json").exists():
            for k, row in enumerate(measure_run(work, inst.scenario())["intents"]):
                out[(inst.name, k)] = (inst.family, row["intent_met"])
    return out


def family_means(met: dict) -> dict[str, float]:
    means = {"all": float(np.mean([v for _, v in met.values()]))}
    for family in FAMILIES:
        means[family] = float(np.mean([v for f, v in met.values() if f == family]))
    return means


def paired(a: dict, b: dict) -> tuple[float, float, float, int]:
    """Mean of a - b over the intents both played, with a bootstrap 95 % interval (eval.benchmark's resampling)."""
    keys = sorted(set(a) & set(b))
    diff = np.array([a[k][1] - b[k][1] for k in keys])
    low, high = bootstrap_ci(diff, stat=np.mean)
    return float(diff.mean()), low, high, len(keys)


def plot_dev_round() -> Figure:
    """Per arm, each seed's kept checkpoint by validation reward (the median rule ranks on it); the median seed is
    ringed. A seed that kept no checkpoint sits in the left column."""
    fig = Figure(figsize=(11, 4.2), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    medians = median_seeds(DEV)
    none_x, equal = 0.2275, None
    for row, arm in enumerate(ARMS):
        y0 = len(ARMS) - 1 - row
        median = medians.get(arm)
        for k, seed in enumerate(range(50, 55)):
            rows = history(DEV, arm, seed)
            equal = rows[0]["equal_split_reward"]
            pick = picked(rows)
            y = y0 + (k - 2) * 0.09
            colour = ACCENT if arm == CHOSEN else NEUTRAL if arm in ("v9", "v9_gamma95") else LIGHT
            if pick is None:
                ax.scatter(none_x, y, s=30, facecolor="none", edgecolor=colour, lw=1.2, zorder=3)
                continue
            ax.scatter(pick["reward"], y, s=34, color=colour, edgecolor=SURFACE, lw=1, zorder=3)
            if median is not None and median.name == f"td3_{arm}_seed{seed}_best.zip":
                ax.scatter(pick["reward"], y, s=150, facecolor="none", edgecolor=TEXT, lw=1.1, zorder=4)
        out = "" if median is not None else "   (median seed has no checkpoint: out)"
        ax.text(0.2585, y0, f"{arm}{out}", va="center", fontsize=8.5, color=TEXT if median else TEXT_MUTED)
    ax.axvline(0.2305, color=GRID, lw=1)
    ax.text(none_x, -0.6, "no checkpoint", ha="center", fontsize=7.5, color=TEXT_MUTED)
    ax.set_xlim(0.224, 0.274)
    ax.set_ylim(-0.8, len(ARMS) - 0.2)
    ax.set_yticks([])
    ax.set_xlabel("validation reward per step of the kept checkpoint (m10_validation)", color=TEXT_MUTED, fontsize=8.5)
    titled(ax, "Dev round: 6 arms x seeds 50-54; ringed = the arm's median seed, which plays the dev benchmark "
               f"(equal split {equal:.3f}, below every kept checkpoint)")
    return fig


def plot_selection(rows: list[dict], formula: dict[str, float], formula_a10: dict[str, float]) -> Figure:
    """One panel per family (and all intents): each benchmarked arm's mean intent met as LLM + RL, against v7's, with
    LLM + formula and formula_a10 as reference lines and the residual arm marked exploratory."""
    fig = Figure(figsize=(14, 3.8), facecolor=SURFACE, layout="constrained")
    panels = [("all", "All 265 intents")] + [(f, FAMILY_TITLES[f]) for f in FAMILIES]
    axes = fig.subplots(1, len(panels), sharey=True)
    names = ["v7"] + [r["study"].removeprefix("m09h_") for r in rows]
    for ax, (key, title) in zip(axes, panels):
        mine, v7 = ("mean_met", "v7_mean_met") if key == "all" else (f"{key}_met", f"{key}_v7")
        values = [float(rows[0][v7])] + [float(r[mine]) for r in rows]
        colours = [NEUTRAL] + [ACCENT if n == CHOSEN else LIGHT for n in names[1:]]
        y = np.arange(len(names))[::-1]
        bars = ax.barh(y, values, color=colours, height=0.62)
        for bar, name in zip(bars, names):
            if name.endswith("_explore"):
                bar.set(facecolor=SURFACE, edgecolor=NEUTRAL, hatch="////", lw=0.8)
        ax.axvline(values[0], color=NEUTRAL, lw=1, ls=(0, (3, 2)))
        ax.axvline(formula[key], color=TEXT, lw=1.2, ls=(0, (1, 1.5)))
        ax.axvline(formula_a10[key], color=TEXT_MUTED, lw=1, ls=(0, (6, 2)))
        for yi, v in zip(y, values):  # at the panel's right edge, clear of the reference lines
            ax.text(1.17, yi, f"{v:.2f}", va="center", ha="right", fontsize=8, color=TEXT_MUTED)
        ax.set_xlim(max(0.0, min(values + [formula_a10[key]]) - 0.15), 1.18)
        ax.set_xticks([t for t in ax.get_xticks() if t <= 1.0])
        ax.set_yticks(y, [n.replace("_residual_explore", "_residual\n(exploratory)") for n in names], fontsize=8)
        titled(ax, title)
        ax.set_xlabel("mean intent met", color=TEXT_MUTED, fontsize=8.5)
    fig.suptitle("LLM + RL on the 200 dev instances, median seed per arm. Lines: v7 (grey dashes), "
                 "LLM + formula (dots), formula_a10 (long dashes)", x=0.01, ha="left", color=TEXT, fontsize=10.5)
    return fig


def plot_goal_test(test: dict) -> Figure:
    """Per confirm seed: the kept checkpoint's T1a and T1b on validation, then on the held-out test."""
    fig = Figure(figsize=(11, 3.9), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 2, sharex=True)
    seeds = range(60, 65)
    for ax, key, threshold, label in ((axes[0], "t1a_share_positive", T1A_MIN_SHARE, "T1a"),
                                      (axes[1], "t1b_captured", T1B_MIN_CAPTURED, "T1b")):
        for x, seed in enumerate(seeds):
            val = picked(history(CONFIRM, CHOSEN, seed))[key]
            tst = test["models"][f"td3_{CHOSEN}_seed{seed}_best"]
            colour = ACCENT if not tst["pass"] else NEUTRAL
            ax.plot([x - 0.15, x + 0.15], [val, tst[key]], color=colour, lw=1.2)
            ax.scatter(x - 0.15, val, s=36, facecolor=SURFACE, edgecolor=colour, lw=1.3, zorder=3)
            ax.scatter(x + 0.15, tst[key], s=40, color=colour, edgecolor=SURFACE, lw=1, zorder=3)
            ax.text(x + 0.2, tst[key], f"{tst[key]:.2f}", va="center", fontsize=7.5, color=colour)
        ax.axhline(threshold, color=NEUTRAL, lw=1, ls=(0, (3, 2)))
        ax.text(4.45, threshold, f"{label} threshold", va="bottom", ha="right", fontsize=7.5, color=NEUTRAL)
        ax.set_xticks(range(len(seeds)), [f"seed {s}" for s in seeds])
        titled(ax, f"{label}: validation (hollow) -> held-out test (filled)")
    axes[0].set_ylim(0.82, 1.0)
    axes[1].set_ylim(0.4, 0.8)
    fig.suptitle(f"Confirm round ({CHOSEN}): m10_test scored once; seed 60 (accent) fails T1a, so the recipe fails",
                 x=0.01, ha="left", color=TEXT, fontsize=10.5)
    return fig


def main() -> None:
    with SELECTION.open() as f:
        rows = [r for r in csv.DictReader(f) if r["study"].startswith("m09h_")]
    met = {name: met_of(c, s) for name, (c, s) in {
        "formula": ("formula", STUDY), "formula_a10": ("formula_a10", STUDY), "v7": ("proposed", STUDY),
        CHOSEN: ("proposed", f"m09h_{CHOSEN}"), "residual": ("proposed", EXPLORE)}.items()}
    print("paired by intent, mean difference [95 % bootstrap interval]:")
    for a, b in ((CHOSEN, "formula"), (CHOSEN, "formula_a10"), (CHOSEN, "v7"), ("residual", "formula"),
                 ("residual", "formula_a10")):
        d, low, high, n = paired(met[a], met[b])
        print(f"  {a:>12} - {b:<12} {d:+.3f} [{low:+.3f}, {high:+.3f}]  ({n} intents)")
    figures = {"M09h-dev-round.png": plot_dev_round(),
               "M09h-selection.png": plot_selection(rows, family_means(met["formula"]), family_means(met["formula_a10"])),
               "M09h-goal-test.png": plot_goal_test(json.loads(TEST.read_text()))}
    for name, fig in figures.items():
        fig.savefig(IMG / name, dpi=150, facecolor=SURFACE)
        print(f"wrote {IMG / name}")


if __name__ == "__main__":
    main()
