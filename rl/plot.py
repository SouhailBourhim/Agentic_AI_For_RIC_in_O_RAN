"""Figures for M02, built from the training logs and runs/m02/eval.json.

Run with:  uv run python -m rl.plot        (after rl.train and rl.evaluate)
Writes into docs/learn/img/:
- M02-training.png      reward per step during training, per algorithm and seed
- M02-goal-gap.png      mean URLLC share under the two objectives, every policy, vs the criterion
- M02-objectives.png    one policy on the URLLC-spike traffic under the two objectives
- M02-eval.png          reward per step over the 100 evaluation episodes, every controller
Every number comes from those files; nothing is typed in by hand.
"""

import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from rl.evaluate import OBJECTIVES, SHARE_GAP_CRITERION, SPIKE_STEPS
from rl.policy import SlicePolicy
from rl.train import HYPERPARAMETERS
from sim.plot import GRID, LINE_WIDTH, SURFACE, TEXT, TEXT_MUTED, _style, plot_spike_comparison, run_episode
from sim.scenarios import urllc_spike

RUN_DIR = Path("runs/m02")
MODEL_DIR = Path("models/m02")
EVAL_PATH = RUN_DIR / "eval.json"
OUT_DIR = Path("docs/learn/img")
EPISODE_STEPS = 200
SMOOTH_EPISODES = 20
# Controllers get their own colours (dataviz slots 7 and 6, and greys for baselines), never the
# slice colours, which keep meaning eMBB / URLLC / BE everywhere in the project.
ALGO_COLOURS = {"ddpg": "#6250d6", "td3": "#008300"}
BASELINE_COLOUR = "#8a8984"
DETAIL_POLICY = "td3_seed0"  # the one shown in detail in M02-objectives.png (said in the note)


def algo_of(name: str) -> str:
    return name.split("_seed")[0]


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average; the first points average over what exists so far."""
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    idx = np.arange(1, len(values) + 1)
    start = np.maximum(idx - window, 0)
    return (cumsum[idx] - cumsum[start]) / (idx - start)


def plot_training(eval_results: dict) -> Figure:
    """One panel per algorithm; one line per seed (smoothed), equal-split level for reference."""
    algos = sorted({algo_of(p.parent.name) for p in RUN_DIR.glob("*_seed*/monitor.csv")})
    fig = Figure(figsize=(12, 4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, len(algos), sharey=True, squeeze=False)[0]
    equal_level = float(np.mean(eval_results["random"]["equal split"]["reward"]))
    for ax, algo in zip(axes, algos):
        for path in sorted(RUN_DIR.glob(f"{algo}_seed*/monitor.csv")):
            rewards = np.loadtxt(path, delimiter=",", skiprows=2, usecols=0) / EPISODE_STEPS
            steps = (np.arange(len(rewards)) + 1) * EPISODE_STEPS
            seed = path.parent.name.split("_seed")[1]
            ax.plot(steps, smooth(rewards, SMOOTH_EPISODES), color=ALGO_COLOURS[algo],
                    linewidth=LINE_WIDTH, alpha=0.4 + 0.3 * int(seed), label=f"seed {seed}")
        ax.axvspan(0, HYPERPARAMETERS["learning_starts"], color="#efeee9", zorder=0)
        ax.axhline(equal_level, color=BASELINE_COLOUR, linewidth=1.0, linestyle="--")
        ax.annotate(f"equal split (eval mean {equal_level:.3f})", (steps[-1], equal_level), xytext=(0, -11),
                    textcoords="offset points", ha="right", fontsize=8, color=TEXT_MUTED)
        ax.set_title(algo.upper(), color=TEXT, fontsize=10, loc="left")
        _style(ax, "reward per step" if ax is axes[0] else "")
        ax.set_xlabel("environment steps (shaded: random warm-up)", color=TEXT_MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Training: episode reward per step, moving average over {SMOOTH_EPISODES} episodes "
                 "(each episode has a random objective and traffic)", color=TEXT, fontsize=12, x=0.01, ha="left")
    return fig


def plot_goal_gap(eval_results: dict) -> Figure:
    """Mean URLLC share under each objective, per policy: the goal-conditioning criterion."""
    gc = eval_results["goal_conditioning"]
    names = list(gc)
    fig = Figure(figsize=(11, 4), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    x = np.arange(len(names))
    for i, name in enumerate(names):
        protect, maxim = gc[name]["protect URLLC"]["share_urllc"], gc[name]["maximise eMBB"]["share_urllc"]
        colour = ALGO_COLOURS[algo_of(name)]
        ax.plot([i, i], [maxim, protect], color=colour, linewidth=2.5, solid_capstyle="round")
        ax.scatter([i], [protect], s=60, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3, marker="^")
        ax.scatter([i], [maxim], s=60, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3, marker="v")
        gap = protect - maxim
        verdict = "≥ 0.10" if gap >= SHARE_GAP_CRITERION else "< 0.10"
        ax.annotate(f"gap {gap:.2f}\n{verdict}", (i, protect), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=TEXT_MUTED)
    ax.set_xticks(x, names)
    ax.set_ylim(0, max(gc[n]["protect URLLC"]["share_urllc"] for n in names) + 0.12)
    _style(ax, "mean URLLC PRB share")
    ax.set_title("▲ protect URLLC   ▼ maximise eMBB   (same URLLC-spike traffic, mean over "
                 f"{SPIKE_STEPS} s; criterion: gap ≥ {SHARE_GAP_CRITERION} on every seed)",
                 color=TEXT_MUTED, fontsize=9, loc="left")
    fig.suptitle("Goal conditioning: does the URLLC share follow the objective?", color=TEXT,
                 fontsize=12, x=0.01, ha="left")
    return fig


def plot_objectives(policy_name: str) -> Figure:
    """One policy, same traffic, the two objectives side by side (reuses the M01 layout)."""
    policy = SlicePolicy.load(MODEL_DIR / f"{policy_name}.zip")
    traffic, _ = urllc_spike()
    runs = {name: run_episode(policy.action, traffic, obj, steps=SPIKE_STEPS) for name, obj in OBJECTIVES.items()}
    return plot_spike_comparison(runs, OBJECTIVES, traffic["urllc"], what=f"— {policy_name} under two objectives")


def plot_eval(eval_results: dict) -> Figure:
    """Reward per step over the evaluation episodes: one box per controller."""
    random = eval_results["random"]
    names = list(random)
    fig = Figure(figsize=(11, 4.5), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    data = [random[n]["reward"] for n in names]
    colours = [ALGO_COLOURS.get(algo_of(n), BASELINE_COLOUR) for n in names]
    box = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                     medianprops={"color": SURFACE, "linewidth": 2},
                     flierprops={"marker": "o", "markersize": 3, "markerfacecolor": TEXT_MUTED,
                                 "markeredgecolor": "none", "alpha": 0.5},
                     whiskerprops={"color": TEXT_MUTED}, capprops={"color": TEXT_MUTED})
    for patch, colour in zip(box["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_edgecolor(colour)
    for i, values in enumerate(data, start=1):
        ax.annotate(f"mean {np.mean(values):.3f}", (i, max(values)), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8, color=TEXT_MUTED)
    ax.axhline(0, color=GRID, linewidth=1.0)
    ax.set_xticks(range(1, len(names) + 1), names)
    _style(ax, "mean reward per step (one point per episode)")
    seeds = eval_results["eval_seeds"]
    ax.set_title(f"{seeds[1] - seeds[0]} random episodes (objective + traffic), the same episodes for every controller",
                 color=TEXT_MUTED, fontsize=9, loc="left")
    fig.suptitle("Evaluation: trained policies vs baselines", color=TEXT, fontsize=12, x=0.01, ha="left")
    return fig


def main() -> None:
    if not EVAL_PATH.exists():
        raise SystemExit(f"{EVAL_PATH} missing; run `uv run python -m rl.evaluate` first")
    results = json.loads(EVAL_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "M02-training.png": plot_training(results),
        "M02-goal-gap.png": plot_goal_gap(results),
        "M02-objectives.png": plot_objectives(DETAIL_POLICY),
        "M02-eval.png": plot_eval(results),
    }
    for filename, fig in figures.items():
        fig.savefig(OUT_DIR / filename, dpi=120)
        print(f"wrote {OUT_DIR / filename}")


if __name__ == "__main__":
    main()
