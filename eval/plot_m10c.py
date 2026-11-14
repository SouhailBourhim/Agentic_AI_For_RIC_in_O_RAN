"""The figures of M10c' (run `eval.m10c --dev`, `--play` and `--score --final` first).

  uv run python -m eval.plot_m10c

Writes into docs/learn/img/:
- M10c-h1.png: (a) H1, m10_rl minus formula_est, pooled and per family, on the test set (filled, 95 % interval) and
  on dev (hollow, the expectation), with the equivalence band of +-0.05; (b) mean intent met per executor on the test
  set, pooled. The RL policy in the project's accent colour, the comparator in dark ink, the rest grey.
- M10c-training.png: the five seeds' validation checkpoints (reward and T1b on m10c_validation) against training
  steps, with the median seed marked; read from models/m10c/*_validation.json (local, git-ignored).
Reads docs/learn/data/M10c-result.json, M10c-dev.json and M10c-policy.json (committed).
"""

import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from eval.generator import M10C_FAMILIES
from eval.m10c import COMPARATOR, EQUIVALENCE, EXECUTORS, RL
from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from sim.plot import GRID, LINE_WIDTH, SURFACE, TEXT, TEXT_MUTED

DATA = Path("docs/learn/data")
MODELS = Path("models/m10c")
LABELS = {"all": "F1, F2, F3, F5", "f1": "F1 load surge", "f2": "F2 new goal", "f3": "F3 fast dynamics", "f5": "F5 easy"}
EXECUTOR_LABELS = {"static": "static", "v7": "v7", "rev4": "rev 4", "formula_est": "formula_est", "mpc_est": "mpc_est",
                   "m10_rl": "m10 RL", "formula_told": "formula\ntold", "oracle": "oracle"}


def colour(executor: str) -> str:
    return ACCENT if executor == RL else TEXT if executor == COMPARATOR else NEUTRAL


def h1_rows(result: dict) -> list[tuple[str, float, float, float]]:
    rows = [("all", result["h1"]["mean"], *result["h1"]["ci95"])]
    return rows + [(f, v["mean"], *v["ci95"]) for f, v in result["h1_by_family"].items()]


def plot_h1() -> Figure:
    test = json.loads((DATA / "M10c-result.json").read_text())
    dev = json.loads((DATA / "M10c-dev.json").read_text())
    fig = Figure(figsize=(12.5, 4.0), facecolor=SURFACE, layout="constrained")
    diff_ax, means_ax = fig.subplots(1, 2, width_ratios=[1.1, 1])
    groups = ["all", *M10C_FAMILIES]
    y = np.arange(len(groups))[::-1]
    diff_ax.axvspan(-EQUIVALENCE, EQUIVALENCE, color=GRID, alpha=0.6, linewidth=0, label="equivalence: +-0.05")
    diff_ax.axvline(0, color=TEXT_MUTED, linewidth=0.8)
    for offset, result, filled, label in ((0.12, test, True, "test (decides)"), (-0.12, dev, False, "dev (expectation)")):
        for k, (_, mean, low, high) in enumerate(h1_rows(result)):
            diff_ax.hlines(y[k] + offset, low, high, color=ACCENT if filled else NEUTRAL, linewidth=2)
            diff_ax.scatter([mean], [y[k] + offset], s=40, zorder=3, label=label if k == 0 else None,
                            facecolors=ACCENT if filled else SURFACE, edgecolors=ACCENT if filled else NEUTRAL,
                            linewidths=1.5)
    diff_ax.set_yticks(y, [LABELS[g] for g in groups], fontsize=8.5)
    diff_ax.set_xlabel("m10 RL minus formula_est, mean intent met (95 % interval)", color=TEXT_MUTED, fontsize=8.5)
    diff_ax.legend(frameon=False, fontsize=8, loc="lower right")
    titled(diff_ax, f"(a) H1: {test['h1']['reading']} ({test['intents']} test intents)")

    x = np.arange(len(EXECUTORS))
    means = [test["means"][e]["all"] for e in EXECUTORS]
    means_ax.vlines(x, 0, means, color=GRID, linewidth=1.2, zorder=1)
    means_ax.scatter(x, means, s=46, color=[colour(e) for e in EXECUTORS], zorder=3)
    for k, m in enumerate(means):
        means_ax.annotate(f"{m:.2f}", (k, m), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5,
                          color=TEXT)
    means_ax.set_xticks(x, [EXECUTOR_LABELS[e] for e in EXECUTORS], fontsize=7.5)
    means_ax.set_ylim(0, 1.1)
    means_ax.set_ylabel("mean intent met", color=TEXT_MUTED, fontsize=8.5)
    titled(means_ax, "(b) every executor on the test set, F1-F3 and F5 pooled")
    return fig


def plot_training() -> Figure:
    chosen = json.loads((DATA / "M10c-policy.json").read_text())
    fig = Figure(figsize=(12, 3.6), facecolor=SURFACE, layout="constrained")
    reward_ax, t1b_ax = fig.subplots(1, 2)
    for path in sorted(MODELS.glob("td3_m10c_seed*_validation.json")):
        name = path.name.removesuffix("_validation.json")
        history = json.loads(path.read_text())
        steps = [m["step"] / 1000 for m in history]
        median = name == chosen["median_seed"]
        style = {"color": ACCENT if median else NEUTRAL, "linewidth": LINE_WIDTH if median else 1.0,
                 "alpha": 1.0 if median else 0.7}
        reward_ax.plot(steps, [m["reward"] for m in history], **style)
        t1b_ax.plot(steps, [m["t1b_captured"] for m in history], **style)
        seed = name.split("seed")[-1]
        reward_ax.annotate(f"seed {seed}" + (" (median)" if median else ""), (steps[-1], history[-1]["reward"]),
                           xytext=(4, 0), textcoords="offset points", fontsize=7.5, va="center",
                           color=TEXT if median else TEXT_MUTED)
        equal = history[-1]["equal_split_reward"]
    reward_ax.axhline(equal, color=TEXT_MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    reward_ax.text(52, equal, "equal split", fontsize=7.5, color=TEXT_MUTED, va="bottom")
    t1b_ax.axhline(0.5, color=TEXT_MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    t1b_ax.text(52, 0.5, "T1b rule 0.5", fontsize=7.5, color=TEXT_MUTED, va="bottom")
    for ax, title in ((reward_ax, "(a) validation reward per checkpoint"), (t1b_ax, "(b) T1b captured per checkpoint")):
        ax.set_xlabel("training steps (thousands)", color=TEXT_MUTED, fontsize=8.5)
        titled(ax, title)
    reward_ax.set_xlim(right=reward_ax.get_xlim()[1] * 1.12)
    return fig


def main(out_dir: Path = IMG) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in (("M10c-h1.png", plot_h1()), ("M10c-training.png", plot_training())):
        fig.savefig(out_dir / name, dpi=150, facecolor=SURFACE)
        print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
