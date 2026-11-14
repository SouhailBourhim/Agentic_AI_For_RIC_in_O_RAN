"""The figures of M09i, local against remote LLM (run `eval.m09i --table` first).

  uv run python -m eval.plot_m09i

Writes, from the committed tables only (docs/learn/data/M09i-{summary,runs,steps}.csv):
- docs/learn/img/M09i-comparison.png: left, parse accuracy per split with its Wilson 95 % interval; right, remote
  minus local intent met in the closed loop per controller, with the cluster-bootstrap 95 % interval.
- docs/learn/img/M09i-latency.png: left, the share of LLM calls answered within x seconds (log x), with the client's
  cumulative retry waits marked; right, LLM + RL, one dot per intent: how much later the remote model's objective
  was applied, against how much intent met it lost (both models applied the same objective on every intent).
Grey is the local model (qwen3:8b), purple the remote one (Nemotron 3 Ultra on NIM's free tier).
"""

import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from agent.llm import RETRY_WAITS_S
from eval.m09i import CONTROLLERS, LOCAL, REMOTE, RUNS_CSV, SPLITS, STEPS_CSV, SUMMARY_CSV
from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from sim.plot import SURFACE, TEXT, TEXT_MUTED

MODEL_LABELS = {LOCAL: "qwen3:8b, local", REMOTE: "Nemotron 3 Ultra, remote"}
COLOURS = {LOCAL: NEUTRAL, REMOTE: ACCENT}
CONTROLLER_LABELS = {"proposed": "LLM + RL", "llm_only_v2": "LLM only v2"}
SPLIT_LABELS = {"tune": "tune split (20)", "heldout": "held-out split (20)"}
LATE_S = 60.0  # beyond 4 retries' waits (30 s): only rate-limit storms make a decision this late


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_comparison(summary: list[dict]) -> Figure:
    fig = Figure(figsize=(11, 3.4), facecolor=SURFACE, layout="constrained")
    parse_ax, loop_ax = fig.subplots(1, 2, width_ratios=[1, 1])
    for y, split in enumerate(SPLITS):
        for k, model in enumerate((LOCAL, REMOTE)):
            [r] = [r for r in summary if r["part"] == "parse" and r["model"] == model and r["split_or_controller"] == split]
            v, lo, hi = (100 * float(r[c]) for c in ("value", "low_95", "high_95"))
            yy = y + (0.15 if k == 0 else -0.15)
            parse_ax.hlines(yy, lo, hi, color=COLOURS[model], linewidth=1.6)
            parse_ax.scatter([v], [yy], s=48, color=COLOURS[model], edgecolors=SURFACE, linewidths=1.5, zorder=3,
                             label=MODEL_LABELS[model] if y == 0 else None)
            parse_ax.text(lo - 0.4, yy, f"{v:.0f} %", ha="right", va="center", fontsize=8, color=TEXT_MUTED)
    parse_ax.set_yticks(range(len(SPLITS)), [SPLIT_LABELS[s] for s in SPLITS], fontsize=8.5)
    parse_ax.set_ylim(-0.6, len(SPLITS) - 0.4)
    parse_ax.invert_yaxis()
    parse_ax.set_xlim(84, 101)
    titled(parse_ax, "Parse step: intents right, 5 repeats pooled (Wilson 95 %)")
    parse_ax.set_xlabel("intents right, %", color=TEXT_MUTED, fontsize=8.5)

    for y, controller in enumerate(CONTROLLERS):
        [r] = [r for r in summary if r["part"] == "loop, remote minus local" and r["split_or_controller"] == controller]
        v, lo, hi = (100 * float(r[c]) for c in ("value", "low_95", "high_95"))
        loop_ax.hlines(y, lo, hi, color=ACCENT, linewidth=1.6)
        loop_ax.scatter([v], [y], s=48, color=ACCENT, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        loop_ax.text(v, y - 0.22, f"{v:+.1f} points [{lo:+.1f}, {hi:+.1f}]", ha="center", fontsize=8, color=TEXT)
    loop_ax.axvline(0, color=TEXT_MUTED, linewidth=1, linestyle=":")
    loop_ax.text(0.4, len(CONTROLLERS) - 0.55, "remote better →", fontsize=8, color=TEXT_MUTED)
    loop_ax.text(-0.4, len(CONTROLLERS) - 0.55, "← local better", fontsize=8, color=TEXT_MUTED, ha="right")
    loop_ax.set_yticks(range(len(CONTROLLERS)), [CONTROLLER_LABELS[c] for c in CONTROLLERS], fontsize=8.5)
    loop_ax.set_ylim(-0.6, len(CONTROLLERS) - 0.3)
    loop_ax.invert_yaxis()
    loop_ax.set_xlim(-30, 8)
    titled(loop_ax, "Closed loop: remote minus local intent met (25 instances, 95 %)")
    loop_ax.set_xlabel("intent met, points of %", color=TEXT_MUTED, fontsize=8.5)
    fig.legend(loc="outside lower center", ncols=2, frameon=False, fontsize=8.5, labelcolor=TEXT)
    fig.suptitle("M09i: the remote model parses at least as well, and loses in the loop", color=TEXT, fontsize=11,
                 x=0.01, ha="left")
    return fig


def plot_latency(steps: list[dict], runs: list[dict]) -> Figure:
    fig = Figure(figsize=(11, 3.8), facecolor=SURFACE, layout="constrained")
    cdf_ax, cost_ax = fig.subplots(1, 2)
    for model in (LOCAL, REMOTE):
        lat = np.sort([float(r["latency_s"]) for r in steps if r["model"] == model])
        share = np.arange(1, len(lat) + 1) / len(lat)
        cdf_ax.step(lat, 100 * share, where="post", color=COLOURS[model], linewidth=1.6,
                    label=f"{MODEL_LABELS[model]} ({len(lat)} calls)")
    waited = np.cumsum(RETRY_WAITS_S)[1:]  # 6, 14, 30, 60, 90 s: the clusters a 429 storm leaves
    for k, w in enumerate(waited, start=2):
        cdf_ax.axvline(w, color=TEXT_MUTED, linewidth=0.8, linestyle=":")
        cdf_ax.text(w * 1.04, 2, f"{k} retries", rotation=90, fontsize=7, color=TEXT_MUTED, va="bottom")
    cdf_ax.set_xscale("log")
    cdf_ax.set_xlim(0.5, 150)
    cdf_ax.set_ylim(0, 102)
    titled(cdf_ax, "Closed loop: share of LLM calls answered within x s")
    cdf_ax.set_xlabel("call latency, s (log), retry waits included", color=TEXT_MUTED, fontsize=8.5)
    cdf_ax.set_ylabel("calls, %", color=TEXT_MUTED)
    cdf_ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), frameon=True, facecolor=SURFACE, edgecolor=SURFACE,
                  framealpha=1, fontsize=8, labelcolor=TEXT)

    mine = {(r["model"], r["instance"], r["intent"]): r for r in runs if r["controller"] == "proposed"}
    xs, ys = [], []
    for (model, instance, intent), remote in mine.items():
        local = mine.get((LOCAL, instance, intent))
        if model != REMOTE or local is None or "" in (remote["decision_s"], local["decision_s"]):
            continue
        xs.append(float(remote["decision_s"]) - float(local["decision_s"]))
        ys.append(100 * (float(remote["intent_met"]) - float(local["intent_met"])))
    cost_ax.scatter(xs, ys, s=40, color=ACCENT, alpha=0.8, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    cost_ax.axhline(0, color=TEXT_MUTED, linewidth=1, linestyle=":")
    cost_ax.axvline(LATE_S, color=TEXT_MUTED, linewidth=0.8, linestyle=":")
    late = [y for x, y in zip(xs, ys) if x >= LATE_S]
    share = sum(late) / sum(ys) if sum(ys) < 0 else 0.0
    cost_ax.text(LATE_S + 2, max(ys), f"{len(late)} intents {LATE_S:.0f}+ s late:\n{share:.0%} of the summed loss",
                 fontsize=8, color=TEXT, va="top")
    titled(cost_ax, f"LLM + RL, {len(xs)} intents, the same objective applied by both models")
    cost_ax.set_xlabel("remote applied it later by, simulated s", color=TEXT_MUTED, fontsize=8.5)
    cost_ax.set_ylabel("remote minus local intent met, points", color=TEXT_MUTED)
    fig.suptitle("M09i: the remote model's tail is the free tier's rate limit, and the loop pays for it", color=TEXT,
                 fontsize=11, x=0.01, ha="left")
    return fig


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    summary, runs, steps = read(SUMMARY_CSV), read(RUNS_CSV), read(STEPS_CSV)
    for name, fig in (("M09i-comparison.png", plot_comparison(summary)), ("M09i-latency.png", plot_latency(steps, runs))):
        fig.savefig(IMG / name, dpi=150)
        print(IMG / name)


if __name__ == "__main__":
    main()
