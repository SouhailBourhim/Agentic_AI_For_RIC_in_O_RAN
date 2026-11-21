"""Figures for M11b/M11b′, the runtime monitor with formula fallback.

  uv run python -m eval.plot_m11b

1. docs/learn/img/M11b-test.png: the test split (m11b_test, played once): mean intent met per family for the RL
   alone, the RL with the monitor and the formula alone, on both cells.
2. docs/learn/img/M11b-revisions.png: rules A and B through the revisions: the worst family's gap against each
   rule's reference, with the rule's margin.
3. docs/learn/img/M11b-timeline.png: one dev instance (m09 F4, two overlapping intents) where the monitor fires:
   best effort's throughput against its target, for the three labels, M11b′'s final trigger.
4. docs/learn/img/M11b-testbed.png: the paired S3 check on the testbed: intent 1 met per repeat for static, LLM + RL
   with and without the monitor, and best effort's throughput over time in the median repeat.
Reads committed tables (docs/learn/data/M11b-*) and, for figures 3 and 4, the raw runs (eval/results/m11b/r6,
eval/results/m11b_testbed), which are git-ignored.
"""

import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from eval import m11b, m11b_testbed
from eval.testbed_scenarios import SCENARIOS
from eval.generator import dev_instances
from eval.metrics import load_run
from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from sim.plot import GRID, SURFACE, TEXT, TEXT_MUTED

DATA = Path("docs/learn/data")
LABELS = {"rl": "RL alone", "rl_mon": "RL + monitor", "formula_est": "formula alone"}
COLOURS = {"rl": "#b4b2ac", "rl_mon": ACCENT, "formula_est": NEUTRAL}
CELL_TITLES = {"m10": "m10 cell (radio channel), M10c′ policy", "m09": "m09 cell, v7 policy"}
# Each scored set, in the order it was played: (file, label under the axis)
STAGES = [("M11b-dev-r0.json", "r0\nall dev"), ("M11b-dev-r1.json", "r1\nall dev"), ("M11b-dev-r2.json", "r2\nall dev"),
          ("M11b-dev-r3-even.json", "r3\neven"), ("M11b-dev-r4-even.json", "r4\neven"),
          ("M11b-dev-r5-even.json", "r5\neven"), ("M11b-dev-r6-even.json", "r6\neven"),
          ("M11b-dev-r6-odd.json", "r6\nodd"), ("M11b-test.json", "r6\ntest")]


def plot_test() -> Figure:
    result = json.loads((DATA / "M11b-test.json").read_text())
    fig = Figure(figsize=(12, 3.8), facecolor=SURFACE, layout="constrained")
    for ax, cell in zip(fig.subplots(1, 2, sharey=True), ("m10", "m09"), strict=True):
        means = result[cell]["means"]
        groups = list(means)  # families, then "all"
        x = np.arange(len(groups))
        for k, label in enumerate(LABELS):
            values = [means[g][label] for g in groups]
            ax.scatter(x + (k - 1) * 0.22, values, s=40, color=COLOURS[label], zorder=3, label=LABELS[label],
                       edgecolors=SURFACE, linewidths=1.5)
            if label == "rl_mon":
                for xi, v in zip(x, values, strict=True):
                    ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 8), textcoords="offset points", ha="center",
                                fontsize=7.5, color=TEXT)
        ax.set_xticks(x, [g.upper() if g != "all" else "all" for g in groups], fontsize=8.5)
        ax.set_ylim(0.4, 1.08)
        titled(ax, f"{CELL_TITLES[cell]} ({result[cell]['runs']} instances, fired in {result[cell]['runs_with_fallback']})")
    fig.axes[0].set_ylabel("mean intent met (accepted intents)", color=TEXT_MUTED, fontsize=8.5)
    fig.axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    return fig


def worst_gaps(result: dict, cell: str) -> tuple[float, float | None]:
    """A: the worst family's rl_mon - formula_est; B: the worst rl_mon - rl where rl >= formula_est (None if none)."""
    means = {f: m for f, m in result[cell]["means"].items() if f != "all"}
    a = min(m["rl_mon"] - m["formula_est"] for m in means.values())
    b = [m["rl_mon"] - m["rl"] for m in means.values() if m["rl"] >= m["formula_est"]]
    return a, (min(b) if b else None)


def plot_revisions() -> Figure:
    fig = Figure(figsize=(12, 3.9), facecolor=SURFACE, layout="constrained")
    x = np.arange(len(STAGES))
    for ax, cell in zip(fig.subplots(1, 2, sharey=True), ("m10", "m09"), strict=True):
        gaps = [worst_gaps(json.loads((DATA / name).read_text()), cell) for name, _ in STAGES]
        a = [g[0] for g in gaps]
        b = [np.nan if g[1] is None else g[1] for g in gaps]
        ax.axhline(-m11b.PROMISE_MARGIN, color=ACCENT, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.axhline(-m11b.NO_HARM_MARGIN, color=NEUTRAL, linewidth=1.0, linestyle=(0, (1, 2)))
        ax.axhline(0, color=GRID, linewidth=1.0)
        ax.plot(x, a, color=ACCENT, linewidth=1.6, marker="o", markersize=5, label="A: worst family, RL + monitor - formula")
        ax.plot(x, b, color=NEUTRAL, linewidth=1.6, marker="s", markersize=5,
                label="B: worst family, RL + monitor - RL alone (where RL alone >= formula)")
        ax.annotate("A's margin -0.05", (-0.35, -m11b.PROMISE_MARGIN), xytext=(0, -11),
                    textcoords="offset points", fontsize=7.5, color=TEXT_MUTED)
        ax.annotate("B's margin -0.02", (-0.35, -m11b.NO_HARM_MARGIN), xytext=(0, -10),
                    textcoords="offset points", fontsize=7.5, color=TEXT_MUTED)
        ax.axvspan(-0.4, 2.5, color=GRID, alpha=0.35, zorder=0)
        ax.annotate("one revision allowed", (0, 0.06), fontsize=7.5, color=TEXT_MUTED)
        ax.set_xticks(x, [label for _, label in STAGES], fontsize=7.5)
        ax.set_ylim(-0.45, 0.1)
        titled(ax, CELL_TITLES[cell])
    fig.axes[0].set_ylabel("difference in mean intent met", color=TEXT_MUTED, fontsize=8.5)
    fig.axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    return fig


def plot_timeline(instance: str = "f4_08") -> Figure:
    cell = m11b.cells()["m09"]
    inst = next(i for i in dev_instances(cell["families"]) if i.name == instance)
    intent = inst.scenario().intents[0]
    root = m11b.ROOT / "r6"
    fig = Figure(figsize=(12, 3.6), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    fired = []
    for label in LABELS:
        work = root / "m09" / m11b.spec_of(cell, inst, label).run_id
        kpis, events = load_run(work)
        be = kpis["be"]
        ax.plot(be["t"], be["thr"], color=COLOURS[label], linewidth=1.6 if label == "rl_mon" else 1.2,
                label=LABELS[label], zorder=3 if label == "rl_mon" else 2)
        if label == "rl_mon":
            fired = [e for e in events if e["event"] in ("executor_fallback", "executor_resume")]
            target = np.minimum(intent.targets["be"], be["load"])
            start, end = intent.window
            inside = (be["t"] >= start) & (be["t"] <= end)
            ax.plot(be["t"][inside], target[inside], color=TEXT, linewidth=1.0, linestyle=(0, (4, 3)),
                    label=f"target: min({intent.targets['be']:g} Mb/s, load)")
    for e in fired:
        ax.axvline(e["ts_backend"], color=ACCENT, linewidth=1.0, linestyle=(0, (1, 2)))
        text = "formula takes over" if e["event"] == "executor_fallback" else "intents over: RL again"
        ax.annotate(text, (e["ts_backend"], ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                    fontsize=7.5, color=TEXT_MUTED)
    for later in inst.scenario().intents[1:]:
        ax.axvline(later.at_s, color=TEXT_MUTED, linewidth=0.8, linestyle=(0, (1, 2)))
        ax.annotate("2nd intent (URLLC) starts: a new segment, the formula keeps it", (later.at_s, 0.04),
                    xycoords=ax.get_xaxis_transform(), xytext=(4, 0), textcoords="offset points", fontsize=7.5,
                    color=TEXT_MUTED)
    ax.set_xlabel("time (s)", color=TEXT_MUTED, fontsize=8.5)
    ax.set_ylabel("best effort throughput (Mb/s)", color=TEXT_MUTED, fontsize=8.5)
    titled(ax, f"m09 {instance} (dev): \"{intent.text}\"")
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.1), ncols=4)
    return fig


TESTBED_LABELS = {"static": "static (equal split)", "mon": "LLM + RL, monitor", "nomon": "LLM + RL, no monitor"}
TESTBED_COLOURS = {"static": NEUTRAL, "mon": ACCENT, "nomon": "#b4b2ac"}
TESTBED_MARKERS = {"static": "s", "mon": "o", "nomon": "D"}


def plot_testbed() -> Figure:
    """Left: S3 intent 1 met per repeat (committed score). Right: best effort over time in the median repeat."""
    result = json.loads((DATA / "M11b-testbed.json").read_text())
    repeats = [k for k in result if k.startswith("repeat")]
    diffs = {k: result[k]["mon"]["met"] - result[k]["nomon"]["met"] for k in repeats}
    median_repeat = sorted(repeats, key=diffs.get)[len(repeats) // 2]
    fig = Figure(figsize=(12.5, 3.8), facecolor=SURFACE, layout="constrained")
    met_ax, thr_ax = fig.subplots(1, 2, width_ratios=(1, 2.4))
    x = np.arange(len(repeats))
    for k, arm in enumerate(TESTBED_LABELS):
        met_ax.scatter(x + (k - 1) * 0.18, [result[r][arm]["met"] for r in repeats], s=40, zorder=3,
                       color=TESTBED_COLOURS[arm], marker=TESTBED_MARKERS[arm], label=TESTBED_LABELS[arm],
                       edgecolors=SURFACE, linewidths=1.2)
    met_ax.set_xticks(x, [f"repeat {r.removeprefix('repeat')}" for r in repeats], fontsize=8.5)
    met_ax.set_ylim(0.5, 1.05)
    met_ax.set_ylabel("S3 intent 1 met", color=TEXT_MUTED, fontsize=8.5)
    titled(met_ax, f"median of mon − nomon: {result['verdict']['median']:+.3f}")
    met_ax.legend(frameon=False, fontsize=8, loc="lower left")
    intent = SCENARIOS["s3"].intents[0]
    target = intent.expected["min_thr_mbps"]["be"]
    start, end = intent.at_s, intent.at_s + intent.expected["duration_s"]
    for arm in ("nomon", "mon"):
        study, controller, _ = next(a for a in m11b_testbed.ARMS if a[0] == arm)
        work = m11b_testbed.RESULTS / m11b_testbed.run_of(study, controller, int(median_repeat[-1])).run_id
        kpis, events = load_run(work)
        be = kpis["be"]
        thr_ax.plot(be["t"], be["thr"], color=TESTBED_COLOURS[arm], linewidth=1.6 if arm == "mon" else 1.2,
                    label=TESTBED_LABELS[arm], zorder=3 if arm == "mon" else 2)
        for e in events if arm == "mon" else []:
            if e["event"] in ("executor_fallback", "executor_resume"):
                thr_ax.axvline(e["ts_backend"], color=ACCENT, linewidth=1.0, linestyle=(0, (1, 2)))
                text = "formula takes over" if e["event"] == "executor_fallback" else "intent over: RL again"
                thr_ax.annotate(text, (e["ts_backend"], 0.96), xycoords=thr_ax.get_xaxis_transform(), xytext=(4, 0),
                                textcoords="offset points", fontsize=7.5, color=TEXT_MUTED)
    thr_ax.hlines(target, start, end, color=TEXT, linewidth=1.0, linestyles=(0, (4, 3)),
                  label=f"stated target {target:g} Mb/s")
    thr_ax.axvspan(start, end, color=GRID, alpha=0.15, zorder=0)
    thr_ax.set_xlabel("time (s)", color=TEXT_MUTED, fontsize=8.5)
    thr_ax.set_ylabel("best effort throughput (Mb/s)", color=TEXT_MUTED, fontsize=8.5)
    titled(thr_ax, f"testbed S3, {median_repeat.replace('repeat', 'repeat ')} (median): \"{intent.text}\"")
    thr_ax.legend(frameon=False, fontsize=8, loc="lower right", ncols=3)
    return fig


def main() -> None:
    for name, fig in (("M11b-test.png", plot_test()), ("M11b-revisions.png", plot_revisions()),
                      ("M11b-timeline.png", plot_timeline()), ("M11b-testbed.png", plot_testbed())):
        fig.savefig(IMG / name, dpi=150, facecolor=SURFACE)
        print(IMG / name)


if __name__ == "__main__":
    main()
