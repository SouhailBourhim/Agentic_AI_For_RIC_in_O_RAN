"""The figures of the control evaluation, from the runs on disk (M08a; run eval.control first).

  uv run python -m eval.plot_control

Controllers are compared by position with the proposed one highlighted, so no colour stands for a
controller; colours keep their project-wide meaning, the slices. Every dot is one seed.
Writes docs/learn/img/M08a-*.png.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from eval.control import RESULTS, TABLE
from eval.metrics import load_run
from eval.scenarios import SCENARIOS, SEEDS, Scenario
from sim.plot import (
    LINE_WIDTH,
    SLICE_COLOURS,
    SLICE_LABELS,
    SURFACE,
    TEXT,
    TEXT_MUTED,
    _style,
)
from store.events import read_events
from verifier.objective import SLICES

IMG = Path("docs/learn/img")
ORDER = ["static", "rl_only", "llm_only", "proposed"]
LABELS = {"static": "Static", "rl_only": "RL only", "llm_only": "LLM only", "proposed": "LLM + RL"}
NEUTRAL, ACCENT = "#6b6a66", "#6250d6"  # the proposed controller stands out; nothing else needs a colour
INTENT_PANELS = [("s1", 1, "S1: URLLC 20 Mb/s, < 10 ms"), ("s2", 1, "S2: URLLC 18 Mb/s, < 5 ms"),
                 ("s3", 1, "S3 first: eMBB and BE 22 Mb/s"), ("s3", 2, "S3 second: URLLC 22 Mb/s, < 5 ms")]


def read_table() -> list[dict]:
    return list(csv.DictReader(TABLE.open()))


def values(rows: list[dict], key: str, **where) -> list[float]:
    """The column's numbers in the rows matching `where`; empty cells are left out."""
    return [float(r[key]) for r in rows if all(r[k] == str(v) for k, v in where.items()) and r[key] != ""]


def strip(ax: Axes, x: float, ys: list[float], colour: str, offset_seed: int = 0) -> None:
    """One dot per seed with a small sideways spread, and the median as a short bar."""
    if not ys:
        return
    jitter = np.random.default_rng(offset_seed).uniform(-0.13, 0.13, len(ys))
    ax.scatter(x + jitter, ys, s=26, color=colour, alpha=0.75, linewidths=0.8, edgecolors=SURFACE, zorder=3)
    ax.hlines(float(np.median(ys)), x - 0.26, x + 0.26, color=TEXT, linewidth=1.6, zorder=4)


def titled(ax: Axes, title: str) -> None:
    """The project's axis style (its text argument is the y label), with a panel title on top."""
    _style(ax, "")
    ax.set_title(title, loc="left", color=TEXT, fontsize=9.5)


def controller_axis(ax: Axes) -> None:
    ax.set_xticks(range(len(ORDER)), [LABELS[c] for c in ORDER], fontsize=8.5)
    ax.set_xlim(-0.6, len(ORDER) - 0.4)


def plot_met(rows: list[dict]) -> Figure:
    fig = Figure(figsize=(12, 3.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 4, sharey=True)
    for ax, (scenario, intent, title) in zip(axes, INTENT_PANELS):
        for x, c in enumerate(ORDER):
            ys = values(rows, "intent_met", study="headline", scenario=scenario, intent=intent, controller=c)
            strip(ax, x, [y * 100 for y in ys], ACCENT if c == "proposed" else NEUTRAL, x)
        controller_axis(ax)
        ax.set_ylim(-4, 104)
        titled(ax, title)
    axes[0].set_ylabel("seconds of the window met, %", color=TEXT_MUTED)
    fig.suptitle("Is the operator's stated goal met? One dot per seed, bar = median", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def plot_reaction(rows: list[dict]) -> Figure:
    fig = Figure(figsize=(12, 3.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 4, sharey=True)
    for ax, (scenario, intent, title) in zip(axes, INTENT_PANELS):
        for x, c in enumerate(ORDER):
            where = {"study": "headline", "scenario": scenario, "intent": intent, "controller": c}
            ys = values(rows, "reaction_s", **where)
            strip(ax, x, ys, ACCENT if c == "proposed" else NEUTRAL, x)
            never = len(values(rows, "intent_met", **where)) - len(ys)
            if never:
                ax.text(x, 245, f"{never}/10\nnever", ha="center", va="top", fontsize=8, color=TEXT_MUTED)
        controller_axis(ax)
        ax.set_ylim(-8, 250)
        titled(ax, title)
    axes[0].set_ylabel("seconds until held for 5 s", color=TEXT_MUTED)
    fig.suptitle("Reaction: from the spike (S1) or the submission (S2, S3)", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def plot_cost(rows: list[dict]) -> Figure:
    """Traffic each slice was refused during the intent: what a goal costs, and whom."""
    fig = Figure(figsize=(12, 3.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 3, sharey=True)
    for ax, (scenario, title) in zip(axes, [("s1", "S1"), ("s2", "S2"), ("s3", "S3, first intent's window")]):
        for x, c in enumerate(ORDER):
            for i, s in enumerate(SLICES):
                where = {"study": "headline", "scenario": scenario, "intent": 1, "controller": c}
                ys = values(rows, f"{s}_unmet", **where)
                jitter = np.random.default_rng(x + i).uniform(-0.05, 0.05, len(ys))
                ax.scatter(x + (i - 1) * 0.24 + jitter, ys, s=20, color=SLICE_COLOURS[s], alpha=0.8, linewidths=0.6,
                           edgecolors=SURFACE, zorder=3, label=SLICE_LABELS[s] if x == 0 else None)
        controller_axis(ax)
        ax.set_ylim(-1, 24)
        titled(ax, title)
    axes[0].set_ylabel("offered but not carried, Mb/s", color=TEXT_MUTED)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left", ncols=3)
    fig.suptitle("Who pays: mean unmet demand per slice during the intent", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def series(scenario: Scenario, controller: str) -> dict[str, dict[str, np.ndarray]]:
    """Per slice and quantity: an array seeds x seconds, from the raw runs."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for seed in SEEDS:
        kpis, _ = load_run(RESULTS / "headline" / scenario.name / controller / f"seed{seed}")
        for s in SLICES:
            for key in ("t", "thr", "share", "lat", "load"):
                out[s][key].append(kpis[s][key])
    return {s: {k: np.array(v) for k, v in q.items()} for s, q in out.items()}


def band(ax: Axes, t: np.ndarray, matrix: np.ndarray, colour: str, label: str | None = None) -> None:
    """Median over the seeds, and their min-max as a light band."""
    ax.fill_between(t, matrix.min(axis=0), matrix.max(axis=0), color=colour, alpha=0.18, linewidth=0)
    ax.plot(t, np.median(matrix, axis=0), color=colour, linewidth=LINE_WIDTH, label=label)


def plot_timeline(scenario: Scenario) -> Figure:
    """One column per controller: split, throughput with its targets, URLLC delay with its bound."""
    fig = Figure(figsize=(13, 7.2), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(3, len(ORDER), sharex=True, sharey="row")
    for col, controller in enumerate(ORDER):
        data = series(scenario, controller)
        t = data["urllc"]["t"][0]
        for s in SLICES:
            band(axes[0, col], t, data[s]["share"], SLICE_COLOURS[s], SLICE_LABELS[s])
            band(axes[1, col], t, data[s]["thr"], SLICE_COLOURS[s])
        band(axes[2, col], t, data["urllc"]["lat"], SLICE_COLOURS["urllc"])
        axes[2, col].set_yscale("log")
        for intent in scenario.intents:
            start, end = intent.window
            for s, target in intent.targets.items():
                axes[1, col].hlines(target, start, end, color=SLICE_COLOURS[s], linewidth=1, linestyle=":")
            if intent.max_latency_ms is not None:
                axes[2, col].hlines(intent.max_latency_ms, start, end, color=TEXT, linewidth=1, linestyle=":")
            for row in range(3):
                axes[row, col].axvline(intent.at_s, color=TEXT_MUTED, linewidth=0.9, linestyle="--")
        titled(axes[0, col], LABELS[controller])
        _style(axes[1, col], "")
        _style(axes[2, col], "")
        axes[2, col].set_xlabel("time, s", color=TEXT_MUTED)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("PRB share", color=TEXT_MUTED)
    axes[1, 0].set_ylabel("throughput, Mb/s", color=TEXT_MUTED)
    axes[2, 0].set_ylabel("URLLC delay, ms (log)", color=TEXT_MUTED)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left", ncols=3)
    fig.suptitle(f"{scenario.name.upper()}, {scenario.title}: median over 10 seeds, band = min-max. Dashed: an intent "
                 "is submitted; dotted: its target or bound", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def plot_latency() -> Figure:
    """Every LLM step of every headline run, per node (retries inside a step are added up)."""
    fig = Figure(figsize=(8, 3.4), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    nodes = [("llm_only", "parse"), ("llm_only", "plan"), ("proposed", "parse"), ("proposed", "plan"), ("proposed", "reflect")]
    for x, (controller, node) in enumerate(nodes):
        ys = [float(e["data"]["latency_s"])
              for path in (RESULTS / "headline").glob(f"*/{controller}/seed*/events.jsonl")
              for e in read_events(path)
              if e["event"] == "agent_step" and e["data"].get("node") == node and e["data"].get("latency_s")]
        strip(ax, x, ys, ACCENT if controller == "proposed" else NEUTRAL, x)
    ax.set_xticks(range(len(nodes)), [f"{LABELS[c]}\n{n}" for c, n in nodes], fontsize=8.5)
    ax.set_ylim(bottom=0)
    titled(ax, "How long the LLM takes per step (each dot is one step)")
    ax.set_ylabel("seconds per step", color=TEXT_MUTED)
    return fig


def plot_bridge(rows: list[dict]) -> Figure:
    """Simulator settings towards testbed settings, one chart per measure."""
    settings = [("0.25", "0", "headline"), ("0.25", "2", "bridge"), ("0.1", "0", "bridge"), ("0.1", "2", "bridge")]
    fig = Figure(figsize=(12, 3.6), facecolor=SURFACE, layout="constrained")
    moved, met = fig.subplots(1, 2)
    ticks = []
    for x, (alpha, delay, study) in enumerate(settings):
        where = {"study": study, "controller": "rl_only", "alpha": alpha, "delay_ticks": delay, "intent": 1}
        strip(moved, x, [v * 100 for v in values(rows, "moved_per_s", **where)], NEUTRAL, x)
        ticks.append(f"alpha {alpha}\n{delay} s late")
    moved.set_xticks(range(4), ticks, fontsize=8.5)
    moved.set_ylim(bottom=0)
    titled(moved, "RL only, all three scenarios: split movement")
    moved.set_ylabel("% of the cell moved per s", color=TEXT_MUTED)
    for x, scenario in enumerate(SCENARIOS):
        for shift, (study, alpha, delay, colour) in zip((-0.2, 0.2), (("headline", "0.25", "0", ACCENT), ("bridge", "0.1", "2", NEUTRAL))):
            ys = values(rows, "intent_met", study=study, scenario=scenario, intent=1, controller="proposed",
                        alpha=alpha, delay_ticks=delay)
            strip(met, x + shift, [y * 100 for y in ys], colour, x)
    met.set_xticks(range(3), ["S1", "S2", "S3 first intent"], fontsize=8.5)
    met.scatter([], [], color=ACCENT, s=26, label="alpha 0.25, no delay")
    met.scatter([], [], color=NEUTRAL, s=26, label="alpha 0.10, 2 s late")
    met.legend(frameon=False, fontsize=8.5, loc="lower left")
    met.set_ylim(0, 104)
    titled(met, "LLM + RL: intent met under each setting")
    met.set_ylabel("seconds of the window met, %", color=TEXT_MUTED)
    return fig


def spread(ys: list[float]) -> str:
    if not ys:
        return "-"
    q1, med, q3 = np.percentile(ys, [25, 50, 75])
    return f"{med:.2f} [{q1:.2f}-{q3:.2f}] {min(ys):.2f}-{max(ys):.2f}"


def print_summary(rows: list[dict]) -> None:
    """The tables of the note: median [quartiles] min-max over the seeds, and the paired differences."""
    print("\nIntent met; then LLM + RL minus each controller, seed by seed, median (min to max)")
    for scenario, intent, title in INTENT_PANELS:
        where = {"study": "headline", "scenario": scenario, "intent": intent}
        print(title)
        proposed = {r["seed"]: float(r["intent_met"]) for r in rows
                    if all(r[k] == str(v) for k, v in where.items()) and r["controller"] == "proposed"}
        for c in ORDER:
            mine = [r for r in rows if all(r[k] == str(v) for k, v in where.items()) and r["controller"] == c]
            gaps = [proposed[r["seed"]] - float(r["intent_met"]) for r in mine]
            print(f"  {LABELS[c]:9s} {spread([float(r['intent_met']) for r in mine])}   "
                  f"difference {np.median(gaps):+.2f} ({min(gaps):+.2f} to {max(gaps):+.2f})")
    print("\nPer controller, medians over the seeds: unmet Mb/s eMBB/URLLC/BE; URLLC share; p95 delay ms eMBB/URLLC/BE")
    for scenario, intent, title in INTENT_PANELS[:3]:
        print(title)
        for c in ORDER:
            w = {"study": "headline", "scenario": scenario, "intent": intent, "controller": c}
            m = lambda key: float(np.median(values(rows, key, **w)))  # noqa: B023
            print(f"  {LABELS[c]:9s} {m('embb_unmet'):5.1f} {m('urllc_unmet'):5.1f} {m('be_unmet'):5.1f}   "
                  f"{m('urllc_share'):.2f}   {m('embb_lat_p95'):7.0f} {m('urllc_lat_p95'):7.0f} {m('be_lat_p95'):7.0f}")
    print("\nSplit movement, % of the cell per s, median: whole run / with no intent in force; blocked proposals")
    for scenario in SCENARIOS:
        for c in ORDER:
            w = {"study": "headline", "scenario": scenario, "intent": 1, "controller": c}
            idle = values(rows, "moved_per_s_no_intent", **w)
            checks = sorted({r["blocked_checks"] for r in rows if all(r[k] == str(v) for k, v in w.items())})
            print(f"  {scenario} {LABELS[c]:9s} {np.median(values(rows, 'moved_per_s', **w)) * 100:5.1f} / "
                  f"{np.median(idle) * 100 if idle else float('nan'):5.1f}   proposals "
                  f"{np.median(values(rows, 'proposals', **w)):.0f}, blocked {checks}")
    print("\nOver-serving lever, S1, median: intent met; URLLC share; unmet eMBB, BE")
    for importance in ("6.0", "8.0"):
        w = {"study": "lever", "importance": importance}
        print(f"  importance {importance}: {np.median(values(rows, 'intent_met', **w)):.3f}  "
              f"{np.median(values(rows, 'urllc_share', **w)):.2f}  {np.median(values(rows, 'embb_unmet', **w)):.1f}  "
              f"{np.median(values(rows, 'be_unmet', **w)):.1f}")
    print("\nLLM + RL, simulator settings against testbed settings, intent met")
    for scenario, intent, title in INTENT_PANELS:
        print(f"  {title}: {spread(values(rows, 'intent_met', study='headline', scenario=scenario, intent=intent, controller='proposed'))}"
              f"  against  {spread(values(rows, 'intent_met', study='bridge', scenario=scenario, intent=intent, controller='proposed'))}")
    w = {"study": "bridge", "controller": "proposed", "intent": 1}
    print(f"  moved %/s at testbed settings, per scenario: "
          f"{[round(float(np.median(values(rows, 'moved_per_s', scenario=s, **w))) * 100, 1) for s in SCENARIOS]}")
    print("\nReal-process check, S2, against the lockstep range")
    for c in ("proposed", "llm_only"):
        for key in ("intent_met", "first_apply_s", "urllc_share", "moved_per_s", "embb_unmet", "be_unmet", "urllc_lat_p95"):
            real = values(rows, key, study="realtime", controller=c)
            lockstep = values(rows, key, study="headline", scenario="s2", controller=c)
            print(f"  {LABELS[c]:9s} {key:15s} {real[0]:8.3f}   lockstep {min(lockstep):.3f}-{max(lockstep):.3f}")
    finished = defaultdict(int)
    for path in (RESULTS / "headline").glob("*/llm_only/seed*/events.jsonl"):
        scenario = path.parts[-4]
        finished[scenario] += any(e["event"] == "agent_step" and e["data"].get("tool") == "finish" for e in read_events(path))
    print(f"\nLLM only: runs where the model called finish at least once, of 10: {dict(sorted(finished.items()))}")


def main() -> None:
    rows = read_table()
    IMG.mkdir(parents=True, exist_ok=True)
    figures = {"M08a-intent-met": plot_met(rows), "M08a-reaction": plot_reaction(rows), "M08a-cost": plot_cost(rows),
               "M08a-llm-latency": plot_latency(), "M08a-bridge": plot_bridge(rows)}
    figures |= {f"M08a-{name}": plot_timeline(scenario) for name, scenario in SCENARIOS.items()}
    for name, fig in figures.items():
        fig.savefig(IMG / f"{name}.png", dpi=120)
        print(f"wrote {IMG / name}.png")
    print_summary(rows)


if __name__ == "__main__":
    main()
