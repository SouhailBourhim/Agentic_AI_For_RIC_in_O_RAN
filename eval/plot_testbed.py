"""The figures of the control evaluation on the testbed (M08c; run eval.testbed_control --final first).

  uv run python -m eval.plot_testbed

Same conventions as eval.plot_control: controllers by position, LLM + RL highlighted, slices keep
their colours. Every dot is one repeat; a hollow dot is a run the grace-period check flagged.
Writes docs/learn/img/M08c-*.png and prints the tables of the note.
"""

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from eval.metrics import load_run
from eval.plot_control import ACCENT, IMG, LABELS, NEUTRAL, ORDER, band, controller_axis, titled, values
from eval.testbed_control import RESULTS, TABLE, TESTBED_CAPACITY_MBPS
from eval.testbed_scenarios import SCENARIOS, Scenario
from sim.plot import SLICE_COLOURS, SLICE_LABELS, SURFACE, TEXT, TEXT_MUTED, _style
from store.events import read_events
from verifier.objective import SLICES

SIM_TABLE = Path("docs/learn/data/M08a-runs.csv")
REPEATS = (1, 2, 3)
PANELS = [("s1", 1, "S1: URLLC 18 Mb/s, < 10 ms"), ("s2", 1, "S2: URLLC 16.5 Mb/s, < 5 ms"),
          ("s3", 1, "S3 first: eMBB and BE 20 Mb/s"), ("s3", 2, "S3 second: URLLC 20 Mb/s, < 5 ms")]
BACKLOGGED_MS = 200.0  # above this RLC delay a queue is standing, so the slice uses all it was given
GRID = np.arange(0.0, 360.0)
SIM_SERVED_CELL = 75.0  # what the simulator assumes, for the figure's reference line


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def work_of(scenario: str, controller: str, repeat: int) -> Path:
    return RESULTS / "testbed" / scenario / controller / f"repeat{repeat}"


def repeat_dots(ax: Axes, x: float, rows: list[dict], key: str, colour: str, scale: float = 1.0, **where) -> None:
    """One dot per repeat, hollow if flagged, and the median as a short bar: all runs count."""
    mine = [r for r in rows if all(r[k] == str(v) for k, v in where.items()) and r[key] != ""]
    if not mine:
        return
    ys = [float(r[key]) * scale for r in mine]
    offsets = np.linspace(-0.14, 0.14, len(ys))
    for offset, y, r in zip(offsets, ys, mine):
        clean = r["valid"] == "True"
        ax.scatter(x + offset, y, s=34, color=colour if clean else SURFACE, edgecolors=colour, linewidths=1.2, zorder=3)
    ax.hlines(float(np.median(ys)), x - 0.26, x + 0.26, color=TEXT, linewidth=1.6, zorder=4)


def plot_met(rows: list[dict]) -> Figure:
    fig = Figure(figsize=(12, 3.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 4, sharey=True)
    for ax, (scenario, intent, title) in zip(axes, PANELS):
        for x, c in enumerate(ORDER):
            repeat_dots(ax, x, rows, "intent_met", ACCENT if c == "proposed" else NEUTRAL, 100.0, scenario=scenario, intent=intent, controller=c)
        controller_axis(ax)
        ax.set_ylim(-4, 104)
        titled(ax, title)
    axes[0].set_ylabel("seconds of the window met, %", color=TEXT_MUTED)
    fig.suptitle("Is the operator's goal met on the real cell? One dot per repeat, hollow = flagged by the grace-period check, bar = median",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def plot_reaction(rows: list[dict]) -> Figure:
    fig = Figure(figsize=(12, 3.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 4, sharey=True)
    for ax, (scenario, intent, title) in zip(axes, PANELS):
        for x, c in enumerate(ORDER):
            where = {"scenario": scenario, "intent": intent, "controller": c}
            repeat_dots(ax, x, rows, "reaction_s", ACCENT if c == "proposed" else NEUTRAL, **where)
            never = len(values(rows, "intent_met", **where)) - len(values(rows, "reaction_s", **where))
            if never:
                ax.text(x, 245, f"{never}/3\nnever", ha="center", va="top", fontsize=8, color=TEXT_MUTED)
        controller_axis(ax)
        ax.set_ylim(-8, 250)
        titled(ax, title)
    axes[0].set_ylabel("seconds until held for 5 s", color=TEXT_MUTED)
    fig.suptitle("Reaction: from the surge as played (S1) or the submission (S2, S3)", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def plot_cost(rows: list[dict]) -> Figure:
    fig = Figure(figsize=(12, 3.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 3, sharey=True)
    for ax, (scenario, title) in zip(axes, [("s1", "S1"), ("s2", "S2"), ("s3", "S3, first intent's window")]):
        for x, c in enumerate(ORDER):
            for i, s in enumerate(SLICES):
                ys = values(rows, f"{s}_unmet", scenario=scenario, intent=1, controller=c)
                ax.scatter(x + (i - 1) * 0.24 + np.linspace(-0.05, 0.05, len(ys)), ys, s=24, color=SLICE_COLOURS[s], linewidths=0.6,
                           edgecolors=SURFACE, zorder=3, label=SLICE_LABELS[s] if x == 0 else None)
        controller_axis(ax)
        ax.set_ylim(-0.5, 26)
        titled(ax, title)
    axes[0].set_ylabel("offered but not carried, Mb/s", color=TEXT_MUTED)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right", ncols=3)
    fig.suptitle("Who pays: mean unmet demand per slice during the intent", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def on_grid(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A run's clock is wall time, with its own fractions of a second: one common grid to compare runs."""
    keep = np.isfinite(y)
    return np.interp(GRID, t[keep], y[keep])


def plot_timeline(scenario: Scenario) -> Figure:
    fig = Figure(figsize=(13, 7.2), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(3, len(ORDER), sharex=True, sharey="row")
    for col, controller in enumerate(ORDER):
        runs = [load_run(work_of(scenario.name, controller, r))[0] for r in REPEATS]
        for s in SLICES:
            band(axes[0, col], GRID, np.array([on_grid(k[s]["t"], k[s]["share"]) for k in runs]), SLICE_COLOURS[s], SLICE_LABELS[s])
            band(axes[1, col], GRID, np.array([on_grid(k[s]["t"], k[s]["thr"]) for k in runs]), SLICE_COLOURS[s])
        delay = np.array([on_grid(k["urllc"]["t"], np.maximum(k["urllc"]["lat"], 0.05)) for k in runs])
        band(axes[2, col], GRID, delay, SLICE_COLOURS["urllc"])
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
    axes[1, 0].set_ylabel("KPM throughput, Mb/s", color=TEXT_MUTED)
    axes[2, 0].set_ylabel("URLLC RLC delay, ms (log)", color=TEXT_MUTED)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left", ncols=3)
    fig.suptitle(f"{scenario.name.upper()}, {scenario.title}: median over 3 repeats, band = min-max. Dashed: an intent "
                 "is submitted; dotted: its target or bound", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def llm_steps(controller: str, node: str) -> list[float]:
    return [float(e["data"]["latency_s"]) for path in (RESULTS / "testbed").glob(f"*/{controller}/repeat*/events.jsonl") for e in read_events(path)
            if e["event"] == "agent_step" and e["data"].get("node") == node and e["data"].get("latency_s")]


def plot_latency() -> Figure:
    fig = Figure(figsize=(8, 3.4), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    nodes = [("llm_only", "parse"), ("llm_only", "plan"), ("proposed", "parse"), ("proposed", "plan"), ("proposed", "reflect")]
    for x, (controller, node) in enumerate(nodes):
        ys = llm_steps(controller, node)
        ax.scatter(x + np.random.default_rng(x).uniform(-0.13, 0.13, len(ys)), ys, s=24, color=ACCENT if controller == "proposed" else NEUTRAL, alpha=0.75, edgecolors=SURFACE, zorder=3)
        ax.hlines(float(np.median(ys)), x - 0.26, x + 0.26, color=TEXT, linewidth=1.6, zorder=4)
    ax.set_xticks(range(len(nodes)), [f"{LABELS[c]}\n{n}" for c, n in nodes], fontsize=8.5)
    ax.set_ylim(bottom=0)
    titled(ax, "How long the LLM takes per step, on the testbed (each dot is one step)")
    ax.set_ylabel("seconds per step", color=TEXT_MUTED)
    return fig


def sim_values(sim: list[dict], controller: str, scenario: str, intent: int) -> tuple[list[float], list[float]]:
    """(the simulator at the served settings, the same with KPIs 2 s late). M08a reran the served settings
    for RL only and LLM + RL; static and LLM only have no filter to change, so they keep the headline study."""
    served = {"proposed": "fixed", "rl_only": "fixed"}.get(controller, "headline")
    now = values(sim, "intent_met", study=served, controller=controller, scenario=scenario, intent=intent)
    late = values(sim, "intent_met", study="fixed_late", controller=controller, scenario=scenario, intent=intent)
    if controller == "rl_only":
        late = values(sim, "intent_met", study="bridge", controller=controller, scenario=scenario, intent=intent, alpha="0.1", delay_ticks="2")
    return now, late


def plot_sim_vs_testbed(rows: list[dict], sim: list[dict]) -> Figure:
    fig = Figure(figsize=(12, 3.8), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 4, sharey=True)
    sources = ((-0.26, "#b9b7b0", "simulator"), (0.0, "#8d8b85", "simulator, KPIs 2 s late"), (0.26, ACCENT, "testbed"))
    for ax, (scenario, intent, title) in zip(axes, PANELS):
        for x, c in enumerate(ORDER):
            now, late = sim_values(sim, c, scenario, intent)
            real = values(rows, "intent_met", scenario=scenario, intent=intent, controller=c)
            for (shift, colour, label), ys in zip(sources, (now, late, real)):
                if ys:
                    ax.scatter(np.full(len(ys), x + shift), np.array(ys) * 100, s=18, color=colour, edgecolors=SURFACE, linewidths=0.5, zorder=3,
                               label=label if x == 0 and ax is axes[1] else None)
                    ax.hlines(float(np.median(ys)) * 100, x + shift - 0.1, x + shift + 0.1, color=TEXT, linewidth=1.3, zorder=4)
        controller_axis(ax)
        ax.set_ylim(-4, 104)
        titled(ax, title)
    axes[0].set_ylabel("seconds of the window met, %", color=TEXT_MUTED)
    for shift, colour, label in sources:
        axes[1].scatter([], [], s=18, color=colour, label=label)
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[1].legend(unique.values(), unique.keys(), frameon=False, fontsize=8.5, loc="center")
    fig.suptitle("Simulator and testbed, equivalent situations. Simulator: 10 seeds. Testbed: 3 repeats. Bar = median", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def backlogged_ticks() -> list[tuple[str, float, float]]:
    """(slice, share, throughput) at every tick where the slice had a standing queue and so used all it was given."""
    out = []
    for path in (RESULTS / "testbed").glob("*/*/repeat*"):
        kpis, _ = load_run(path)
        for s in SLICES:
            k = kpis[s]
            mask = (k["lat"] > BACKLOGGED_MS) & (k["t"] > 20) & (k["t"] < 355) & (k["share"] > 0.05) & (k["thr"] > 1)
            out += [(s, float(a), float(b)) for a, b in zip(k["share"][mask], k["thr"][mask])]
    return out


def received_against_kpm() -> list[tuple[str, float, float]]:
    """(slice, KPM median, iperf3 median) over t=170-290 of each run's surge segment: the check from outside the RAN."""
    out = []
    for scenario in ("s1", "s3"):
        for controller in ORDER:
            for repeat in REPEATS:
                work = work_of(scenario, controller, repeat)
                kpis, events = load_run(work)
                t0 = next(e for e in events if e["event"] == "scenario_start")["ts_wall"]
                start = [e["ts_wall"] - t0 for e in events if e["event"] == "segment_start"][1]
                for s in SLICES:
                    intervals = json.loads((work / f"seg1_{s}.json").read_text()).get("intervals", [])
                    received = [i["sum"]["bits_per_second"] / 1e6 for n, i in enumerate(intervals) if 170 <= start + n <= 290]
                    k = kpis[s]
                    out.append((s, float(np.median(k["thr"][(k["t"] >= 170) & (k["t"] <= 290)])), float(np.median(received))))
    return out


def plot_cell_rate() -> Figure:
    fig = Figure(figsize=(12, 4.2), facecolor=SURFACE, layout="constrained")
    rate, check = fig.subplots(1, 2)
    ticks = backlogged_ticks()
    for s in SLICES:
        mine = [(a, b) for name, a, b in ticks if name == s]
        rate.scatter(*zip(*mine), s=5, color=SLICE_COLOURS[s], alpha=0.25, linewidths=0, label=SLICE_LABELS[s])
    per_cell = np.median([b / a for _, a, b in ticks])
    xs = np.array([0.0, 0.5])
    rate.plot(xs, xs * per_cell, color=TEXT, linewidth=1.3, label=f"median of these ticks: {per_cell:.0f} Mb/s for the cell")
    rate.plot(xs, xs * SIM_SERVED_CELL, color=TEXT_MUTED, linewidth=1, linestyle="--", label=f"simulator: {SIM_SERVED_CELL:.0f}")
    rate.plot(xs, xs * TESTBED_CAPACITY_MBPS, color=TEXT_MUTED, linewidth=1, linestyle=":", label=f"M07b, used by the verifier: {TESTBED_CAPACITY_MBPS:.0f}")
    rate.set_xlim(0, 0.5)
    rate.set_ylim(bottom=0)
    rate.legend(frameon=False, fontsize=8.5, loc="upper left")
    titled(rate, "Slices with a standing queue: what a share carried (one dot per second)")
    rate.set_xlabel("PRB share", color=TEXT_MUTED)
    rate.set_ylabel("KPM throughput, Mb/s", color=TEXT_MUTED)
    pairs = received_against_kpm()
    for s in SLICES:
        mine = [(a, b) for name, a, b in pairs if name == s]
        check.scatter(*zip(*mine), s=26, color=SLICE_COLOURS[s], edgecolors=SURFACE, linewidths=0.6, zorder=3, label=SLICE_LABELS[s])
    top = max(max(a, b) for _, a, b in pairs) * 1.05
    check.plot([0, top], [0, top], color=TEXT_MUTED, linewidth=1, linestyle=":", label="equal")
    check.set_xlim(0, top)
    check.set_ylim(0, top)
    check.legend(frameon=False, fontsize=8.5, loc="upper left")
    titled(check, "KPM against iperf3 received, t=170-290, S1 and S3 (slice x run)")
    check.set_xlabel("KPM, Mb/s", color=TEXT_MUTED)
    check.set_ylabel("iperf3 received, Mb/s", color=TEXT_MUTED)
    return fig


def segment_starts(scenario: str, controller: str, repeat: int) -> list[float]:
    events = read_events(work_of(scenario, controller, repeat) / "events.jsonl")
    t0 = next(e for e in events if e["event"] == "scenario_start")["ts_wall"]
    return [e["ts_wall"] - t0 for e in events if e["event"] == "segment_start"]


def plot_segments() -> Figure:
    fig = Figure(figsize=(10, 3.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 2)
    for ax, scenario in zip(axes, ("s1", "s3")):
        for x, c in enumerate(ORDER):
            ys = [segment_starts(scenario, c, r)[1] - 120.0 for r in REPEATS]
            ax.scatter(x + np.linspace(-0.14, 0.14, 3), ys, s=34, color=ACCENT if c == "proposed" else NEUTRAL, edgecolors=SURFACE, zorder=3)
            ax.hlines(float(np.median(ys)), x - 0.26, x + 0.26, color=TEXT, linewidth=1.6, zorder=4)
        controller_axis(ax)
        ax.set_ylim(bottom=0)
        titled(ax, f"{scenario.upper()}: the surge starts this late")
    axes[0].set_ylabel("seconds after t=120", color=TEXT_MUTED)
    fig.suptitle("The next iperf3 waits for the previous one to close, so a controller that builds a queue delays the traffic", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def spread(ys: list[float]) -> str:
    return f"{np.median(ys):.2f} [{min(ys):.2f}-{max(ys):.2f}]" if ys else "-"


def print_summary(rows: list[dict], sim: list[dict]) -> None:
    print("\nIntent met, median [min-max] over the repeats; then the same without the flagged runs")
    for scenario, intent, title in PANELS:
        print(title)
        for c in ORDER:
            where = {"scenario": scenario, "intent": intent, "controller": c}
            print(f"  {LABELS[c]:9s} {spread(values(rows, 'intent_met', **where))}   unflagged {spread(values(rows, 'intent_met', valid='True', **where))}")
    print("\nSimulator / simulator 2 s late / testbed, intent met medians")
    for scenario, intent, title in PANELS:
        print(title)
        for c in ORDER:
            now, late = sim_values(sim, c, scenario, intent)
            real = values(rows, "intent_met", scenario=scenario, intent=intent, controller=c)
            print(f"  {LABELS[c]:9s} {np.median(now) if now else float('nan'):.2f}  {np.median(late) if late else float('nan'):.2f}  {np.median(real):.2f}")
    print("\nMedians: unmet Mb/s eMBB/URLLC/BE; URLLC share; p95 delay ms eMBB/URLLC/BE; ping URLLC median, p95 ms; moved %/s; reaction s")
    for scenario, intent, title in PANELS:
        print(title)
        for c in ORDER:
            w = {"scenario": scenario, "intent": intent, "controller": c}
            m = lambda key: float(np.median(values(rows, key, **w) or [float("nan")]))  # noqa: B023
            print(f"  {LABELS[c]:9s} {m('embb_unmet'):5.1f} {m('urllc_unmet'):5.1f} {m('be_unmet'):5.1f}   {m('urllc_share'):.2f}   "
                  f"{m('embb_lat_p95'):7.0f} {m('urllc_lat_p95'):7.0f} {m('be_lat_p95'):7.0f}   {m('ping_urllc_median_ms'):7.1f} {m('ping_urllc_p95_ms'):7.1f}   "
                  f"{m('moved_per_s') * 100:4.1f}   {m('reaction_s'):6.1f}")
    print("\nFlagged runs (grace-period check only):")
    flagged = sorted({(r["scenario"], r["controller"], r["seed"], r["wall_s"]) for r in rows if r["valid"] != "True"})
    for item in flagged:
        print(" ", *item)
    print(f"  {len(flagged)} of {len({(r['scenario'], r['controller'], r['seed']) for r in rows})}")
    print("\nSeconds after t=120 the surge started, per repeat")
    for scenario in ("s1", "s3"):
        for c in ORDER:
            print(f"  {scenario} {LABELS[c]:9s} {[round(segment_starts(scenario, c, r)[1] - 120.0, 1) for r in REPEATS]}")
    ticks = backlogged_ticks()
    ratios = np.array([b / a for _, a, b in ticks])
    print(f"\nCell rate from backlogged ticks: median {np.median(ratios):.1f} Mb/s, 10-90 % {np.percentile(ratios, 10):.1f}-{np.percentile(ratios, 90):.1f}, n={len(ticks)}")
    pairs = received_against_kpm()
    print(f"iperf3 received / KPM, median over slices and runs: {np.median([b / a for _, a, b in pairs]):.3f}")


def main() -> None:
    rows = read_csv(TABLE)
    sim = read_csv(SIM_TABLE)
    IMG.mkdir(parents=True, exist_ok=True)
    figures = {"M08c-intent-met": plot_met(rows), "M08c-reaction": plot_reaction(rows), "M08c-cost": plot_cost(rows), "M08c-llm-latency": plot_latency(),
               "M08c-sim-vs-testbed": plot_sim_vs_testbed(rows, sim), "M08c-cell-rate": plot_cell_rate(), "M08c-segments": plot_segments()}
    figures |= {f"M08c-{name}": plot_timeline(scenario) for name, scenario in SCENARIOS.items()}
    for name, fig in figures.items():
        fig.savefig(IMG / f"{name}.png", dpi=120)
        print(f"wrote {IMG / name}.png")
    print_summary(rows, sim)


if __name__ == "__main__":
    main()
