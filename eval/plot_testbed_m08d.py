"""The figures of the testbed rerun (M08d; run eval.testbed_control --final first).

  uv run python -m eval.plot_testbed_m08d

Same conventions as eval.plot_testbed (M08c): controllers by position, LLM + RL highlighted, slices keep their
colours, one dot per run, a hollow dot is a run the close check flagged. All runs count, flagged or not.
Writes docs/learn/img/M08d-*.png and prints the tables of the note.
"""

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from eval.metrics import load_run, measure_intent
from eval.plot_control import ACCENT, IMG, LABELS, NEUTRAL, ORDER, controller_axis, titled, values
from eval.plot_testbed import REPEATS, on_grid, plot_latency, plot_timeline, read_csv, spread, work_of
from eval.testbed_control import CLOSE_LIMIT_S
from eval.testbed_scenarios import SCENARIOS, SEGMENTS, TESTBED_CAPACITY_MBPS
from eval.testbed_scenarios_m08c import TESTBED_CAPACITY_MBPS as M08C_CAPACITY_MBPS
from sim.plot import SLICE_COLOURS, SURFACE, TEXT, TEXT_MUTED
from store.events import read_events

RESULTS = Path("eval/results/m08d")
M08C_RESULTS = Path("eval/results/m08c")
TABLE = Path("docs/learn/data/M08d-runs.csv")
M08C_TABLE = Path("docs/learn/data/M08c-runs.csv")
PANELS = [("s1", 1, "S1: URLLC 21 Mb/s, < 10 ms"), ("s2", 1, "S2: URLLC 19 Mb/s, < 5 ms"),
          ("s3", 1, "S3 first: eMBB and BE 23 Mb/s"), ("s3", 2, "S3 second: URLLC 23 Mb/s, < 5 ms")]
M08C_GREY = "#8d8b85"
SURGE_FROM_S = 125.0  # S3's surge starts at t=120; a few seconds for the policy to settle on the new load


def colour(controller: str) -> str:
    return ACCENT if controller == "proposed" else NEUTRAL


def dots(ax: Axes, x: float, rows: list[dict], key: str, col: str, scale: float = 1.0, **where) -> list[float]:
    """One dot per run (replays and their originals both), hollow if flagged, and the median as a bar."""
    mine = [r for r in rows if all(r[k] == str(v) for k, v in where.items()) and r[key] != ""]
    ys = [float(r[key]) * scale for r in mine]
    for offset, y, r in zip(np.linspace(-0.16, 0.16, len(ys)), ys, mine):
        ax.scatter(x + offset, y, s=30, color=col if r["valid"] == "True" else SURFACE, edgecolors=col, linewidths=1.2, zorder=3)
    if ys:
        ax.hlines(float(np.median(ys)), x - 0.26, x + 0.26, color=TEXT, linewidth=1.6, zorder=4)
    return ys


def plot_met(rows: list[dict], m08c: list[dict]) -> Figure:
    fig = Figure(figsize=(12, 3.8), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 4, sharey=True)
    for ax, (scenario, intent, title) in zip(axes, PANELS):
        for x, c in enumerate(ORDER):
            dots(ax, x, rows, "intent_met", colour(c), 100.0, study="testbed", scenario=scenario, intent=intent, controller=c)
            before = values(m08c, "intent_met", scenario=scenario, intent=intent, controller=c)
            ax.scatter(x + 0.36, float(np.median(before)) * 100, marker="D", s=26, color=M08C_GREY, edgecolors=SURFACE, zorder=3,
                       label="M08c median (69 Mb/s)" if x == 0 and scenario == "s2" else None)
        controller_axis(ax)
        ax.set_ylim(-4, 104)
        titled(ax, title)
    axes[0].set_ylabel("seconds of the window met, %", color=TEXT_MUTED)
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower center")
    axes[2].annotate("2 of 3: 0.99 with the\nplayed load (gap, note §5)", (2, 79.5), (1.2, 55), fontsize=8, color=TEXT_MUTED,
                     arrowprops={"arrowstyle": "-", "color": TEXT_MUTED, "linewidth": 0.8})
    fig.suptitle("Is the operator's goal met? M08d at 79 Mb/s: one dot per run, hollow = flagged by the close check, bar = median. "
                 "Grey diamond: M08c", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def surge_points(results: Path, scenario: str, controller: str) -> tuple[np.ndarray, np.ndarray]:
    """(BE share, BE throughput) at every second of S3's surge, all repeats of one controller."""
    shares, thrs = [], []
    for r in REPEATS:
        be = load_run(work_of(scenario, controller, r, results))[0]["be"]
        keep = (be["t"] > SURGE_FROM_S) & (be["t"] <= 360.0)
        shares += list(be["share"][keep])
        thrs += list(be["thr"][keep])
    return np.array(shares), np.array(thrs)


def plot_margin() -> Figure:
    """Why LLM + RL misses S3's first goal more often at 79 Mb/s than at 69: the share it picks for BE."""
    fig = Figure(figsize=(12, 4.2), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)
    for ax, results, capacity, target, name in ((axes[0], M08C_RESULTS, M08C_CAPACITY_MBPS, 20.0, "M08c"),
                                                (axes[1], RESULTS, TESTBED_CAPACITY_MBPS, 23.0, "M08d")):
        share, thr = surge_points(results, "s3", "proposed")
        ax.scatter(share, thr, s=7, color=SLICE_COLOURS["be"], alpha=0.35, linewidths=0, label="LLM + RL, one dot per second")
        for c, marker in (("static", "s"), ("llm_only", "^")):
            s, t = surge_points(results, "s3", c)
            ax.scatter(np.median(s), np.median(t), marker=marker, s=46, color=NEUTRAL, edgecolors=SURFACE, zorder=4, label=f"{LABELS[c]}, median")
        xs = np.array([0.15, 0.45])
        ax.plot(xs, xs * capacity, color=TEXT, linewidth=1.2, label=f"share x {capacity:.0f} Mb/s: what the verifier and policy assume")
        ax.axhline(target, color=SLICE_COLOURS["be"], linewidth=1.2, linestyle=":", label=f"BE target {target:.0f} Mb/s")
        met = float(np.mean(thr >= target))
        titled(ax, f"{name}: BE after the surge, S3. LLM + RL at or above target {met:.0%} of seconds")
        ax.set_xlim(0.15, 0.45)
        ax.set_ylim(10, 30)
        ax.set_xlabel("BE PRB share", color=TEXT_MUTED)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    axes[0].set_ylabel("BE KPM throughput, Mb/s", color=TEXT_MUTED)
    return fig


def share_swing(results: Path, scenario: str, controller: str) -> list[float]:
    """Standard deviation of the eMBB share during S2's intent window (t=120-300), per repeat: how much the policy swings."""
    out = []
    for r in REPEATS:
        embb = load_run(work_of(scenario, controller, r, results))[0]["embb"]
        keep = (embb["t"] > 120.0) & (embb["t"] <= 300.0)
        out.append(float(np.nanstd(embb["share"][keep])))
    return out


def plot_swing(rows: list[dict]) -> Figure:
    fig = Figure(figsize=(12, 3.8), facecolor=SURFACE, layout="constrained")
    swing, after = fig.subplots(1, 2)
    labels = []
    for x, (c, results, name) in enumerate([("rl_only", M08C_RESULTS, "M08c"), ("rl_only", RESULTS, "M08d"),
                                            ("proposed", M08C_RESULTS, "M08c"), ("proposed", RESULTS, "M08d")]):
        ys = share_swing(results, "s2", c)
        swing.scatter(x + np.linspace(-0.12, 0.12, len(ys)), ys, s=30, color=colour(c) if name == "M08d" else M08C_GREY, edgecolors=SURFACE, zorder=3)
        swing.hlines(float(np.median(ys)), x - 0.24, x + 0.24, color=TEXT, linewidth=1.6, zorder=4)
        labels.append(f"{LABELS[c]}\n{name}")
    swing.set_xticks(range(len(labels)), labels, fontsize=8.5)
    swing.set_ylim(bottom=0)
    swing.set_ylabel("std of the eMBB share, t=120-300", color=TEXT_MUTED)
    titled(swing, "S2: how much the policy's split swings")
    for x, c in enumerate(ORDER):
        dots(after, x, rows, "over_bound_after_s", colour(c), study="testbed", scenario="s2", intent=1, controller=c)
    controller_axis(after)
    after.set_ylim(-2, 62)
    after.set_ylabel("seconds over 5 ms", color=TEXT_MUTED)
    titled(after, "S2: URLLC over its bound in the 60 s after the window (t=300-360)")
    return fig


def closed_after(path: Path) -> float | None:
    stop = [e for e in read_events(path / "events.jsonl") if e["event"] == "scenario_stop"]
    return stop[-1]["data"].get("closed_after_s") if stop else None


def plot_close() -> Figure:
    """How long after the scenario's end the last iperf3 and ping closed; the run waits at most CLOSE_LIMIT_S."""
    fig = Figure(figsize=(12, 3.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 3, sharey=True)
    for ax, scenario in zip(axes, ("s1", "s2", "s3")):
        for x, c in enumerate(ORDER):
            paths = sorted((RESULTS / "testbed" / scenario / c).glob("repeat*"))
            ys = [y for p in paths if (y := closed_after(p)) is not None]
            ax.scatter(x + np.linspace(-0.16, 0.16, len(ys)), ys, s=30, color=colour(c), edgecolors=SURFACE, zorder=3)
        ax.axhline(CLOSE_LIMIT_S, color=TEXT, linewidth=1, linestyle=":")
        controller_axis(ax)
        ax.set_ylim(0, CLOSE_LIMIT_S + 15)
        titled(ax, scenario.upper())
    axes[0].set_ylabel("seconds after t=360", color=TEXT_MUTED)
    fig.suptitle(f"Time for the traffic to close after the end (dotted: the {CLOSE_LIMIT_S:.0f} s limit). S3 RL only: originals and replays",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def plot_what_if(rows: list[dict]) -> Figure:
    """S3 for LLM + RL, as run (URLLC base latency 2 ms in the verifier) and with the verifier told 0.6 ms."""
    fig = Figure(figsize=(12, 3.8), facecolor=SURFACE, layout="constrained")
    met, shares = fig.subplots(1, 2, width_ratios=(1, 1.6))
    for x, (study, intent, label) in enumerate([("testbed", 1, "first\nas run"), ("what_if", 1, "first\nwhat-if"),
                                                 ("testbed", 2, "second\nas run"), ("what_if", 2, "second\nwhat-if")]):
        dots(met, x, rows, "intent_met", ACCENT if study == "testbed" else NEUTRAL, 100.0, study=study, scenario="s3", intent=intent, controller="proposed")
    met.set_xticks(range(4), ["first goal\nas run", "first goal\nwhat-if", "second goal\nas run", "second goal\nwhat-if"], fontsize=8.5)
    met.set_ylim(-4, 104)
    met.set_ylabel("seconds of the window met, %", color=TEXT_MUTED)
    titled(met, "S3, LLM + RL: goal met")
    grid = np.arange(0.0, 360.0)
    for study, style in (("testbed", "-"), ("what_if", "--")):
        for s in ("embb", "urllc", "be"):
            runs = [load_run(work_of("s3", "proposed", r, RESULTS, study))[0][s] for r in REPEATS]
            median = np.median([on_grid(k["t"], k["share"]) for k in runs], axis=0)
            shares.plot(grid, median, color=SLICE_COLOURS[s], linestyle=style, linewidth=1.4,
                        label=f"{s.upper() if s != 'be' else 'BE'}, {'as run' if study == 'testbed' else 'what-if'}")
    for at in (60.0, 120.0):
        shares.axvline(at, color=TEXT_MUTED, linewidth=0.9, linestyle="--")
    shares.set_ylim(0, 1)
    shares.set_xlabel("time, s", color=TEXT_MUTED)
    shares.set_ylabel("PRB share, median of 3", color=TEXT_MUTED)
    shares.legend(frameon=False, fontsize=8, ncols=3, loc="upper left")
    titled(shares, "The split: both intents accepted (what-if) moves the cell to URLLC")
    return fig


def with_played_load(kpis: dict, scenario: str) -> dict:
    """Fill missing offered-load ticks with the rate iperf3 was playing at that time (the scenario's own schedule).
    Used only to show what two runs score without the gap a left-over traffic thread caused (note, part 5)."""
    for s, k in kpis.items():
        played, start = np.full(len(k["t"]), np.nan), 0.0
        for seconds, rates in SEGMENTS[scenario]:
            played[(k["t"] > start) & (k["t"] <= start + seconds)] = rates[s]
            start += seconds
        k["load"] = np.where(np.isnan(k["load"]), played, k["load"])
    return kpis


def load_gaps() -> list[tuple[str, int, float, float]]:
    """(run, missing ticks, intent 1 met as logged, intent 1 met with the played load) for every run with a gap."""
    out = []
    for path in sorted((RESULTS / "testbed").glob("*/*/repeat*")):
        kpis, events = load_run(path)
        missing = int(np.isnan(kpis["be"]["load"]).sum())
        if missing:
            scenario = SCENARIOS[path.parts[-3]]
            logged = measure_intent(kpis, events, scenario, 0)["intent_met"]
            filled = measure_intent(with_played_load(kpis, scenario.name), events, scenario, 0)["intent_met"]
            out.append((str(path.relative_to(RESULTS)), missing, logged, filled))
    return out


def print_summary(rows: list[dict], m08c: list[dict]) -> None:
    main_rows = [r for r in rows if r["study"] == "testbed" and r["invalid_copy"] == "False"]
    print("\nIntent met, median [min-max]: M08d (the run each repeat ended with) | M08c")
    for scenario, intent, title in PANELS:
        print(title)
        for c in ORDER:
            w = {"scenario": scenario, "intent": intent, "controller": c}
            print(f"  {LABELS[c]:9s} {spread(values(main_rows, 'intent_met', **w)):22s} {spread(values(m08c, 'intent_met', **w))}")
    print("\nS3 RL only, originals (flagged) against replays (flagged)")
    for intent in (1, 2):
        w = {"study": "testbed", "scenario": "s3", "intent": intent, "controller": "rl_only"}
        print(f"  intent {intent}: originals {spread(values(rows, 'intent_met', invalid_copy='True', **w))}, "
              f"replays {spread(values(rows, 'intent_met', invalid_copy='False', **w))}")
    print("\nWhat-if, S3 LLM + RL:", *(f"intent {i} {spread(values(rows, 'intent_met', study='what_if', intent=i))}" for i in (1, 2)))
    print("\nMedians (intent 1): unmet Mb/s eMBB/URLLC/BE; URLLC share; URLLC delay median, p95 ms; ping URLLC median ms; over bound after s; reaction s; first apply s")
    for scenario, intent, title in PANELS[:3]:
        print(title)
        for c in ORDER:
            w = {"scenario": scenario, "intent": intent, "controller": c}
            m = lambda key: float(np.median(values(main_rows, key, **w) or [float("nan")]))  # noqa: B023
            print(f"  {LABELS[c]:9s} {m('embb_unmet'):5.1f} {m('urllc_unmet'):5.1f} {m('be_unmet'):5.1f}   {m('urllc_share'):.2f}   "
                  f"{m('urllc_lat_median'):8.1f} {m('urllc_lat_p95'):8.0f}   {m('ping_urllc_median_ms'):7.1f}   {m('over_bound_after_s'):5.1f}   "
                  f"{m('reaction_s'):6.1f}   {m('first_apply_s'):6.1f}")
    print("\nLLM step latency, median s (parse / plan):")
    for c in ("llm_only", "proposed"):
        w = {"study": "testbed", "intent": 1, "controller": c}
        print(f"  {LABELS[c]:9s} {spread(values(rows, 'llm_parse_median_s', **w))} / {spread(values(rows, 'llm_plan_median_s', **w))}")
    print("\nS2 eMBB share std, t=120-300:")
    for c in ("rl_only", "proposed"):
        print(f"  {LABELS[c]:9s} M08c {[round(v, 3) for v in share_swing(M08C_RESULTS, 's2', c)]}  M08d {[round(v, 3) for v in share_swing(RESULTS, 's2', c)]}")
    print("\nS3 after the surge, BE: median share, median Mb/s, seconds at or above target")
    for name, results, target in (("M08c", M08C_RESULTS, 20.0), ("M08d", RESULTS, 23.0)):
        for c in ("static", "llm_only", "proposed"):
            s, t = surge_points(results, "s3", c)
            print(f"  {name} {LABELS[c]:9s} {np.median(s):.3f}  {np.median(t):5.1f}  {np.mean(t >= target):.2f}  ({np.median(t) / np.median(s):.1f} Mb/s per whole cell)")
    print("\nRuns with missing offered load (ticks), S3 first goal as logged -> with the played load:")
    for run, missing, logged, filled in load_gaps():
        print(f"  {run}: {missing} ticks, {logged:.3f} -> {filled:.3f}")
    print("\nClosed after (s), every run directory:")
    for path in sorted((RESULTS / "testbed").glob("*/*/repeat*")):
        print(f"  {path.relative_to(RESULTS)}: {closed_after(path)}")


def main() -> None:
    rows = read_csv(TABLE)
    m08c = read_csv(M08C_TABLE)
    IMG.mkdir(parents=True, exist_ok=True)
    figures = {"M08d-intent-met": plot_met(rows, m08c), "M08d-margin": plot_margin(), "M08d-swing": plot_swing(rows),
               "M08d-close": plot_close(), "M08d-what-if": plot_what_if(rows), "M08d-llm-latency": plot_latency(RESULTS)}
    figures |= {f"M08d-{name}": plot_timeline(scenario, RESULTS) for name, scenario in SCENARIOS.items()}
    for name, fig in figures.items():
        fig.savefig(IMG / f"{name}.png", dpi=120)
        print(f"wrote {IMG / name}.png")
    print_summary(rows, m08c)


if __name__ == "__main__":
    main()
