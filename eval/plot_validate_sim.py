"""The figures of M09a: the calibrated simulator against the testbed (run eval.validate_sim --final and --compare first).

  uv run python -m eval.plot_validate_sim

Writes docs/learn/img/M09a-*.png and prints the effect checks the rule asked to report beside the verdict.
Same conventions as M08: controllers by position, LLM + RL highlighted.
"""

import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from eval.calibrate import RESULTS as TESTBED_ROOT
from eval.calibrate import read_kpis, static_runs
from eval.metrics import load_run
from eval.plot_control import ACCENT, IMG, LABELS, NEUTRAL, ORDER, controller_axis, titled
from eval.sim_profiles import calibration
from eval.validate_sim import PANELS, RESULTS, STUDIES, testbed_values
from sim.plot import SLICE_COLOURS, SLICE_LABELS, SURFACE, TEXT, TEXT_MUTED
from verifier.objective import SLICES

ATTEMPTS = {1: Path("docs/learn/data/M09a-validation-attempt1.csv"), 2: Path("docs/learn/data/M09a-validation.csv")}
SIM_RUNS = Path("docs/learn/data/M09a-sim-runs.csv")
TOLERANCE = 0.15
STUDY_CAPACITY = {"m08c": 81.0, "m08d": 79.0}
STUDY_LABEL = {"m08c": "M08c (told 69)", "m08d": "M08d (told 79)"}


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def colour(controller: str) -> str:
    return ACCENT if controller == "proposed" else NEUTRAL


def plot_agreement() -> Figure:
    """Each cell: testbed median against simulator median, attempt 1 and attempt 2."""
    fig = Figure(figsize=(12, 5.2), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)
    for ax, (attempt, path) in zip(axes, ATTEMPTS.items()):
        cells = read_csv(path)
        ax.fill_between([0, 1], [-TOLERANCE, 1 - TOLERANCE], [TOLERANCE, 1 + TOLERANCE], color="#e9e8e4", zorder=0,
                        label=f"within {TOLERANCE:g}")
        ax.plot([0, 1], [0, 1], color=TEXT_MUTED, linewidth=0.9, linestyle=":")
        for c in ORDER:
            mine = [r for r in cells if r["controller"] == c]
            ax.scatter([float(r["testbed_median"]) for r in mine], [float(r["sim_median"]) for r in mine], s=42,
                       color=colour(c), marker={"static": "s", "rl_only": "^", "llm_only": "D", "proposed": "o"}[c],
                       edgecolors=SURFACE, linewidths=0.8, zorder=3, label=LABELS[c])
        within = sum(r["within"] == "True" for r in cells)
        titled(ax, f"Attempt {attempt}: {within} of {len(cells)} cells within {TOLERANCE:g}")
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("testbed median, intent met", color=TEXT_MUTED)
    axes[0].set_ylabel("simulator median (10 seeds)", color=TEXT_MUTED)
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Calibrated simulator against the testbed: one point per controller and goal, M08c and M08d. Rule: 26 of 32 within, order in 7 of 8 panels",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def plot_panels() -> Figure:
    """Attempt 2 per panel: the simulator's 10 seeds (dots) and the testbed's median (diamond)."""
    runs = read_csv(SIM_RUNS)
    fig = Figure(figsize=(13, 6.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(2, 4, sharey=True)
    titles = {("s1", 1): "S1 URLLC", ("s2", 1): "S2 URLLC", ("s3", 1): "S3 first: eMBB + BE", ("s3", 2): "S3 second: URLLC"}
    for row, study in enumerate(STUDIES):
        real = testbed_values(study)
        for col, (scenario, intent) in enumerate(PANELS):
            ax = axes[row, col]
            for x, c in enumerate(ORDER):
                ys = [float(r["intent_met"]) * 100 for r in runs if r["study"] == study and r["scenario"] == scenario
                      and r["intent"] == str(intent) and r["controller"] == c]
                ax.scatter(x - 0.12 + np.linspace(-0.1, 0.1, len(ys)), ys, s=12, color=colour(c), alpha=0.7, linewidths=0, zorder=3)
                ax.scatter(x + 0.22, float(np.median(real[(scenario, intent, c)])) * 100, marker="D", s=34, color=TEXT,
                           edgecolors=SURFACE, zorder=4, label="testbed median" if (row, col, x) == (0, 1, 0) else None)
            controller_axis(ax)
            ax.set_ylim(-4, 104)
            titled(ax, f"{STUDY_LABEL[study]}, {titles[(scenario, intent)]}")
    for row in range(2):
        axes[row, 0].set_ylabel("seconds of the window met, %", color=TEXT_MUTED)
    axes[0, 1].legend(frameon=False, fontsize=8.5, loc="lower center")
    fig.suptitle("Attempt 2: the simulator's seeds (dots) and the testbed's median (diamond) per goal", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def plot_calibration() -> Figure:
    """Why the delay model changed: KPM's RLC delay against how full the slice is, static runs, with both models."""
    fig = Figure(figsize=(9, 4.6), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    values = calibration()
    for study, capacity in STUDY_CAPACITY.items():
        for work in static_runs(study):
            kpis = read_kpis(work)
            for s in SLICES:
                k = kpis[s]
                keep = k["t"] > 20
                rho = k["load"][keep] / (k["share"][keep] * capacity)
                ax.scatter(rho, np.maximum(k["lat"][keep], 0.05), s=5, alpha=0.25, color=SLICE_COLOURS[s], linewidths=0,
                           label=SLICE_LABELS[s] if (study, work) == ("m08c", static_runs("m08c")[0]) else None)
    rho = np.linspace(0.05, 0.95, 100)
    ax.plot(rho, 2.0 / (1 - rho), color=TEXT_MUTED, linewidth=1.2, linestyle="--", label="M01 model (M/M/1, URLLC base 2 ms)")
    ax.plot(rho, np.full_like(rho, values["base_latency_ms_urllc"]), color=TEXT, linewidth=1.4,
            label=f"M09a model below saturation: {values['base_latency_ms_urllc']:.2f} ms")
    ax.axvline(1.0, color=TEXT_MUTED, linewidth=0.8, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("offered load / (share x capacity)", color=TEXT_MUTED)
    ax.set_ylabel("KPM RLC delay, ms (log)", color=TEXT_MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    titled(ax, "Static runs, M08c and M08d, one dot per slice and second: flat until the slice saturates, then the queue")
    return fig


def sim_work(study: str, scenario: str, controller: str) -> list[Path]:
    return sorted((RESULTS / study / scenario / controller).glob("seed*"))


def testbed_work(study: str, scenario: str, controller: str) -> list[Path]:
    return sorted((TESTBED_ROOT / study / "testbed" / scenario / controller).glob("repeat?"))


def embb_swing(paths: list[Path]) -> list[float]:
    out = []
    for path in paths:
        k = load_run(path)[0]["embb"]
        keep = (k["t"] > 120) & (k["t"] <= 300)
        out.append(float(np.nanstd(k["share"][keep])))
    return out


def be_after_surge(paths: list[Path]) -> tuple[float, float]:
    shares, thrs = [], []
    for path in paths:
        k = load_run(path)[0]["be"]
        keep = (k["t"] > 125) & (k["t"] <= 360)
        shares += list(k["share"][keep])
        thrs += list(k["thr"][keep])
    return float(np.median(shares)), float(np.median(thrs))


def print_effects() -> None:
    runs = read_csv(SIM_RUNS)
    print("\nRL only, S2: std of the eMBB share, t=120-300 (median over runs)")
    for study in STUDIES:
        print(f"  {study}: testbed {np.median(embb_swing(testbed_work(study, 's2', 'rl_only'))):.3f}  "
              f"sim {np.median(embb_swing(sim_work(study, 's2', 'rl_only'))):.3f}")
    print("LLM + RL, S3 after the surge: BE median share, median Mb/s")
    for study in STUDIES:
        tb, sm = be_after_surge(testbed_work(study, "s3", "proposed")), be_after_surge(sim_work(study, "s3", "proposed"))
        print(f"  {study}: testbed {tb[0]:.3f} / {tb[1]:.1f}  sim {sm[0]:.3f} / {sm[1]:.1f}")
    print("S2, seconds over the bound in the 60 s after the window (median), M08d")
    m08d = read_csv(Path("docs/learn/data/M08d-runs.csv"))
    for c in ORDER:
        tb = [float(r["over_bound_after_s"]) for r in m08d if r["study"] == "testbed" and r["scenario"] == "s2" and r["controller"] == c
              and r["invalid_copy"] == "False" and r["over_bound_after_s"] != ""]
        sm = [float(r["over_bound_after_s"]) for r in runs if r["study"] == "m08d" and r["scenario"] == "s2" and r["controller"] == c
              and r["over_bound_after_s"] != ""]
        print(f"  {LABELS[c]:9s} testbed {np.median(tb):5.1f}  sim {np.median(sm):5.1f}")
    print("LLM only, S1: first URLLC share it applied (sim seeds / testbed repeats)")
    import json
    for study in STUDIES:
        def first(path: Path) -> float | None:
            for line in (path / "events.jsonl").read_text().splitlines():
                e = json.loads(line)
                if e["event"] == "shares_applied":
                    return e["data"]["shares"][1]
            return None
        print(f"  {study}: sim {[first(p) for p in sim_work(study, 's1', 'llm_only')]}  testbed {[first(p) for p in testbed_work(study, 's1', 'llm_only')]}")


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    figures = {"M09a-agreement": plot_agreement(), "M09a-panels": plot_panels(), "M09a-calibration": plot_calibration()}
    for name, fig in figures.items():
        fig.savefig(IMG / f"{name}.png", dpi=120)
        print(f"wrote {IMG / name}.png")
    print_effects()


if __name__ == "__main__":
    main()
