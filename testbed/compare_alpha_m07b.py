"""Does alpha 0.10 calm the policy on the real cell? Compares two surge runs already on disk.

  uv run python testbed/compare_alpha_m07b.py

loop_rl (alpha 0.25) and loop_alpha10 (alpha 0.10): same scenario, same policy, one run each, so a
small difference between them may be run-to-run noise. Same measures as
testbed/sim_stability_m07b.py where the real cell allows it. Throughput and delay come from KPM,
the split from the kpi rows the runner wrote. The surge is also cut into four equal parts, to see
whether a difference holds all along. Writes testbed/results/m07b/alpha_compare.json and
docs/learn/img/M07b-alpha.png.
"""

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent))  # testbed/, for plot_m07b
from plot_m07b import load_kpi, ping_p95, segment_times  # noqa: E402

from sim.plot import LINE_WIDTH, SLICE_COLOURS, SLICE_LABELS, SURFACE, TEXT, TEXT_MUTED, _style  # noqa: E402

RESULTS = Path("testbed/results/m07b")
RUNS = [("loop_rl", 0.25), ("loop_alpha10", 0.10)]
SETTLE_S = 30.0  # the surge's first seconds are the reaction, measured apart
TARGET_MBPS = 20.0
RELEASE_SHARE = 0.25


def dominant_period(share: np.ndarray) -> float:
    """Period in s of the strongest swing (one sample per tick, about 1 s)."""
    power = np.abs(np.fft.rfft(share - share.mean()))
    return float(1 / np.fft.rfftfreq(len(share), 1.0)[int(np.argmax(power[1:]) + 1)])


def run_start(run: str) -> float:
    rows = (RESULTS / f"{run}_kpi.csv").read_text().splitlines()[1:]
    return min(float(r.split(",")[0]) for r in rows)


def quarters(embb: dict, urllc: dict, start: float, end: float) -> list[dict]:
    """The same measures on four equal parts of the settled surge."""
    edges = np.linspace(start, end, 5)
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (urllc["t"] >= a) & (urllc["t"] < b)
        parts.append({"embb_moved_per_s": float(np.abs(np.diff(embb["alloc"][m])).mean()),
                      "urllc_met_20": float((urllc["thr"][m] >= TARGET_MBPS).mean()),
                      "urllc_delay_median_ms": float(np.median(urllc["lat"][m]))})
    return parts


def measure(run: str, alpha: float) -> dict:
    kpi = load_kpi(run)
    t0 = run_start(run)
    surge_start, _, surge_end = segment_times(run, t0)[1]
    post_start, _, post_end = segment_times(run, t0)[2]
    embb, urllc, be = kpi["embb"], kpi["urllc"], kpi["be"]
    in_surge = (urllc["t"] >= surge_start + SETTLE_S) & (urllc["t"] < surge_end)
    in_post = (urllc["t"] >= post_start) & (urllc["t"] < post_end)
    after_start = np.where(urllc["t"] >= surge_start)[0]
    after_end = (urllc["t"] >= surge_end) & (urllc["alloc"] < RELEASE_SHARE)
    return {
        "run": run, "alpha": alpha, "surge_seconds": int(in_surge.sum()),
        "embb_moved_per_s": float(np.abs(np.diff(embb["alloc"][in_surge])).mean()),
        "urllc_moved_per_s": float(np.abs(np.diff(urllc["alloc"][in_surge])).mean()),
        "embb_share_min": float(embb["alloc"][in_surge].min()), "embb_share_max": float(embb["alloc"][in_surge].max()),
        "embb_share_p5": float(np.percentile(embb["alloc"][in_surge], 5)),
        "embb_share_p95": float(np.percentile(embb["alloc"][in_surge], 95)),
        "embb_period_s": dominant_period(embb["alloc"][in_surge]),
        "embb_mbps": float(embb["thr"][in_surge].mean()), "embb_mbps_std": float(embb["thr"][in_surge].std()),
        "urllc_mbps": float(urllc["thr"][in_surge].mean()), "be_mbps": float(be["thr"][in_surge].mean()),
        "urllc_met_20": float((urllc["thr"][in_surge] >= TARGET_MBPS).mean()),
        "urllc_delay_median_ms": float(np.median(urllc["lat"][in_surge])),
        "urllc_delay_p95_ms": float(np.percentile(urllc["lat"][in_surge], 95)),
        "embb_delay_median_ms": float(np.median(embb["lat"][in_surge])),
        "urllc_ping_p95_ms": ping_p95(RESULTS / f"{run}_seg1_ping_urllc.txt"),
        "s_to_20_mbps": float(next(urllc["t"][i] - surge_start for i in after_start
                                   if (urllc["thr"][i:i + 5] >= TARGET_MBPS).all())),
        "s_to_release": float(urllc["t"][after_end][0] - surge_end),
        "after_surge_embb_mbps": float(embb["thr"][in_post].mean()),
        "after_surge_urllc_mbps": float(urllc["thr"][in_post].mean()),
        "after_surge_be_mbps": float(be["thr"][in_post].mean()),
        "surge_quarters": quarters(embb, urllc, surge_start + SETTLE_S, surge_end),
    }


def plot() -> Figure:
    """One column per run, same scales: the split on top, URLLC's queueing delay below."""
    fig = Figure(figsize=(11, 5.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(2, len(RUNS), sharex=True, sharey="row")
    for col, (run, alpha) in enumerate(RUNS):
        kpi = load_kpi(run)
        segments = segment_times(run, run_start(run))
        for s in ("embb", "urllc", "be"):
            axes[0, col].plot(kpi[s]["t"], kpi[s]["alloc"], color=SLICE_COLOURS[s], linewidth=LINE_WIDTH, label=SLICE_LABELS[s])
        axes[1, col].plot(kpi["urllc"]["t"], kpi["urllc"]["lat"] / 1000, color=SLICE_COLOURS["urllc"], linewidth=LINE_WIDTH)
        for ax in axes[:, col]:
            ax.axvspan(segments[1][0], segments[1][2], color=TEXT_MUTED, alpha=0.08, linewidth=0)
        _style(axes[0, col], f"alpha {alpha}: PRB share")
        _style(axes[1, col], "URLLC queueing delay, s")
        axes[1, col].set_xlabel("time, s (grey: URLLC surge, sent 30 Mb/s)", color=TEXT)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left", ncols=3)
    fig.suptitle("The same policy on the real cell, one run per filter setting", color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def main() -> None:
    results = [measure(run, alpha) for run, alpha in RUNS]
    (RESULTS / "alpha_compare.json").write_text(json.dumps(results, indent=1))
    for key in results[0]:
        if key != "surge_quarters":
            print(f"{key:24s}" + "".join(f"{r[key]:>14.3f}" if isinstance(r[key], float) else f"{r[key]!s:>14}" for r in results))
    for r in results:
        for q in r["surge_quarters"]:
            print(f"{r['run']:14s} moved {q['embb_moved_per_s']:.3f}/s, URLLC >= 20 Mb/s {q['urllc_met_20']:.2f}, "
                  f"URLLC delay {q['urllc_delay_median_ms']:.0f} ms")
    plot().savefig("docs/learn/img/M07b-alpha.png", dpi=120)
    print("wrote docs/learn/img/M07b-alpha.png")


if __name__ == "__main__":
    main()
