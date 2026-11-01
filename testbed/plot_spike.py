"""Figure and summary of the M06-spike throughput runs.

Run with:  uv run python testbed/plot_spike.py
Reads the raw iperf3 JSON files written by testbed/measure_throughput.sh and the saved
log of FlexRIC's xapp_kpm_moni (overload run), and writes
- docs/learn/img/M06-spike-throughput.png (iperf3, receiver side) plus a summary table;
- docs/learn/img/M06-spike-kpm.png (per-UE KPM reports during an overload).
Every number comes from those files; nothing is typed in by hand.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `import sim`

from sim.plot import GRID, LINE_WIDTH, SURFACE, TEXT, TEXT_MUTED, _style  # noqa: E402

RESULTS_DIR = Path("testbed/results/spike")
OUT_PATH = Path("docs/learn/img/M06-spike-throughput.png")
KPM_LOG = RESULTS_DIR / "xapp_kpm_moni_overload.log"
KPM_OUT_PATH = Path("docs/learn/img/M06-spike-kpm.png")
# Offered downlink load per KPM ran_ue_id in the overload run. The UE <-> ran_ue_id mapping is
# inferred from the throughput each UE received (only one UE could reach 40 Mb/s).
KPM_OFFERED = {1: "80", 2: "20", 3: "20"}
UES = ["ue", "ue2", "ue3"]
UE_LABELS = {"ue": "UE 1", "ue2": "UE 2", "ue3": "UE 3"}
# The UEs are not slices yet, so they get neutral categorical colours (dataviz slots 1, 7, 6),
# not the slice colours.
UE_COLOURS = {"ue": "#2a78d6", "ue2": "#6250d6", "ue3": "#008300"}


WARMUP_S = 3.0  # the first seconds ramp up (31 -> 61 -> 79 Mb/s); excluded from the steady mean


def load_run(path: Path) -> dict[str, np.ndarray | float]:
    """Per-second received throughput (Mb/s) and the summary of one iperf3 run.

    With `-R` the iperf3 client (in the UE) is the receiver. `end.sum` then describes the
    *sender* (ext-dn) and always shows the offered rate, so we read `end.sum_received` and the
    per-second intervals (receiver side). Delivered fraction = received bytes / sent bytes:
    iperf3's own `lost_percent` only counts gaps the receiver noticed, which understates drops.
    """
    data = json.loads(path.read_text())
    if "error" in data:
        raise SystemExit(f"{path}: iperf3 failed: {data['error']}")
    t = np.array([iv["sum"]["end"] for iv in data["intervals"]])
    mbps = np.array([iv["sum"]["bits_per_second"] / 1e6 for iv in data["intervals"]])
    sent, received = data["end"]["sum_sent"], data["end"]["sum_received"]
    return {
        "t": t,
        "mbps": mbps,
        "offered_mbps": sent["bits_per_second"] / 1e6,
        "mean_mbps": received["bits_per_second"] / 1e6,
        "steady_mbps": float(mbps[t > WARMUP_S].mean()),
        "delivered": received["bytes"] / sent["bytes"],
    }


def parse_kpm_log(path: Path) -> dict[int, dict[str, list[tuple[int, float]]]]:
    """xapp_kpm_moni prints one block per report: a counter line, then per UE `ran_ue_id = N`
    followed by `<measurement> = <value> [unit]` lines. Returns {ue: {name: [(report, value)]}}.
    """
    series: dict[int, dict[str, list[tuple[int, float]]]] = {}
    report = ue = None
    for line in path.read_text().splitlines():
        if m := re.match(r"\s*(\d+) KPM ind_msg latency", line):
            report = int(m.group(1))
        elif m := re.match(r"ran_ue_id = (\d+)", line):
            ue = int(m.group(1))
        elif (m := re.match(r"((?:DRB|RRU)\.\w+) = ([\d.]+)", line)) and report and ue:
            series.setdefault(ue, {}).setdefault(m.group(1), []).append((report, float(m.group(2))))
    return series


def plot_kpm(series: dict[int, dict[str, list[tuple[int, float]]]]) -> Figure:
    """Three panels from the KPM reports: throughput, PRB share, RLC SDU delay (log)."""
    fig = Figure(figsize=(12, 4), facecolor=SURFACE, layout="constrained")
    panels = [
        ("DRB.UEThpDl", 1e-3, "throughput (Mb/s)", "DRB.UEThpDl"),
        ("RRU.PrbTotDl", 1.0, "PRBs used (% of cell)", "RRU.PrbTotDl"),
        ("DRB.RlcSduDelayDl", 1e-3, "RLC SDU delay (ms, log)", "DRB.RlcSduDelayDl"),
    ]
    colours = dict(zip(sorted(series), UE_COLOURS.values()))
    for ax, (name, scale, ylabel, title) in zip(fig.subplots(1, 3, sharex=True), panels):
        for ue, measures in sorted(series.items()):
            points = [(r, v * scale) for r, v in measures.get(name, []) if r > 1]  # report 1 is garbage
            if name == "DRB.RlcSduDelayDl":
                points = [(r, v) for r, v in points if v > 0]  # log scale; 0 = no queue
            if points:
                r, v = zip(*points)
                ax.plot(r, v, color=colours[ue], linewidth=LINE_WIDTH,
                        label=f"ran_ue_id {ue} ({KPM_OFFERED.get(ue, '?')} Mb/s offered)")
        if name == "DRB.RlcSduDelayDl":
            ax.set_yscale("log")
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")
        _style(ax, ylabel)
        ax.set_xlabel("KPM report number (about 2 per second, 2 subscriptions)", color=TEXT_MUTED, fontsize=8)
    fig.axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("FlexRIC xapp_kpm_moni during an overload: 80 + 20 + 20 Mb/s offered to a ~75 Mb/s cell",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    return fig


def main() -> None:
    runs = {
        (mode, ue): load_run(RESULTS_DIR / f"{mode}_{ue}.json")
        for mode in ("alone", "together")
        for ue in UES
        if (RESULTS_DIR / f"{mode}_{ue}.json").exists()
    }
    if not runs:
        raise SystemExit(f"no iperf3 results in {RESULTS_DIR}; run testbed/measure_throughput.sh first")

    print(f"{'run':<10} {'UE':<4} {'offered':>8} {'mean':>7} {'steady':>7} {'delivered':>10}   (Mb/s; steady = after {WARMUP_S:.0f} s)")
    for (mode, ue), r in runs.items():
        print(f"{mode:<10} {ue:<4} {r['offered_mbps']:>8.1f} {r['mean_mbps']:>7.1f} "
              f"{r['steady_mbps']:>7.1f} {r['delivered']:>9.0%}")
    alone = [r["steady_mbps"] for (m, _), r in runs.items() if m == "alone"]
    together = [r["steady_mbps"] for (m, _), r in runs.items() if m == "together"]
    if alone:  # separate runs, so only the per-UE mean makes sense
        print(f"alone: steady mean per UE {np.mean(alone):.1f} Mb/s")
    if together:  # simultaneous runs: the sum is what the cell delivered in total
        print(f"together: steady sum over UEs {np.sum(together):.1f} Mb/s")

    fig = Figure(figsize=(11, 4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)
    for ax, mode, title in zip(axes, ("alone", "together"), ("Each UE alone", "All three UEs at once")):
        for ue in UES:
            r = runs.get((mode, ue))
            if r is None:
                continue
            ax.plot(r["t"], r["mbps"], color=UE_COLOURS[ue], linewidth=LINE_WIDTH,
                    label=f"{UE_LABELS[ue]}: steady {r['steady_mbps']:.1f} Mb/s")
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")
        _style(ax, "received Mb/s" if mode == "alone" else "")
        ax.set_xlabel("time (s)", color=TEXT_MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        ax.axvspan(0, WARMUP_S, color="#efeee9", zorder=0)
        ax.axhline(0, color=GRID, linewidth=1.0)
    fig.suptitle("rfsim downlink throughput received by each UE, UDP iperf3 offered at 200 Mb/s (OAI v3.0.0, 106 PRB, 30 kHz; shaded: warm-up)",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"wrote {OUT_PATH}")

    if KPM_LOG.exists():
        plot_kpm(parse_kpm_log(KPM_LOG)).savefig(KPM_OUT_PATH, dpi=120)
        print(f"wrote {KPM_OUT_PATH}")


if __name__ == "__main__":
    main()
