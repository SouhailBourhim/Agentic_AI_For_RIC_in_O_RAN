"""Figures of M06: the three slices on the testbed, under no load, below capacity and in overload.

Run with:  uv run python testbed/plot_m06.py
Reads testbed/results/m06/ (written by testbed/traffic.py) and writes
- docs/learn/img/M06-slices.png: received throughput (top) and RTT (bottom, log scale) per slice,
  for one slice alone / idle, three slices below capacity, three slices in overload;
- docs/learn/img/M06-resources.png: CPU and memory per container group, idle and in overload.
Every number comes from those files; nothing is typed in by hand.
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `import sim`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # testbed/, for plot_spike and traffic

from plot_spike import load_run  # noqa: E402
from sim.plot import LINE_WIDTH, SLICE_COLOURS, SLICE_LABELS, SURFACE, TEXT, TEXT_MUTED, _style  # noqa: E402
from traffic import PING_INTERVAL_S, SLICES  # noqa: E402

RESULTS = Path("testbed/results/m06")
OUT_SLICES = Path("docs/learn/img/M06-slices.png")
OUT_RESOURCES = Path("docs/learn/img/M06-resources.png")
# (column title, iperf3 run label or None, ping run label)
COLUMNS = [
    ("Each slice alone (200 Mb/s offered) / idle", "alone_200M", "idle"),
    ("Below capacity: 3 × 20 Mb/s", "together_20M_20M_20M", "together_20M_20M_20M"),
    ("Overload: 3 × 40 Mb/s", "together_40M_40M_40M", "together_40M_40M_40M"),
]
# Container groups for the resource figure (the iperf3 clients have random docker names).
GROUPS = {"gNB": ["rfsim5g-oai-gnb"], "3 UEs": [s["container"] for s in SLICES.values()],
          "5G core (7 NFs + mysql)": ["oai-amf", "oai-smf", "oai-upf", "oai-nrf", "oai-udr", "oai-udm",
                                      "oai-ausf", "mysql"],
          "ext-dn": ["oai-ext-dn"], "near-RT RIC": ["nearRT-RIC"]}
IPERF_GROUP = "iperf3 clients"


def ping_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(send time in s, RTT in ms) of every reply; send time = icmp_seq x interval."""
    pairs = re.findall(r"icmp_seq=(\d+) .*time=([\d.]+) ms", path.read_text())
    seq, rtt = (np.array(x, dtype=float) for x in zip(*pairs))
    return seq * PING_INTERVAL_S, rtt


def plot_slices() -> Figure:
    fig = Figure(figsize=(13, 6.5), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(2, 3, sharex=True)
    for col, (title, iperf_label, ping_label) in enumerate(COLUMNS):
        ax_thr, ax_rtt = axes[0, col], axes[1, col]
        for name in SLICES:
            run = load_run(RESULTS / f"{iperf_label}_{name}.json")
            ax_thr.plot(run["t"], run["mbps"], color=SLICE_COLOURS[name], linewidth=LINE_WIDTH,
                        label=f"{SLICE_LABELS[name]} ({run['steady_mbps']:.0f} Mb/s steady)")
            t, rtt = ping_series(RESULTS / f"{ping_label}_ping_{name}.txt")
            ax_rtt.plot(t, rtt, color=SLICE_COLOURS[name], linewidth=LINE_WIDTH, label=SLICE_LABELS[name])
        if iperf_label.startswith("together"):
            offered = load_run(RESULTS / f"{iperf_label}_embb.json")["offered_mbps"]
            ax_thr.axhline(offered, color=TEXT_MUTED, linestyle=":", linewidth=1.2)
            ax_thr.annotate("offered per slice", (20, offered), xytext=(0, -12), textcoords="offset points",
                            color=TEXT_MUTED, fontsize=8)
        ax_thr.set_title(title, color=TEXT, fontsize=10, loc="left")
        ax_thr.set_ylim(0, 85)
        ax_rtt.set_yscale("log")
        ax_rtt.set_ylim(5, 30_000)
        _style(ax_thr, "received (Mb/s)" if col == 0 else "")
        _style(ax_rtt, "RTT UE → ext-dn (ms, log)" if col == 0 else "")
        ax_rtt.set_xlabel("time in the run (s)", color=TEXT_MUTED, fontsize=8)
        ax_thr.legend(frameon=False, fontsize=8, loc="lower right" if col == 0 else "upper right")
    axes[0, 0].annotate("one run per slice,\none after the other", (1, 8), color=TEXT_MUTED, fontsize=8)
    fig.suptitle("Three slices on the rfsim testbed, unmodified OAI scheduler: equal shares whatever the slice",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    return fig


def resource_means(label: str) -> tuple[dict[str, float], dict[str, float]]:
    """Mean CPU (in cores, 100 % = 1 core) and mean memory (GiB) per group over the run's samples."""
    cpu, mem = defaultdict(float), defaultdict(float)
    samples = set()
    known = {c: g for g, members in GROUPS.items() for c in members}
    for row in csv.DictReader((RESULTS / f"{label}_docker_stats.csv").open()):
        samples.add(row["t_wall"])
        group = known.get(row["container"], IPERF_GROUP)
        cpu[group] += float(row["cpu_percent"]) / 100
        mem[group] += float(row["mem_mib"]) / 1024
    n = len(samples)
    return {g: v / n for g, v in cpu.items()}, {g: v / n for g, v in mem.items()}


def plot_resources() -> Figure:
    runs = [("idle (pings only)", "idle"), ("overload (3 × 40 Mb/s)", "together_40M_40M_40M")]
    groups = [*GROUPS, IPERF_GROUP]
    fig = Figure(figsize=(11, 3.6), facecolor=SURFACE, layout="constrained")
    ax_cpu, ax_mem = fig.subplots(1, 2, sharey=True)
    y = np.arange(len(groups))
    height = 0.38
    colours = ["#9b9a94", "#52514e"]  # neutral: the bars are runs, not slices
    for k, (run_label, label) in enumerate(runs):
        cpu, mem = resource_means(label)
        offset = (k - 0.5) * height
        ax_cpu.barh(y + offset, [cpu.get(g, 0.0) for g in groups], height=height - 0.04, color=colours[k],
                    label=f"{run_label}: {sum(cpu.values()):.1f} cores in all")
        ax_mem.barh(y + offset, [mem.get(g, 0.0) for g in groups], height=height - 0.04, color=colours[k],
                    label=f"{run_label}: {sum(mem.values()):.1f} GiB in all")
    ax_cpu.set_yticks(y, groups)
    ax_cpu.invert_yaxis()
    _style(ax_cpu, "")
    _style(ax_mem, "")
    ax_cpu.set_xlabel("CPU (cores; the host has 32 threads)", color=TEXT_MUTED, fontsize=8)
    ax_mem.set_xlabel("memory (GiB; ~23 GiB given to WSL)", color=TEXT_MUTED, fontsize=8)
    ax_cpu.legend(frameon=False, fontsize=8, loc="lower right")
    ax_mem.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Testbed load (docker stats, mean over the run)", color=TEXT, fontsize=12, x=0.01, ha="left")
    return fig


def main() -> None:
    OUT_SLICES.parent.mkdir(parents=True, exist_ok=True)
    plot_slices().savefig(OUT_SLICES, dpi=120)
    plot_resources().savefig(OUT_RESOURCES, dpi=120)
    print(f"wrote {OUT_SLICES} and {OUT_RESOURCES}")


if __name__ == "__main__":
    main()
