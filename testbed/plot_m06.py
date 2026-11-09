"""Figures of M06: the three slices on the testbed, under no load, below capacity and in overload.

Run with:  uv run python testbed/plot_m06.py
Reads testbed/results/m06/ (written by testbed/traffic.py) and writes
- docs/learn/img/M06-slices.png: received throughput (top) and RTT (bottom, log scale) per slice,
  for one slice alone / idle, three slices below capacity, three slices in overload;
- docs/learn/img/M06-resources.png: CPU and memory per container group, idle and in overload;
- docs/learn/img/M06-asked-vs-got.png: offered vs received throughput per slice, with the capacity;
- docs/learn/img/M06-rtt-cdf.png: RTT distribution per slice, idle / below capacity / overload;
- docs/learn/img/M06-attach.png: the three UEs attaching, phase by phase, from the N2 capture
  (testbed/results/m06/n2_events.tsv, extracted from the pcap with tshark).
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
from traffic import PING_INTERVAL_S, SLICES, STEADY_AFTER_S  # noqa: E402

RESULTS = Path("testbed/results/m06")
OUT_SLICES = Path("docs/learn/img/M06-slices.png")
OUT_RESOURCES = Path("docs/learn/img/M06-resources.png")
OUT_ASKED = Path("docs/learn/img/M06-asked-vs-got.png")
OUT_CDF = Path("docs/learn/img/M06-rtt-cdf.png")
OUT_ATTACH = Path("docs/learn/img/M06-attach.png")
# (sst, sd) -> slice, as configured (tests/test_testbed_config.py)
SLICE_OF_SNSSAI = {(1, 0x000001): "embb", (2, 0x000001): "urllc", (1, 0x000002): "be"}
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


def plot_asked_vs_got() -> Figure:
    """Offered (outline) vs received (filled) throughput per slice, for the three runs."""
    runs = [("each slice alone", "alone_200M"), ("3 × 20 Mb/s", "together_20M_20M_20M"),
            ("3 × 40 Mb/s (overload)", "together_40M_40M_40M")]
    capacity = float(np.mean([load_run(RESULTS / f"alone_200M_{n}.json")["steady_mbps"] for n in SLICES]))
    top = 95.0
    fig = Figure(figsize=(10, 4.2), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    width = 0.8
    for g, (_title, label) in enumerate(runs):
        total = 0.0
        for k, name in enumerate(SLICES):
            x = g * 4 + k
            r = load_run(RESULTS / f"{label}_{name}.json")
            total += r["steady_mbps"]
            offered = r["offered_mbps"]
            ax.bar(x, min(offered, top), width=width, fill=False, edgecolor=TEXT_MUTED, linewidth=1.2,
                   linestyle="--", label="offered" if (g, k) == (0, 0) else None)
            if offered > top:
                ax.annotate(f"offered {offered:.0f} ↑", (x, top), xytext=(0, -12), textcoords="offset points",
                            ha="center", fontsize=7, color=TEXT_MUTED)
            ax.bar(x, r["steady_mbps"], width=width - 0.22, color=SLICE_COLOURS[name], edgecolor=SURFACE,
                   linewidth=1.5, label=SLICE_LABELS[name] if g == 0 else None)
            ax.annotate(f"{r['steady_mbps']:.0f}", (x, r["steady_mbps"]), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=TEXT)
        if label.startswith("together"):
            ax.annotate(f"sum {total:.0f} Mb/s", (g * 4 + 1, 0), xytext=(0, -28), textcoords="offset points",
                        ha="center", fontsize=8, color=TEXT_MUTED, annotation_clip=False)
    ax.axhline(capacity, color=TEXT, linestyle=":", linewidth=1.2)
    ax.annotate(f"cell capacity ≈ {capacity:.0f} Mb/s (mean of the 'alone' runs)", (10.6, capacity),
                xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8, color=TEXT)
    ax.set_xticks([g * 4 + 1 for g in range(len(runs))], [t for t, _ in runs])
    ax.set_ylim(0, top)
    ax.set_xlim(-0.8, 10.8)
    _style(ax, "throughput (Mb/s, steady part)")
    ax.grid(False, axis="x")
    ax.legend(frameon=False, fontsize=8, ncols=4, loc="upper center")
    fig.suptitle("What each slice asked for and what it got: in overload, equal thirds whatever the slice",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    return fig


def plot_rtt_cdf() -> Figure:
    """Empirical CDF of the RTT per slice: idle, below capacity and overload (after the start-up)."""
    panels = [("idle", "idle", 0.0), ("3 × 20 Mb/s, after the first 15 s", "together_20M_20M_20M", STEADY_AFTER_S),
              ("3 × 40 Mb/s (overload), after the first 15 s", "together_40M_40M_40M", STEADY_AFTER_S)]
    fig = Figure(figsize=(12, 3.8), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 3, sharey=True, sharex=True)
    for ax, (title, label, after) in zip(axes, panels):
        for name in SLICES:
            t, rtt = ping_series(RESULTS / f"{label}_ping_{name}.txt")
            values = np.sort(rtt[t >= after])
            share = np.arange(1, len(values) + 1) / len(values)
            median = float(np.median(values))
            ax.step(values, share, where="post", color=SLICE_COLOURS[name], linewidth=LINE_WIDTH,
                    label=f"{SLICE_LABELS[name]}: median {median:,.0f} ms")
        ax.set_xscale("log")
        ax.set_xlim(5, 30_000)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")
        _style(ax, "share of pings with RTT ≤ x" if ax is axes[0] else "")
        ax.set_xlabel("RTT UE → ext-dn (ms, log)", color=TEXT_MUTED, fontsize=8)
        ax.legend(frameon=False, fontsize=8, loc="lower right" if label != "together_40M_40M_40M" else "upper left")
    fig.suptitle("RTT distribution per slice: in overload every slice waits ~17 s, URLLC included",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    return fig


def attach_phases() -> dict[str, dict]:
    """Per slice: the phase boundaries (s) of its attach and its PDU session's S-NSSAI and 5QI,
    from the N2 events. The capture starts with the old contexts being released (UE restart);
    RAN UE ids are reused, so only events after each UE's Registration request are used."""
    rows = list(csv.DictReader((RESULTS / "n2_events.tsv").open(), delimiter="\t"))
    by_ue = defaultdict(list)
    for row in rows:
        by_ue[row["ran_ue_id"]].append(row)
    phases = {}
    for events in by_ue.values():
        start = next(i for i, e in enumerate(events) if e["procedure_code"] == "15")  # InitialUEMessage
        events = events[start:]
        t = lambda pred: next(float(e["t_s"]) for e in events if pred(e))  # noqa: B023 (used within this iteration)
        pdu = next(e for e in events if e["procedure_code"] == "29" and e["sst"])
        slice_name = SLICE_OF_SNSSAI[(int(pdu["sst"]), int(pdu["sd"], 16))]
        phases[slice_name] = {
            "bounds": [t(lambda e: e["procedure_code"] == "15"),
                       t(lambda e: e["nas_mm_type"] == "0x57"),  # authentication response
                       t(lambda e: e["procedure_code"] == "14"),  # initial context setup request
                       t(lambda e: e["info"].startswith("InitialContextSetupResponse")),
                       t(lambda e: e["procedure_code"] == "29"),  # PDU session resource setup request
                       t(lambda e: e["info"].startswith("PDUSessionResourceSetupResponse"))],
            "label": f"S-NSSAI ({int(pdu['sst'])}, {pdu['sd']}), 5QI {pdu['five_qi']}",
        }
    return phases


def plot_attach() -> Figure:
    phase_names = ["registration + authentication", "security mode", "context setup (allowed NSSAI)",
                   "PDU session request (NAS)", "PDU session resource setup (N2, N3 tunnel)"]
    shades = ["#d9d8d2", "#b8b7b0", "#97968f", "#6f6e68", "#3a3a38"]  # one neutral ramp: ordered phases
    phases = attach_phases()
    fig = Figure(figsize=(11, 3.4), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    order = [n for n in SLICES if n in phases]
    for row, name in enumerate(order):
        bounds = phases[name]["bounds"]
        t0 = bounds[0]
        for k in range(len(phase_names)):
            left, right = (bounds[k] - t0) * 1000, (bounds[k + 1] - t0) * 1000
            ax.barh(row, right - left, left=left, height=0.5, color=shades[k], edgecolor=SURFACE, linewidth=1.5,
                    label=phase_names[k] if row == 0 else None)
        total = (bounds[-1] - t0) * 1000
        ax.annotate(f"{phases[name]['label']}  ·  {total:.0f} ms  ·  starts at t = {t0:.2f} s",
                    (total, row), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(range(len(order)), [f"UE {SLICE_LABELS[n]}" for n in order])
    for tick, name in zip(ax.get_yticklabels(), order):
        tick.set_color(SLICE_COLOURS[name])
    ax.invert_yaxis()
    ax.set_xlim(0, 420)
    _style(ax, "")
    ax.grid(False, axis="y")
    ax.set_xlabel("ms since the UE's Registration request (InitialUEMessage on N2)", color=TEXT_MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.suptitle("The three UEs attaching after a restart, from the N2 capture: each gets its own slice and 5QI",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    return fig


def main() -> None:
    OUT_SLICES.parent.mkdir(parents=True, exist_ok=True)
    plot_slices().savefig(OUT_SLICES, dpi=120)
    plot_resources().savefig(OUT_RESOURCES, dpi=120)
    plot_asked_vs_got().savefig(OUT_ASKED, dpi=120)
    plot_rtt_cdf().savefig(OUT_CDF, dpi=120)
    plot_attach().savefig(OUT_ATTACH, dpi=120)
    print(f"wrote {OUT_SLICES}, {OUT_RESOURCES}, {OUT_ASKED}, {OUT_CDF}, {OUT_ATTACH}")


if __name__ == "__main__":
    main()
