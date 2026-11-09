"""Figures for docs/learn/M07a-kpm.md, from the files in testbed/results/m07a/.

  uv run python testbed/plot_m07a.py

1. M07a-stock-vs-patched.png: per-slice KPM throughput next to what iperf3 received, on the
   stock OAI gNB (one subscription per slice -> wrapped counters) and on our patched gNB.
2. M07a-prb.png: per-slice RRU.PrbTotDl (stacked) during the overload: who uses the cell. A full
   cell reads ~75 %, not 100 % (see docs/learn/M07a-kpm.md section 5).
3. M07a-delay-vs-rtt.png: KPM's RLC queueing delay next to the ping RTT, per slice, overload.
4. M07a-parity.png: every second of every patched run, KPM throughput against iperf3's.
5. M07a-e2-timeline.png: the E2 messages at the RIC while the xApp was recreated (capture).
6. M07a-report-timing.png: how late KPM reports arrive, and when each subscription fires.

KPM and iperf3 are put on one time axis with their own clocks: the KPM report's collection
start (gNB clock) and iperf3's start timestamp (the receiving UE's container). All containers
share the WSL kernel clock, so the two line up to well under the 1 s report period.
"""

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent))  # testbed/, for plot_spike and plot_m06
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `import sim`
from plot_m06 import ping_series  # noqa: E402
from plot_spike import load_run  # noqa: E402
from sim.plot import LINE_WIDTH, SLICE_COLOURS, SLICE_LABELS, SURFACE, TEXT, TEXT_MUTED, _style  # noqa: E402

RESULTS = Path("testbed/results/m07a")
OUT_STOCK = Path("docs/learn/img/M07a-stock-vs-patched.png")
OUT_PRB = Path("docs/learn/img/M07a-prb.png")
OUT_DELAY = Path("docs/learn/img/M07a-delay-vs-rtt.png")
OUT_PARITY = Path("docs/learn/img/M07a-parity.png")
OUT_E2 = Path("docs/learn/img/M07a-e2-timeline.png")
OUT_TIMING = Path("docs/learn/img/M07a-report-timing.png")
# The patched runs (iperf3 label, marker, legend text, seconds skipped at the start: link adaptation)
PARITY_RUNS = [("patched_together_30M_10M_5M", "o", "30 / 10 / 5 Mb/s together", 6),
               ("patched_together_40M_40M_40M", "s", "3 × 40 Mb/s (overload)", 6),
               ("patched_alone_200M", "^", "one slice alone, 200 Mb/s", 6)]
RIC, GNB, XAPP = "192.168.70.150", "192.168.70.151", "192.168.70.156"
SLICES = ("embb", "urllc", "be")


def load_kpm(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """The xApp's JSON lines -> per slice: t (s, report period start), thp (Mb/s), prb (%), delay (ms).
    One UE per slice on this testbed, so a slice's series is its UE's series."""
    rows: dict[str, list[tuple[float, float, float, float]]] = {s: [] for s in SLICES}
    for line in path.read_text().splitlines():
        if line.startswith('{"slice"'):
            o = json.loads(line)
            rows[o["slice"]].append((o["collect_start_us"] / 1e6, o["DRB.UEThpDl"] / 1000,
                                     o["RRU.PrbTotDl"], o["DRB.RlcSduDelayDl"] / 1000))
    out = {}
    for s, r in rows.items():
        a = np.array(r)
        out[s] = {"t": a[:, 0], "thp": a[:, 1], "prb": a[:, 2], "delay": a[:, 3]}
    return out


def iperf_series(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """(start time, absolute time of each 1 s interval end, received Mb/s)."""
    start = json.loads(path.read_text())["start"]["timestamp"]["timesecs"]
    run = load_run(path)
    return start, start + run["t"], run["mbps"]


def plot_stock_vs_patched() -> Figure:
    fig = Figure(figsize=(13, 4.8), facecolor=SURFACE, layout="constrained")
    ax_stock, ax_fix = fig.subplots(1, 2)
    cases = [(ax_stock, "stock_gnb_3subs.jsonl", "stock_gnb_together_30M_10M_5M",
              "Stock OAI v3.0.0 gNB: deltas against another slice's counter (log scale)"),
             (ax_fix, "patched_monitor_raw.jsonl", "patched_together_30M_10M_5M",
              "Patched gNB (previous values kept per UE)")]
    for ax, kpm_file, run, title in cases:
        kpm = load_kpm(RESULTS / kpm_file)
        t0 = None
        for s in SLICES:
            start, t_iperf, mbps = iperf_series(RESULTS / f"{run}_{s}.json")
            t0 = t0 or start
            k = kpm[s]
            window = (k["t"] >= start - 3) & (k["t"] <= t_iperf[-1] + 3)
            ax.plot(t_iperf - t0, mbps, color=SLICE_COLOURS[s], linewidth=1.2, linestyle="--")
            ax.plot(k["t"][window] - t0, np.maximum(k["thp"][window], 0.01), color=SLICE_COLOURS[s],
                    linewidth=LINE_WIDTH, marker="o", markersize=3, label=SLICE_LABELS[s])
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("time in the run (s)", color=TEXT_MUTED, fontsize=8)
        _style(ax, "throughput (Mb/s)")
        ax.legend(frameon=False, fontsize=8, loc="lower right" if ax is ax_stock else "upper right",
                  title="KPM DRB.UEThpDl (solid)\niperf3 received (dashed)",
                  title_fontsize=8, labelcolor=TEXT)
    ax_stock.set_yscale("log")
    ax_stock.set_ylim(0.01, 1e5)
    ax_fix.set_ylim(0, 65)  # the start-up spike (link adaptation, M06) is real: iperf3 sees it too
    fig.suptitle("Offered 30 / 10 / 5 Mb/s (eMBB / URLLC / BE), one KPM subscription per slice",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    return fig


def plot_prb() -> Figure:
    """Stacked per-slice PRB % during the patched overload run, with the iperf3 run's span."""
    kpm = load_kpm(RESULTS / "patched_monitor_raw.jsonl")
    start, t_iperf, _ = iperf_series(RESULTS / "patched_together_40M_40M_40M_embb.json")
    fig = Figure(figsize=(9, 4), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    t_grid = np.arange(start - 5, t_iperf[-1] + 6, 1.0)
    stack = []
    for s in SLICES:  # each slice's report nearest in time (reports are 1 s apart, ~1 ms between slices)
        k = kpm[s]
        idx = np.clip(np.searchsorted(k["t"], t_grid), 0, len(k["t"]) - 1)
        stack.append(k["prb"][idx])
    ax.stackplot(t_grid - start, *stack, colors=[SLICE_COLOURS[s] for s in SLICES],
                 labels=[SLICE_LABELS[s] for s in SLICES], edgecolor=SURFACE, linewidth=1.0)
    total = np.sum(stack, axis=0)
    ax.plot(t_grid - start, total, color=TEXT, linewidth=1.0, label="sum of the three")
    ax.set_ylim(0, 110)
    ax.set_xlabel("time in the run (s): 3 × 40 Mb/s offered from 0 to 40 s", color=TEXT_MUTED, fontsize=8)
    _style(ax, "RRU.PrbTotDl (% of the cell's PRBs)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncols=4, labelcolor=TEXT)
    ax.set_title("Overload: equal PRB use per slice; the full cell reads ~75 % (73 % for one slice alone)",
                 color=TEXT, fontsize=11, loc="left")
    return fig


def plot_delay_vs_rtt() -> Figure:
    kpm = load_kpm(RESULTS / "patched_monitor_raw.jsonl")
    fig = Figure(figsize=(13, 4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 3, sharey=True)
    for ax, s in zip(axes, SLICES):
        start, t_iperf, _ = iperf_series(RESULTS / f"patched_together_40M_40M_40M_{s}.json")
        t_ping, rtt = ping_series(RESULTS / f"patched_together_40M_40M_40M_ping_{s}.txt")
        k = kpm[s]
        window = (k["t"] >= start - 3) & (k["t"] <= t_iperf[-1] + 3)
        ax.plot(t_ping, rtt, color=TEXT_MUTED, linewidth=1.0, label="ping RTT UE → ext-dn")
        ax.plot(k["t"][window] - start, np.maximum(k["delay"][window], 0.05), color=SLICE_COLOURS[s],
                linewidth=LINE_WIDTH, marker="o", markersize=3, label="KPM DRB.RlcSduDelayDl")
        ax.set_yscale("log")
        ax.set_ylim(0.05, 50_000)
        ax.set_title(SLICE_LABELS[s], color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("time in the run (s)", color=TEXT_MUTED, fontsize=8)
        _style(ax, "ms (log)" if s == "embb" else "")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT)
    fig.suptitle("Overload (3 × 40 Mb/s): the RLC queueing delay reported by KPM follows the ping RTT",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    return fig


def kpm_lag_s(kpm: dict, runs: list) -> float:
    """How much later KPM shows a second of traffic than iperf3 does, in whole report periods.

    A report describes a period that has already ended, and both tools stamp time differently;
    the lag that best lines the two up (tried: 0, 1, 2 s) is used for the parity plot."""
    errors = {}
    for lag in (0.0, 1.0, 2.0):
        diffs = []
        for label, _, _, skip in runs:
            for s in SLICES:
                start, t_iperf, mbps = iperf_series(RESULTS / f"{label}_{s}.json")
                k = kpm[s]
                for t, v in zip(t_iperf[skip:], mbps[skip:]):
                    i = np.argmin(np.abs(k["t"] - lag - t))
                    diffs.append(abs(k["thp"][i] - v))
        errors[lag] = float(np.median(diffs))
    return min(errors, key=errors.get)


def plot_parity() -> Figure:
    kpm = load_kpm(RESULTS / "patched_monitor_raw.jsonl")
    lag = kpm_lag_s(kpm, PARITY_RUNS)
    fig = Figure(figsize=(6.5, 6), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    rel_errors = []
    for label, marker, _text, skip in PARITY_RUNS:
        for s in SLICES:
            start, t_iperf, mbps = iperf_series(RESULTS / f"{label}_{s}.json")
            k = kpm[s]
            idx = [int(np.argmin(np.abs(k["t"] - lag - t))) for t in t_iperf[skip:]]
            ax.scatter(mbps[skip:], k["thp"][idx], s=22, marker=marker, color=SLICE_COLOURS[s],
                       edgecolor=SURFACE, linewidth=0.6, alpha=0.85)
            rel_errors += list(np.abs(k["thp"][idx] - mbps[skip:]) / np.maximum(mbps[skip:], 1))
    ax.plot([3, 100], [3, 100], color=TEXT_MUTED, linewidth=1.0, linestyle=":")
    ax.annotate("KPM = iperf3", (12, 13.5), color=TEXT_MUTED, fontsize=8, rotation=45)
    for s in SLICES:  # legend: colour = slice, shape = run
        ax.scatter([], [], color=SLICE_COLOURS[s], marker="o", label=SLICE_LABELS[s])
    for _, marker, text, _ in PARITY_RUNS:
        ax.scatter([], [], color=TEXT_MUTED, marker=marker, label=text)
    ax.set_xscale("log")  # 5 to 75 Mb/s: on a linear scale the low rates pile up in a corner
    ax.set_yscale("log")
    ax.set_xlim(3, 100)
    ax.set_ylim(3, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("iperf3 received in that second (Mb/s, log)", color=TEXT_MUTED, fontsize=9)
    _style(ax, "KPM DRB.UEThpDl for the same second (Mb/s, log)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TEXT)
    ax.annotate("alone at full load: iperf3's 1 s counts jump\n(UDP arrives in bursts), KPM stays at 71–75",
                (32, 40), color=TEXT_MUTED, fontsize=8, ha="center")
    ax.set_title(f"Patched gNB: KPM vs iperf3, one dot per slice per second\n"
                 f"median error {100 * np.median(rel_errors):.1f} % (KPM shifted by {lag:.0f} s, start-up skipped)",
                 color=TEXT, fontsize=10, loc="left")
    return fig


def load_e2_events(path: Path) -> list[tuple[float, str, str, str]]:
    """(time, source, destination, message) from e2_events.tsv (exported from m07a_e2.pcap)."""
    events = []
    for line in path.read_text().splitlines()[1:]:
        _, t, src, dst, messages = line.split("\t")
        for m in messages.split(","):
            events.append((float(t), src, dst, m))
    return events


def plot_e2_timeline() -> Figure:
    events = load_e2_events(RESULTS / "e2_events.tsv")
    lanes = [  # (label, source, destination, message names)
        ("gNB → RIC: RIC indication (KPM report)", GNB, RIC, {"RICindication"}),
        ("RIC → xApp: RIC indication (forwarded, E42)", RIC, XAPP, {"RICindication"}),
        ("RIC → gNB: subscription delete / request", RIC, GNB, {"RICsubscriptionDeleteRequest", "RICsubscriptionRequest"}),
        ("gNB → RIC: delete / subscription response", GNB, RIC, {"RICsubscriptionDeleteResponse", "RICsubscriptionResponse"}),
    ]
    style = {"RICindication": ("|", TEXT_MUTED), "RICsubscriptionDeleteRequest": ("x", "#c0392b"),
             "RICsubscriptionDeleteResponse": ("x", "#c0392b"), "RICsubscriptionRequest": ("^", "#2a78d6"),
             "RICsubscriptionResponse": ("v", "#2a78d6")}
    fig = Figure(figsize=(12, 3.8), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    for row, (_, src, dst, names) in enumerate(lanes):
        for t, s, d, m in events:
            if s == src and d == dst and m in names:
                marker, colour = style[m]
                ax.scatter(t, row, marker=marker, color=colour, s=60 if marker != "|" else 120, linewidth=1.5)
    for m, text in (("RICsubscriptionDeleteRequest", "subscription delete (old xApp stopped)"),
                    ("RICsubscriptionRequest", "subscription request / response (new xApp)")):
        marker, colour = style[m]
        ax.scatter([], [], marker=marker, color=colour, label=text)
    ax.scatter([], [], marker="|", color=TEXT_MUTED, s=120, label="RIC indication: 3 per second, one per slice")
    deletes = [t for t, _, _, m in events if m == "RICsubscriptionDeleteRequest"]
    requests = [t for t, _, _, m in events if m == "RICsubscriptionRequest"]
    ax.axvspan(max(deletes), min(requests), color=TEXT_MUTED, alpha=0.08)
    ax.annotate(f"no xApp: {min(requests) - max(deletes):.1f} s", ((max(deletes) + min(requests)) / 2, 3.45),
                ha="center", color=TEXT_MUTED, fontsize=8)
    ax.set_yticks(range(len(lanes)), [lane[0] for lane in lanes], fontsize=8)
    ax.set_ylim(-0.6, 3.8)
    ax.set_xlabel("time in the capture (s), at the near-RT RIC", color=TEXT_MUTED, fontsize=8)
    _style(ax, "")
    ax.legend(frameon=False, fontsize=8, loc="center right", labelcolor=TEXT)
    ax.set_title("Recreating the KPM xApp, seen on E2: old subscriptions deleted, three new ones answered in 1 ms",
                 color=TEXT, fontsize=11, loc="left")
    return fig


def plot_report_timing() -> Figure:
    """Left: delivery delay (xApp receive time - collection start). Right: when in the second each
    slice's subscription fires, over the whole run."""
    rows = [json.loads(line) for line in (RESULTS / "patched_monitor_raw.jsonl").read_text().splitlines()
            if line.startswith('{"slice"')]
    fig = Figure(figsize=(12, 3.8), facecolor=SURFACE, layout="constrained")
    ax_lat, ax_phase = fig.subplots(1, 2, width_ratios=[1, 1.6])
    t0 = min(r["collect_start_us"] for r in rows) / 1e6
    bins = np.linspace(0, 2.0, 41)
    for s in SLICES:
        mine = [r for r in rows if r["slice"] == s]
        delay_ms = np.array([(r["recv_us"] - r["collect_start_us"]) / 1000 for r in mine])
        ax_lat.hist(np.clip(delay_ms, 0, 2.0), bins=bins, histtype="step", linewidth=LINE_WIDTH,
                    color=SLICE_COLOURS[s], label=f"{SLICE_LABELS[s]}: median {np.median(delay_ms):.2f} ms")
    ax_lat.set_xlabel("xApp receive time - report collection start (ms)", color=TEXT_MUTED, fontsize=8)
    _style(ax_lat, "reports")
    ax_lat.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    ax_lat.set_title("A report reaches the xApp < 1 ms after the gNB builds it", color=TEXT, fontsize=10, loc="left")
    # Each subscription has its own 1000 ms timer: offset of URLLC's and BE's report from eMBB's.
    starts = {s: np.array(sorted({r["collect_start_us"] for r in rows if r["slice"] == s})) / 1e6 for s in SLICES}
    ref = starts["embb"]
    for s in ("urllc", "be"):
        idx = np.clip(np.searchsorted(starts[s], ref), 1, len(starts[s]) - 1)
        nearest = np.where(np.abs(starts[s][idx] - ref) < np.abs(starts[s][idx - 1] - ref), starts[s][idx], starts[s][idx - 1])
        ok = np.abs(nearest - ref) < 0.5
        ax_phase.plot(ref[ok] - t0, (nearest[ok] - ref[ok]) * 1000, ".", markersize=3, color=SLICE_COLOURS[s],
                      label=f"{SLICE_LABELS[s]} report - eMBB report")
    drift = np.polyfit(ref - t0, (ref % 1.0) * 1000, 1)[0]
    ax_phase.set_xlabel("time since the monitor started (s)", color=TEXT_MUTED, fontsize=8)
    _style(ax_phase, "offset from eMBB's report (ms)")
    ax_phase.set_ylim(-1, 5)
    ax_phase.legend(frameon=False, fontsize=8, labelcolor=TEXT, markerscale=3, loc="upper right")
    ax_phase.set_title(f"The three subscriptions fire 1–2 ms apart, every second; the period is "
                       f"{1000 + drift:.1f} ms, not 1000", color=TEXT, fontsize=10, loc="left")
    return fig


def main() -> None:
    OUT_STOCK.parent.mkdir(parents=True, exist_ok=True)
    plot_stock_vs_patched().savefig(OUT_STOCK, dpi=120)
    plot_prb().savefig(OUT_PRB, dpi=120)
    plot_delay_vs_rtt().savefig(OUT_DELAY, dpi=120)
    plot_parity().savefig(OUT_PARITY, dpi=120)
    plot_e2_timeline().savefig(OUT_E2, dpi=120)
    plot_report_timing().savefig(OUT_TIMING, dpi=120)
    print(f"wrote {OUT_STOCK}, {OUT_PRB}, {OUT_DELAY}, {OUT_PARITY}, {OUT_E2}, {OUT_TIMING}")


if __name__ == "__main__":
    main()
