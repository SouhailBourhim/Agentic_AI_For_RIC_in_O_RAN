"""Figure and table for the per-slice floor (M04a): best effort during the URLLC surge, before vs after.

Run with:
  uv run python -m rl.plot_floor --policy "v3 + floor=var/policy_cache/td3_noreward_seed11_best.zip" \
      --policy "new champion + floor=models/m04a_floor/confirm/<name>_best.zip"

Each policy runs the scenario of rl/plot_runner.py (demo surge from t=120 s, intent "20 Mb/s URLLC
within 10 ms" in force from t=150 to 450 s) through the real runner, so always with the runner's
5 % floor. The pre-registered numbers are taken over t = 200-440 s. The "before" row (v3 without
any floor) is the M04a record (commit 5d1b76a): the current runner cannot run without the floor,
on purpose. Writes docs/learn/img/M04a-floor.png.
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from rl.plot_runner import simulate
from sim.plot import SLICE_COLOURS, SURFACE, TEXT, TEXT_MUTED, _style
from sim.scenarios import DEMO_SURGE_END_S, DEMO_SURGE_START_S
from verifier.objective import MIN_SLICE_SHARE

OUT = Path("docs/learn/img/M04a-floor.png")
WINDOW = (200.0, 440.0)  # pre-registered: intent in force, surge on
BEFORE = {"BE share": 0.02, "BE median latency (s)": 41.3, "URLLC median latency (ms)": 4.9}  # M04a record
LINE_STYLES = ["-", (0, (5, 2))]


def window_stats(data: dict[str, np.ndarray]) -> dict[str, float]:
    def inside(s: str) -> np.ndarray:
        rows = data[s]
        return rows[(rows[:, 0] > WINDOW[0]) & (rows[:, 0] <= WINDOW[1])]

    be, urllc = inside("be"), inside("urllc")
    return {"BE share": be[:, 2].mean(), "BE median latency (s)": np.median(be[:, 3]) / 1000.0,
            "URLLC median latency (ms)": np.median(urllc[:, 3])}


def plot(runs: dict[str, dict[str, np.ndarray]]) -> Figure:
    fig = Figure(figsize=(12, 7.5), facecolor=SURFACE, layout="constrained")
    ax_share, ax_be, ax_urllc = fig.subplots(3, 1, sharex=True)
    for (label, data), style in zip(runs.items(), LINE_STYLES):
        t = data["be"][:, 0]
        ax_share.plot(t, data["be"][:, 2], color=SLICE_COLOURS["be"], linestyle=style, linewidth=1.6, label=label)
        ax_be.plot(t, data["be"][:, 3] / 1000.0, color=SLICE_COLOURS["be"], linestyle=style, linewidth=1.6, label=label)
        ax_urllc.plot(t, data["urllc"][:, 3], color=SLICE_COLOURS["urllc"], linestyle=style, linewidth=1.6,
                      label=label)
    ax_share.axhline(MIN_SLICE_SHARE, color=TEXT_MUTED, linewidth=1.0, linestyle=(0, (2, 3)))
    ax_share.annotate("floor 5 %", (DEMO_SURGE_END_S, MIN_SLICE_SHARE), xytext=(4, 3), textcoords="offset points",
                      fontsize=8, color=TEXT_MUTED)
    ax_urllc.axhline(10.0, color=TEXT_MUTED, linewidth=1.0, linestyle=(0, (2, 3)))
    ax_urllc.annotate("URLLC bound 10 ms", (DEMO_SURGE_END_S, 10.0), xytext=(4, 3), textcoords="offset points",
                      fontsize=8, color=TEXT_MUTED)
    ax_be.set_yscale("log")
    ax_urllc.set_yscale("log")
    for ax in (ax_share, ax_be, ax_urllc):
        ax.axvspan(DEMO_SURGE_START_S, DEMO_SURGE_END_S, color="#efeee9", zorder=0)
        ax.axvspan(150.0, 450.0, color="#e8e4fb", zorder=0)
    _style(ax_share, "best-effort PRB share")
    _style(ax_be, "best-effort latency (s, log)")
    _style(ax_urllc, "URLLC latency (ms, log)")
    ax_urllc.set_xlabel("backend clock (s); grey: URLLC surge, lilac: URLLC intent in force", color=TEXT_MUTED,
                        fontsize=9)
    handles, labels = ax_share.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(runs), frameon=False, fontsize=9)
    fig.suptitle("Per-slice floor: best effort during the URLLC surge (both runs through the runner's 5 % floor)",
                 color=TEXT, fontsize=10.5)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", action="append", required=True, help='"label=path/to/model.zip", repeatable')
    args = parser.parse_args()
    runs, rows = {}, {"v3, no floor (M04a record)": BEFORE}
    for item in args.policy:
        label, path = item.split("=", 1)
        data, record, _ = simulate(Path(path))
        assert record["state"] == "applied", record
        runs[label] = data
        rows[label] = window_stats(data)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plot(runs).savefig(OUT, dpi=110)
    print(f"wrote {OUT}\n\nt = {WINDOW[0]:g}-{WINDOW[1]:g} s (surge on, intent in force)")
    print(f"{'':32}{'BE share':>10}{'BE median lat.':>16}{'URLLC median lat.':>19}")
    for label, r in rows.items():
        print(f"{label:32}{r['BE share']:10.3f}{r['BE median latency (s)']:14.1f} s"
              f"{r['URLLC median latency (ms)']:16.1f} ms")


if __name__ == "__main__":
    main()
