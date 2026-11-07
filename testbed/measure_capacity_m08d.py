"""M08d: what can this cell carry, and why did M07b and M08c disagree (0.654 against about 0.75 Mb/s per PRB)?

Run from the repo root with the testbed up (testbed/up.sh):

  uv run python testbed/measure_capacity_m08d.py          about 13 minutes

A bench test like M07b's `cap`: splits go straight to the slice-control xApp, with no policy and no agent.
Each split runs under three conditions, which differ from M07b in one thing at a time:

  A  80 Mb/s per slice, 25 s            M07b's settings, with its docker stats sampling
  B  80 Mb/s per slice, 90 s            is it the short run?
  C  120 % of each slice's share, 90 s  the range the scenarios work in: is it the extreme overload?

The rule that turns C into a capacity was written in docs/DECISIONS.md before this ran. Writes raw files to
eval/results/m08d/capacity/ and the table to docs/learn/data/M08d-capacity.csv.
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # testbed/, for traffic.py and measure_m07b.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import traffic  # noqa: E402
from measure_m07b import SETTLE_S, SliceXapp  # noqa: E402

from xapps.kpm_monitor import replay  # noqa: E402

RESULTS = Path("eval/results/m08d/capacity")
TABLE = Path("docs/learn/data/M08d-capacity.csv")
IMAGE = "agentic-ric/rl-xapp:latest"
N_PRB = 106
REFERENCE_MBPS = 81.0  # M08c's cell rate: what "120 % of a share" is taken of
SLICE_ORDER = ("embb", "urllc", "be")
SPLITS = {"equal": (36, 35, 35), "s60_30_10": (64, 32, 10), "s20_20_60": (21, 21, 64)}  # as in M07b
SKIP_S = {"A": 15.0, "B": 30.0, "C": 30.0}  # A: M07b's steady part; B and C: the rule's first 30 s
SECONDS = {"A": 25, "B": 90, "C": 90}
KEEP_WITHIN_MBPS = 3.0  # the rule: this close to 69, the capacity stays 69
OLD_CAPACITY_MBPS = 69.0


def offered_rates(condition: str, prbs: tuple[int, int, int]) -> dict[str, str]:
    """iperf3's -b per slice. C overloads every slice by the same 20 %, so all use what they're given."""
    if condition in ("A", "B"):
        return {s: "80M" for s in SLICE_ORDER}
    return {s: f"{round(p / N_PRB * REFERENCE_MBPS * 1.2 * 2) / 2:g}M" for s, p in zip(SLICE_ORDER, prbs)}


def window_median(totals: list[tuple[float, float]], start: float, end: float) -> float:
    """Median per-tick value over a run's steady part."""
    return float(np.median([v for t, v in totals if start <= t <= end]))


def capacity_from(rows: list[dict]) -> float:
    """The frozen rule: median over the splits of condition C's cell total, rounded to 1 Mb/s; within 3 Mb/s of 69 it stays 69."""
    value = float(np.median([float(r["kpm_total_mbps"]) for r in rows if r["condition"] == "C"]))
    return OLD_CAPACITY_MBPS if abs(value - OLD_CAPACITY_MBPS) <= KEEP_WITHIN_MBPS else float(round(value))


def iperf_total(label: str, skip_s: float) -> float:
    """Sum over the slices of iperf3's median received rate per second, after the skipped part."""
    total = 0.0
    for s in SLICE_ORDER:
        intervals = json.loads((RESULTS / f"{label}_{s}.json").read_text()).get("intervals", [])
        total += float(np.median([i["sum"]["bits_per_second"] / 1e6 for n, i in enumerate(intervals) if n >= skip_s]))
    return total


def play(xapp: SliceXapp, split: str, condition: str) -> dict:
    label = f"{condition}_{split}"
    prbs = SPLITS[split]
    rates = offered_rates(condition, prbs)
    xapp.send(label, " ".join(map(str, prbs)))
    time.sleep(SETTLE_S)
    started = time.time()
    if condition == "A":
        traffic.run(label, rates, SECONDS[condition], with_ping=True, stats_label=label)  # exactly M07b's call
    else:
        procs = [traffic.iperf(s, rates[s], SECONDS[condition], RESULTS / f"{label}_{s}.json") for s in SLICE_ORDER]
        procs += [traffic.ping(s, SECONDS[condition], RESULTS / f"{label}_ping_{s}.txt") for s in SLICE_ORDER]
        for proc in procs:
            proc.wait()
    print(f"{label}: done in {time.time() - started:.0f} s", flush=True)
    return {"condition": condition, "split": split, "prbs": " ".join(map(str, prbs)), "rates": " ".join(rates.values()), "started": started}


def kpm_totals() -> list[tuple[float, float]]:
    """(tick time, cell total Mb/s) from what the KPM xApp logged, through the evaluation's own aggregation."""
    lines = subprocess.run(["docker", "logs", "xapp-kpm-slices"], capture_output=True, text=True, check=True).stdout.splitlines()
    return [(t[0].ts, sum(r.dl_thr_mbps for r in t)) for t in replay(lines, 200.0) if len(t) == len(SLICE_ORDER)]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    traffic.RESULTS = RESULTS
    traffic.start_servers()
    xapp = SliceXapp(IMAGE, RESULTS / "controls.jsonl")
    runs = []
    try:
        for split in SPLITS:
            for condition in ("A", "B", "C"):
                runs.append(play(xapp, split, condition))
        xapp.send("end", "del")  # leave the gNB on its stock scheduler
    finally:
        xapp.close()
    totals = kpm_totals()
    rows = []
    for run in runs:
        c = run["condition"]
        start, end = run["started"] + SKIP_S[c], run["started"] + SECONDS[c]
        kpm = window_median(totals, start, end)
        received = iperf_total(f"{c}_{run['split']}", SKIP_S[c])
        rows.append({**{k: run[k] for k in ("condition", "split", "prbs", "rates")}, "seconds": SECONDS[c], "skip_s": SKIP_S[c],
                     "kpm_total_mbps": round(kpm, 2), "iperf_total_mbps": round(received, 2), "iperf_per_prb": round(received / N_PRB, 3)})
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)
    print(f"capacity by the frozen rule: {capacity_from(rows):g} Mb/s")


if __name__ == "__main__":
    main()
