"""M10d calibration: does the gNB's DRB.PdcpSduVolumeDL measure each slice's demand?

Run from the repo root with the testbed up on the M10d gNB and KPM xApp images (testbed/up.sh), about 5 minutes:

  uv run python testbed/pdcp_check_m10d.py

Three phases, each with fixed iperf3 UDP rates (testbed/traffic.py) and a fixed split sent straight to the
slice-control xApp (testbed/measure_m07b.SliceXapp, as in M07b's bench test, not through the RL runner):
1. steady: 30 / 10 / 5 Mb/s on eMBB / URLLC / BE together, stock scheduler, 60 s;
2. steps: URLLC alone at 4, 12, 20 and 8.4 Mb/s, 20 s each; every start and stop is a step;
3. overload: URLLC offered 30 Mb/s while held at 5 of the 106 PRBs (about 3 Mb/s of service), 40 s.
Phase 3 is the point: PDCP must read what arrives (30), not what is served.

Writes testbed/results/m10d/: calib_runs.jsonl (each run's label, rates and wall-clock window), calib_kpm.jsonl
(the KPM xApp's lines over the whole check), the iperf3 JSON of each run. eval/m10d.py --calibration scores them
against the rules frozen in docs/DECISIONS.md before this was first run.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # testbed/, for traffic.py and measure_m07b.py
import traffic  # noqa: E402
from measure_m07b import SliceXapp  # noqa: E402

RESULTS = Path("testbed/results/m10d")
SETTLE_S = 2  # after a split command, before traffic starts
STOCK = "del"  # the gNB's stock scheduler: each slice may use the whole cell
URLLC_AT_5_PRB = "51 5 50"  # eMBB / URLLC / BE PRBs, summing to 106; 5 is the gNB's minimum

# (phase, label, split, rates, seconds)
PLAN = [
    ("steady", "steady", STOCK, {"embb": "30M", "urllc": "10M", "be": "5M"}, 60),
    ("steps", "step_04", STOCK, {"urllc": "4M"}, 20),
    ("steps", "step_12", STOCK, {"urllc": "12M"}, 20),
    ("steps", "step_20", STOCK, {"urllc": "20M"}, 20),
    ("steps", "step_8p4", STOCK, {"urllc": "8.4M"}, 20),
    ("overload", "overload", URLLC_AT_5_PRB, {"urllc": "30M"}, 40),
]


def mbps(rate: str) -> float:
    """iperf3's -b ('8.4M') as Mb/s."""
    assert rate.endswith("M"), rate
    return float(rate[:-1])


def main() -> None:
    traffic.RESULTS = RESULTS
    RESULTS.mkdir(parents=True, exist_ok=True)
    traffic.start_servers()
    start = time.time()
    runs = (RESULTS / "calib_runs.jsonl").open("w")
    xapp = SliceXapp("agentic-ric/rl-xapp:latest", RESULTS / "calib_controls.jsonl")
    try:
        split = None
        for phase, label, line, rates, seconds in PLAN:
            if line != split:
                xapp.send(label, line)
                split = line
                time.sleep(SETTLE_S)
            began = time.time()
            traffic.run(label, rates, seconds, with_ping=False, stats_label=f"calib_{label}")
            runs.write(json.dumps({"phase": phase, "label": label, "split": line, "seconds": seconds,
                                   "rates_mbps": {s: mbps(r) for s, r in rates.items()},
                                   "start_wall": began, "end_wall": time.time()}) + "\n")
            runs.flush()
        xapp.send("end", STOCK)  # leave the gNB on its stock scheduler
    finally:
        xapp.close()
        runs.close()
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start - 1))
    kpm = subprocess.run(["docker", "logs", "--since", since, "xapp-kpm-slices"], capture_output=True, text=True)
    lines = [line for line in kpm.stdout.splitlines() if line.startswith("{")]
    (RESULTS / "calib_kpm.jsonl").write_text("".join(line + "\n" for line in lines))
    with_pdcp = sum('"DRB.PdcpSduVolumeDL"' in line for line in lines)
    print(f"{len(lines)} KPM lines, {with_pdcp} with DRB.PdcpSduVolumeDL; score: uv run python -m eval.m10d --calibration")


if __name__ == "__main__":
    main()
