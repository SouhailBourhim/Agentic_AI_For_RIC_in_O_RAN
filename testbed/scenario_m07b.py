"""The closed-loop traffic scenario on the testbed (M07b): the simulator's demo surge, in real time.

  uv run python testbed/scenario_m07b.py --run loop1                 rates written to the store
  uv run python testbed/scenario_m07b.py --run loop_blind --no-target

Same shape as sim/scenarios.py demo_traffic with the 75 Mb/s cell: eMBB 37.5 and BE 22.5 Mb/s
throughout; URLLC 7.5 Mb/s, surging to 30 Mb/s (then 90 Mb/s offered in a ~71 Mb/s cell).
Each segment is one iperf3 run per slice (testbed/traffic.py), plus pings on URLLC.

At the start of each segment the rates are written to traffic_target in var/ric-testbed.db, where
FlexRICBackend reads them as offered_load_mbps: the policy was trained with the offered load as an
input, and KPM does not measure it. This is an oracle a real gNB would not have; --no-target
leaves the table empty so the policy sees throughput in its place (rl/runner.py), to measure
what the oracle is worth.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # testbed/, for traffic.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for store
import traffic  # noqa: E402

from store import db, testbed  # noqa: E402
from store.events import log_event  # noqa: E402
from xapps.kpm_monitor import DEFAULT_TESTBED_DB  # noqa: E402

RESULTS = Path("testbed/results/m07b")
# (seconds, Mb/s per slice)
SEGMENTS = [
    (60, {"embb": 37.5, "urllc": 7.5, "be": 22.5}),
    (240, {"embb": 37.5, "urllc": 30.0, "be": 22.5}),
    (60, {"embb": 37.5, "urllc": 7.5, "be": 22.5}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="label of the output files")
    parser.add_argument("--no-target", action="store_true", help="do not tell the backend the offered rates")
    parser.add_argument("--scale", type=float, default=1.0, help="multiply every segment's duration (tests: 0.1)")
    args = parser.parse_args()
    os.environ.setdefault("RIC_DB", str(DEFAULT_TESTBED_DB))
    conn = db.connect()
    conn.execute("DELETE FROM traffic_target")
    traffic.RESULTS = RESULTS
    RESULTS.mkdir(parents=True, exist_ok=True)
    traffic.start_servers()
    log_event("scenario", "scenario_start", data={"run": args.run, "segments": SEGMENTS, "target": not args.no_target})
    for i, (seconds, rates) in enumerate(SEGMENTS):
        seconds = max(5, round(seconds * args.scale))
        if not args.no_target:
            testbed.write_traffic_target(conn, rates)
        log_event("scenario", "segment_start", data={"run": args.run, "segment": i, "rates_mbps": rates,
                                                      "seconds": seconds, "wall": time.time()})
        print(f"segment {i}: {rates} for {seconds} s")
        label = f"{args.run}_seg{i}"
        procs = [traffic.iperf(s, f"{rate}M", seconds, RESULTS / f"{label}_{s}.json") for s, rate in rates.items()]
        procs.append(traffic.ping("urllc", seconds, RESULTS / f"{label}_ping_urllc.txt"))
        for proc in procs:
            proc.wait()
    conn.execute("DELETE FROM traffic_target")
    log_event("scenario", "scenario_stop", data={"run": args.run})


if __name__ == "__main__":
    main()
