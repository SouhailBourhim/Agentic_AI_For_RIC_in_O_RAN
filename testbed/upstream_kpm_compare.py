"""KPM per slice against iperf3 for one run of testbed/upstream_kpm_run.sh, over seconds 10..30 of
the iperf3 run (the first 10 s are the link-adaptation transient).

Usage: uv run python testbed/upstream_kpm_compare.py testbed/results/upstream_kpm/<label>
"""
import json
import statistics
import sys
from pathlib import Path

run = Path(sys.argv[1])
kpm = [json.loads(line) for line in (run / "kpm_raw.jsonl").read_text().splitlines()]
print(run.name, "|", (run / "gnb_image.txt").read_text().strip()[:90])
for s in ("embb", "urllc", "be"):
    ip = json.loads((run / f"together_30M_10M_5M_{s}.json").read_text())
    t0 = ip["start"]["timestamp"]["timesecs"]
    iperf = [i["sum"]["bits_per_second"] / 1e6 for i in ip["intervals"] if 10 <= i["sum"]["start"] < 30]
    rows = [r for r in kpm if r["slice"] == s and t0 + 10 <= r["collect_start_us"] / 1e6 < t0 + 30]
    thp = [r["DRB.UEThpDl"] / 1000 for r in rows]
    prb = [r["RRU.PrbTotDl"] for r in rows]
    print(f"  {s:5s} iperf3 {statistics.mean(iperf):6.1f} Mb/s | KPM DRB.UEThpDl mean {statistics.mean(thp):10.1f} "
          f"min {min(thp):10.1f} max {max(thp):10.1f} Mb/s | RRU.PrbTotDl median {statistics.median(prb):>9} "
          f"min {min(prb)} max {max(prb)} % | n={len(rows)}")
