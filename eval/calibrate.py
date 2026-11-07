"""M09a: the simulator's testbed profile, computed from the cell's physics by rules written first (docs/DECISIONS.md).

  uv run python -m eval.calibrate

Data: the static runs of M08c and M08d (eval/results/m08c, m08d). Only static runs, because their split never moves
(a third each) and no controller's score enters any parameter. Seconds t <= 20 are left out: the link-adaptation
transient after idle (M06). Writes docs/learn/data/M09a-calibration.csv and prints the values.

The profile itself (eval.sim_profiles) reads that CSV, so no number is copied by hand.
"""

import csv
from pathlib import Path

import numpy as np

from verifier.objective import SLICES

STUDIES = {"m08c": 81.0, "m08d": 79.0}  # true capacity for the "unsaturated" test: M08c's backlogged rate, M08d's measurement
RESULTS = Path("eval/results")
OUT = Path("docs/learn/data/M09a-calibration.csv")
TRANSIENT_S = 20.0
UNSATURATED = 0.8  # offered load at most this fraction of the slice's capacity
SATURATED = 1.05  # offered load above this many times the throughput
SEGMENT_EDGE_S = 10.0  # seconds left out after a change of offered load, for the noise


def static_runs(study: str) -> list[Path]:
    return sorted((RESULTS / study / "testbed").glob("*/static/repeat?"))


def read_kpis(work: Path) -> dict[str, dict[str, np.ndarray]]:
    with (work / "kpi.csv").open() as f:
        rows = list(csv.DictReader(f))
    out = {}
    for s in SLICES:
        mine = [r for r in rows if r["slice"] == s and r["offered_load_mbps"] != ""]
        out[s] = {k: np.array([float(r[col]) for r in mine]) for k, col in
                  (("t", "ts"), ("thr", "dl_thr_mbps"), ("load", "offered_load_mbps"), ("lat", "latency_ms"), ("share", "prb_share_alloc"))}
    return out


def unsaturated(k: dict[str, np.ndarray], capacity: float) -> np.ndarray:
    return (k["t"] > TRANSIENT_S) & (k["load"] <= UNSATURATED * k["share"] * capacity) & (k["thr"] >= 0.5 * k["load"])


def saturated(k: dict[str, np.ndarray]) -> np.ndarray:
    return (k["t"] > TRANSIENT_S) & (k["load"] > SATURATED * k["thr"])


def settled(k: dict[str, np.ndarray]) -> np.ndarray:
    """Seconds at least SEGMENT_EDGE_S after the offered load last changed: inside one constant-rate segment."""
    changed_at = -np.inf
    keep = np.zeros(len(k["t"]), bool)
    for i in range(len(k["t"])):
        if i and k["load"][i] != k["load"][i - 1]:
            changed_at = k["t"][i]
        keep[i] = k["t"][i] - changed_at >= SEGMENT_EDGE_S
    return keep


def calibrate() -> list[dict]:
    base: dict[str, list[float]] = {s: [] for s in SLICES}
    backlog: dict[str, list[float]] = {s: [] for s in SLICES}
    noise: list[float] = []
    runs = 0
    for study, capacity in STUDIES.items():
        for work in static_runs(study):
            runs += 1
            kpis = read_kpis(work)
            for s in SLICES:
                k = kpis[s]
                calm = unsaturated(k, capacity)
                base[s] += list(k["lat"][calm])
                full = saturated(k)
                backlog[s] += list(k["lat"][full] / 1000.0 * k["thr"][full])  # Mb queued: delay x drain rate
                steady = calm & settled(k)
                for value in np.unique(k["load"][steady]):  # one constant-rate segment at a time
                    seg = k["thr"][steady & (k["load"] == value)]
                    if len(seg) >= 20:
                        noise.append(float(np.std(seg) / np.mean(seg)))
    per_slice_buffer = {s: float(np.percentile(backlog[s], 95)) for s in SLICES if backlog[s]}
    rows = [{"parameter": f"base_latency_ms_{s}", "value": round(float(np.median(base[s])), 3), "n": len(base[s]),
             "rule": "median KPM RLC delay, offered <= 0.8 x share x capacity"} for s in SLICES if base[s]]
    measured = [r["value"] for r in rows]
    for s in SLICES:  # deviation, recorded in DECISIONS: under static eMBB is never unsaturated (39.5 offered, ~26 carried)
        if not base[s]:
            rows.append({"parameter": f"base_latency_ms_{s}", "value": round(float(np.median(measured)), 3), "n": 0,
                         "rule": "no unsaturated second under static: median of the other slices' values (same RLC scheduler)"})
    rows += [{"parameter": f"backlog_p95_mb_{s}", "value": round(v, 1), "n": len(backlog[s]),
              "rule": "95th percentile of delay x throughput, saturated seconds"} for s, v in per_slice_buffer.items()]
    rows.append({"parameter": "max_backlog_mb", "value": round(float(np.median(list(per_slice_buffer.values()))), 1),
                 "n": len(per_slice_buffer), "rule": "median over slices of backlog_p95_mb"})
    rows.append({"parameter": "noise_std", "value": round(float(np.median(noise)), 4), "n": len(noise),
                 "rule": "median of std / mean of KPM throughput within constant-rate unsaturated segments"})
    rows.append({"parameter": "static_runs", "value": runs, "n": runs, "rule": "M08c + M08d static runs used"})
    return rows


def main() -> None:
    rows = calibrate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["parameter", "value", "n", "rule"])
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['parameter']:22s} {r['value']:>10}  (n={r['n']})  {r['rule']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
