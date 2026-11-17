"""M10d: the controller on the demand the gNB measures (KPM DRB.PdcpSduVolumeDL) instead of the iperf3 rate.

  uv run python -m eval.m10d --calibration   score testbed/pdcp_check_m10d.py's run against the frozen rules
  uv run python -m eval.m10d --final         the S1 matrix on the testbed: 18 runs, about 2.2 hours, resumes if stopped
  uv run python -m eval.m10d --score         the S1 table and the medians per executor and demand source
  uv run python -m eval.m10d --demand-series RUN_DIR DB_COPY   one S1 run's demand sources over time, for the note

The calibration rules (docs/DECISIONS.md, 2026-11-16, written before the first calibration run):
- steady: per slice, the median |PDCP - target| over the settled reports is within max(1 Mb/s, 5 % of target);
- overload: URLLC held at 5 PRBs and offered 30 Mb/s: PDCP's median within the same tolerance of 30, and the
  KPM throughput's median at most half of it (so PDCP reads arrivals, not service);
- steps: the third report after each step's start is within tolerance (a lag of 2 reports at most).
One logged revision is allowed; if it still fails, M10d stops and S1 isn't played (--final refuses).

Revision 1 (2026-11-16, after the first calibration run failed on the overload check only): a run's window is
iperf3's sending time, [start, start + seconds], not the time testbed/traffic.run took to return. A saturated slice
drains its backlog for long after the sender stops (about 100 s in that run), and PDCP rightly reads 0 there: no
arrivals. The failed run is kept in testbed/results/m10d/r0_failed/ and is not rescored; only a calibration run
recorded after the revision (REVISION_1_NOT_BEFORE) is scored.

S1 (eval/testbed_scenarios.py, M08d): {LLM + RL, LLM + formula} x {offered, measured, throughput} x 3 repeats, played
by M08d's own player (eval.testbed_control.play) with the executor and the demand source as extra runner arguments.
Descriptive; the expectation written first, H-d: LLM + RL on the measured demand is within 0.05 of the same on the
offered rate, in S1 goal met (median of 3). "throughput" is M07b's blind run, expected to fail after the surge.
"""

import argparse
import csv
import json
import sqlite3
import subprocess
from pathlib import Path

import numpy as np

from eval import testbed_control as T
from eval.metrics import measure_run
from rl.runner import DEMANDS
from store.events import read_events
from xapps.kpm_monitor import KPM_PERIOD_S, UeReport, parse_line

TOLERANCE_MBPS = 1.0
TOLERANCE_REL = 0.05
SETTLE_REPORTS = 2  # reports dropped at a run's start (iperf3 starting mid-period, the queue filling)
STEP_REPORT = 2  # the third report after a step must be within tolerance
OVERLOAD_SERVICE_MAX = 0.5  # in the overload phase, throughput must stay at most this fraction of the offered rate
H_D_MARGIN = 0.05
REVISION_1_NOT_BEFORE = 1794865373.0  # Unix time of revision 1: earlier calibration data is refused (see the docstring)

CALIB_DIR = Path("testbed/results/m10d")
CALIB_OUT = Path("docs/learn/data/M10d-calibration.json")
RESULTS = Path("eval/results/m10d")
TABLE = Path("docs/learn/data/M10d-s1-runs.csv")
SUMMARY = Path("docs/learn/data/M10d-s1.json")
DEMAND_SERIES = Path("docs/learn/data/M10d-s1-measured-demand.csv")
EXECUTORS = ("rl", "formula")
CELLS = [(executor, demand) for executor in EXECUTORS for demand in DEMANDS]


# ---------------------------------------------------------------- calibration


def tolerance(target_mbps: float) -> float:
    return max(TOLERANCE_MBPS, TOLERANCE_REL * target_mbps)


def reports_by_slice(lines: list[str]) -> dict[str, list[tuple[float, float | None, float | None]]]:
    """Per slice, its report periods in time order: (period start s, PDCP Mb/s, throughput Mb/s), summed over UEs."""
    periods: dict[tuple[str, int], list[UeReport]] = {}
    for line in lines:
        r = parse_line(line)
        if isinstance(r, UeReport):
            periods.setdefault((r.slice, r.collect_start_us), []).append(r)
    out: dict[str, list] = {}
    for (s, start_us), ues in sorted(periods.items(), key=lambda kv: kv[0][1]):
        pdcp = [u.pdcp_dl_mb for u in ues if u.pdcp_dl_mb is not None]
        thr = [u.thp_dl_kbps for u in ues if u.thp_dl_kbps is not None]
        out.setdefault(s, []).append((start_us / 1e6, sum(pdcp) / KPM_PERIOD_S if pdcp else None,
                                      sum(thr) / 1000 if thr else None))
    return out


def in_run(reports: list, run: dict) -> list:
    """The report periods inside the run's sending window (revision 1): iperf3 sends for run["seconds"] from its start;
    after that a saturated slice only drains, with nothing arriving."""
    end = min(run["end_wall"], run["start_wall"] + run["seconds"])
    return [r for r in reports if run["start_wall"] <= r[0] and r[0] + KPM_PERIOD_S <= end]


def check(name: str, value: float | None, limit: float, ok: bool | None = None) -> dict:
    if value is None:
        return {"check": name, "value": None, "limit": limit, "ok": False, "why": "no PDCP values"}
    return {"check": name, "value": round(float(value), 3), "limit": round(limit, 3), "ok": bool(value <= limit if ok is None else ok)}


def calibration(runs: list[dict], reports: dict[str, list]) -> dict:
    """Every rule of the calibration, and whether all pass."""
    checks = []
    for run in runs:
        for s, target in run["rates_mbps"].items():
            mine = in_run(reports.get(s, []), run)
            settled = [r for r in mine[SETTLE_REPORTS:] if r[1] is not None]
            err = float(np.median([abs(r[1] - target) for r in settled])) if settled else None
            if run["phase"] == "steady":
                checks.append(check(f"steady {s} at {target:g} Mb/s: median |PDCP - target|", err, tolerance(target)))
            elif run["phase"] == "steps":
                third = mine[STEP_REPORT][1] if len(mine) > STEP_REPORT else None
                checks.append(check(f"step {s} to {target:g} Mb/s: |PDCP - target| at report {STEP_REPORT + 1}",
                                    None if third is None else abs(third - target), tolerance(target)))
            elif run["phase"] == "overload":
                checks.append(check(f"overload {s} at {target:g} Mb/s: median |PDCP - target|", err, tolerance(target)))
                served = [r[2] for r in mine[SETTLE_REPORTS:] if r[2] is not None]
                checks.append(check(f"overload {s}: median throughput (must stay well under the offered rate)",
                                    float(np.median(served)) if served else None, OVERLOAD_SERVICE_MAX * target))
    return {"pass": bool(checks) and all(c["ok"] for c in checks), "checks": checks}


def score_calibration(directory: Path = CALIB_DIR) -> dict:
    runs = [json.loads(line) for line in (directory / "calib_runs.jsonl").read_text().splitlines()]
    if min(run["start_wall"] for run in runs) < REVISION_1_NOT_BEFORE:
        raise SystemExit(f"{directory} was recorded before revision 1: run testbed/pdcp_check_m10d.py again")
    result = calibration(runs, reports_by_slice((directory / "calib_kpm.jsonl").read_text().splitlines()))
    result["revision"] = 1
    CALIB_OUT.parent.mkdir(parents=True, exist_ok=True)
    CALIB_OUT.write_text(json.dumps(result, indent=1) + "\n")
    return result


def require_calibration_pass() -> None:
    if not CALIB_OUT.exists() or not json.loads(CALIB_OUT.read_text())["pass"]:
        raise SystemExit(f"the PDCP calibration has not passed ({CALIB_OUT}): S1 is not played (rules in the docstring)")


# ---------------------------------------------------------------- the S1 matrix


def run_of(executor: str, demand: str, repeat: int) -> T.TestbedRun:
    """M08d's run description; the cell is in the study name, so the controller stays 'proposed' (LLM + agent)."""
    return T.TestbedRun(f"m10d_{executor}_{demand}", "s1", "proposed", repeat)


def extra_args(executor: str, demand: str) -> str:
    return ("--baseline formula " if executor == "formula" else "") + f"--demand {demand}"


def order(repeat: int) -> list[tuple[str, str]]:
    """The six cells, rotated per repeat so slow drift doesn't always land on the same cell."""
    shift = (repeat - 1) * 2 % len(CELLS)
    return CELLS[shift:] + CELLS[:shift]


def check_image() -> None:
    T.check_image()
    help_text = subprocess.run(["docker", "run", "--rm", "--entrypoint", "/opt/venv/bin/python", "-w", "/app", T.RL_IMAGE,
                                "-m", "rl.runner", "--help"], capture_output=True, text=True, check=False).stdout
    if "--demand" not in help_text:
        raise SystemExit(f"{T.RL_IMAGE} has no --demand: run xapps/rl/build.sh")


def play_matrix() -> None:
    for repeat in range(1, T.REPEATS + 1):
        missing = [c for c in order(repeat) if not (RESULTS / run_of(*c, repeat).run_id / "run.json").exists()]
        if not missing:
            continue
        T.restart_testbed()
        for i, (executor, demand) in enumerate(missing, 1):
            work = T.play(run_of(executor, demand, repeat), extra_args=extra_args(executor, demand), results=RESULTS)
            spec = json.loads((work / "run.json").read_text())
            print(f"[repeat {repeat}, {i}/{len(missing)}] {executor} {demand}: {spec['wall_s']:.0f} s, valid {spec['valid']} "
                  f"{spec['problems']}", flush=True)
    replay_invalid()


def replay_invalid() -> None:
    """Once only, as M08d: the invalid run stays on disk under another name, and its replay counts whatever it shows."""
    for path in sorted(RESULTS.glob("m10d_*/s1/proposed/repeat?/run.json")):
        work = path.parent
        if json.loads(path.read_text())["valid"] or work.with_name(work.name + "_invalid").exists():
            continue
        executor, demand = work.parts[-4].removeprefix("m10d_").split("_", 1)
        repeat = int(work.name.removeprefix("repeat"))
        work.rename(work.with_name(work.name + "_invalid"))
        T.restart_testbed()
        T.play(run_of(executor, demand, repeat), extra_args=extra_args(executor, demand), results=RESULTS)
        print(f"replayed {work}", flush=True)


# ---------------------------------------------------------------- S1 scoring


def score_runs(results: Path = RESULTS) -> list[dict]:
    rows = []
    for path in sorted(results.glob("m10d_*/s1/proposed/repeat*/run.json")):
        spec, work = json.loads(path.read_text()), path.parent
        executor, demand = work.parts[-4].removeprefix("m10d_").split("_", 1)
        played = T.scenario_as_played("s1", read_events(work / "events.jsonl"))
        intent = measure_run(work, played)["intents"][0]
        rows.append({"executor": executor, "demand": demand, "repeat": spec["repeat"], "valid": spec["valid"],
                     "invalid_copy": work.name.endswith("_invalid"), "intent_met": intent["intent_met"],
                     "reaction_s": intent.get("reaction_s"), "problems": "; ".join(spec["problems"])})
    return rows


def summarise(rows: list[dict]) -> dict:
    """Median intent met per cell over the valid runs (an invalid run's replay replaces it), and H-d."""
    counted = [r for r in rows if r["valid"] and not r["invalid_copy"]]
    medians = {}
    for executor, demand in CELLS:
        values = [r["intent_met"] for r in counted if r["executor"] == executor and r["demand"] == demand]
        medians[f"{executor}/{demand}"] = {"median": float(np.median(values)) if values else None, "n": len(values)}
    rl_offered, rl_measured = medians["rl/offered"]["median"], medians["rl/measured"]["median"]
    gap = None if rl_offered is None or rl_measured is None else rl_measured - rl_offered
    return {"medians": medians, "h_d": {"rl_measured_minus_offered": gap, "margin": H_D_MARGIN,
                                        "holds": None if gap is None else abs(gap) <= H_D_MARGIN}}


def sent_windows(work: Path) -> dict[str, list[tuple[float, float, float]]]:
    """Per slice, when its iperf3 sessions really sent, in run seconds: (start, end, Mb/s). A session starts when
    iperf3's control handshake completes ("start.timestamp"), which on a slice with a long DL queue is seconds
    after the segment's clock (the handshake waits in that queue), then sends for the segment's length."""
    t0 = json.loads((work / "run.json").read_text())["t0"]
    segments = [e["data"] for e in read_events(work / "events.jsonl") if e["event"] == "segment_start"]
    out: dict[str, list] = {}
    for seg in segments:
        for s, rate in seg["rates_mbps"].items():
            start = json.loads((work / f"seg{seg['segment']}_{s}.json").read_text())["start"]["timestamp"]["timesecs"] - t0
            out.setdefault(s, []).append((start, start + seg["seconds"], rate))
    return out


def export_demand_series(work: Path, db: Path, out: Path = DEMAND_SERIES) -> int:
    """The S1 run's three demand sources and what iperf3 really sent, from its run directory and a copy of the
    testbed database taken right after it (the kpi table holds only the last run, and kpi.csv leaves the measured
    column out so earlier runs' files stay byte-identical). Returns the rows written."""
    t0 = json.loads((work / "run.json").read_text())["t0"]
    windows = sent_windows(work)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute("SELECT ts, slice, dl_thr_mbps, offered_load_mbps, measured_load_mbps FROM kpi ORDER BY ts, slice")
    n = 0
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "slice", "dl_thr_mbps", "offered_load_mbps", "measured_load_mbps", "sent_mbps"])
        for ts, s, thr, offered, measured in rows:
            t = ts - t0
            if 0 <= t <= T.SCENARIOS["s1"].seconds:
                sent = sum(rate for start, end, rate in windows.get(s, []) if start <= t < end)
                writer.writerow([f"{t:.3f}", s, thr, offered, measured, sent])
                n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--calibration", action="store_true")
    mode.add_argument("--final", action="store_true")
    mode.add_argument("--score", action="store_true")
    mode.add_argument("--demand-series", nargs=2, metavar=("RUN_DIR", "DB_COPY"),
                      help=f"write {DEMAND_SERIES} from one S1 run and the testbed database copied right after it")
    args = parser.parse_args()
    if args.demand_series:
        print(export_demand_series(Path(args.demand_series[0]), Path(args.demand_series[1])), "rows ->", DEMAND_SERIES)
        return
    if args.calibration:
        result = score_calibration()
        for c in result["checks"]:
            print(f"{'ok  ' if c['ok'] else 'FAIL'} {c['check']}: {c['value']} (limit {c['limit']})")
        print(f"calibration {'PASSES' if result['pass'] else 'FAILS'}; written to {CALIB_OUT}")
    elif args.final:
        T.guard_final(T.commit())
        require_calibration_pass()
        check_image()
        play_matrix()
        subprocess.run(["testbed/down.sh"], check=False, capture_output=True)
    else:
        rows = score_runs()
        if rows:
            TABLE.parent.mkdir(parents=True, exist_ok=True)
            with TABLE.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        summary = summarise(rows)
        SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
