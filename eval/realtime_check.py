"""Is lockstep time fair? One S2 run per LLM controller as the real separate processes, at real speed.

  uv run python -m eval.realtime_check        about 13 minutes; needs Ollama and port 8000 free
  uv run python -m eval.realtime_check --smoke  10 x faster, to test this script; not in the table

The runner (speed 1), the agent service and the worker run as in the live demo, in evaluation mode,
each with a temporary database, and the intent arrives through POST /intents at t=120 s. The result
is measured by the same code as eval/harness.py's runs and printed next to their spread: it should
fall inside it. Goes to eval/results/m08a/realtime/ and to the table as study "realtime".
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from eval.control import RESULTS, write_table
from eval.harness import export_kpis
from eval.metrics import measure_run
from eval.scenarios import SCENARIOS, SEEDS
from store import db, kpi

PORT = 8000
SCENARIO = SCENARIOS["s2"]
SEED = SEEDS[0]


def start(command: list[str], env: dict, log: Path) -> subprocess.Popen:
    return subprocess.Popen(command, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT)


def run_realtime(controller: str, study: str, speed: float) -> Path:
    work = RESULTS / study / SCENARIO.name / controller / f"seed{SEED}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "events.jsonl").unlink(missing_ok=True)
    env = {**os.environ, "RIC_DB": str(work / "ric.db"), "RIC_EVENTS": str(work / "events.jsonl"),
           "RIC_MODE": "eval", "RIC_CONTROLLER": controller}
    py = sys.executable
    baseline = ["--baseline", "equal"] if controller == "llm_only" else []
    processes = [start([py, "-m", "rl.runner", "--scenario", SCENARIO.name, "--speed", str(speed), "--seed", str(SEED),
                        "--ticks", str(SCENARIO.seconds), *baseline], env, work / "runner.out")]
    try:
        conn = None
        deadline = time.monotonic() + 90
        while conn is None or (kpi.latest_ts(conn) or 0.0) < 1.0:  # the runner clears the db when it starts
            if time.monotonic() > deadline:
                raise TimeoutError(f"the runner did not start; see {work / 'runner.out'}")
            time.sleep(0.5)
            conn = conn or (db.connect(work / "ric.db") if (work / "ric.db").exists() else None)
        processes.append(start([py, "-m", "uvicorn", "agent.service:app", "--port", str(PORT)], env, work / "service.out"))
        processes.append(start([py, "-m", "agent.worker"], env, work / "worker.out"))
        intent = SCENARIO.intents[0]
        while kpi.latest_ts(conn) < intent.at_s:
            time.sleep(0.2)
        request = urllib.request.Request(f"http://127.0.0.1:{PORT}/intents", json.dumps({"text": intent.text}).encode(),
                                         {"content-type": "application/json"})
        urllib.request.urlopen(request, timeout=10).read()
        processes[0].wait()
        export_kpis(conn, work / "kpi.csv")
        conn.close()
    finally:
        for process in processes:
            process.terminate()
    for leftover in ("ric.db", "ric.db-wal", "ric.db-shm"):
        (work / leftover).unlink(missing_ok=True)
    (work / "run.json").write_text(json.dumps(
        {"study": study, "scenario": SCENARIO.name, "controller": controller, "seed": SEED, "alpha": 0.25,
         "delay_ticks": 0, "importance": None, "wall_s": SCENARIO.seconds}, indent=1))
    return work


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    study, speed = ("smoke", 10.0) if args.smoke else ("realtime", 1.0)
    for controller in ("proposed", "llm_only"):
        print(f"{controller}: {SCENARIO.seconds} s at speed {speed:g}...", flush=True)
        measured = measure_run(run_realtime(controller, study, speed), SCENARIO)["intents"][0]
        runs = [measure_run(p.parent, SCENARIO)["intents"][0]["intent_met"]
                for p in (RESULTS / "headline" / SCENARIO.name / controller).glob("seed*/run.json")]
        spread = f"{min(runs):.2f}-{max(runs):.2f} over {len(runs)} runs" if runs else "no lockstep runs yet"
        print(f"  intent met {measured['intent_met']:.2f} (lockstep: {spread}); first apply {measured['first_apply_s']} s; "
              f"URLLC share {np.round(measured['urllc_share'], 2)}")
    print(f"{write_table()} rows in the table")


if __name__ == "__main__":
    main()
