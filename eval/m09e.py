"""M09e: the testbed checks M09d's written predictions (rules: docs/DECISIONS.md, 2026-11-11, M09e pre-registration).

  uv run python -m eval.m09e --dev [--instances f1_00 f3_00] [--controllers formula] [--until 120]   dry run, dev instances
  uv run python -m eval.m09e --final            the 8 M09d instances x 4 controllers x 2 repeats, one night; resumes if stopped
  uv run python -m eval.m09e --score --final    the pass rule against docs/learn/data/M09d-predictions.csv, ONCE

The testbed must be up (testbed/up.sh; `sudo modprobe sctp` after a WSL restart), Ollama on the GPU, and the RL xApp
image built from this code (xapps/rl/build.sh): --final restarts the testbed at the start of each repeat, as M08d did.

What differs from M08d's player (eval/testbed_control.py, kept unchanged for M08d's record): a generated instance
changes one slice's rate many times (F3: 18-19 URLLC phases of 10-30 s; F1a ramps in 5 s steps), so each slice gets a
new iperf3 only when its own rate changes, on its own port, and the clients run in one long-lived helper container
per slice (`docker exec` starts in a fraction of a second; `docker run` took about one). Every process still stops
itself DRAIN_S after the scenario's end. The verifier and the formula are told the M09a calibrated URLLC base
latency, as the M09 simulator profile that made the predictions was (M08c and M08d told 2 ms). As in M08, every
controller reads each slice's offered load from the traffic script (traffic_target): KPM doesn't report it.
Each run also needs the RF simulator near real time (gNB frame counter): it doesn't start below 0.95x (idle), and it
is invalid below 0.75x over the scenario's seconds (under traffic a healthy host runs it at 0.83-0.90x).
Raw runs: eval/results/m09e/ (git-ignored). Tables: docs/learn/data/M09e-*.csv.
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from eval import testbed_control as T
from eval.generator import SECONDS, Instance, draw, generate
from eval.interface import wait_for_service
from eval.m09d import M09E_CONTROLLERS, PREDICTIONS_CSV, TIE, guard, m09e_sample
from eval.metrics import measure_run
from eval.realtime_check import start
from eval.sim_profiles import calibration
from eval.spent import mark_spent
from eval.validate_sim import CELL_TOLERANCE
from rl.runner import SERVED_SMOOTHING_ALPHA, clear_run_state
from store import db, testbed
from store.events import log_event

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "testbed"))  # for traffic.py
import traffic  # noqa: E402

RESULTS = Path("eval/results/m09e")
STUDY, DEV_STUDY = "testbed", "dev"
SPENT_NAME = "m09e/testbed"
DATA = Path("docs/learn/data")
RUNS_CSV, SCORES_CSV, PAIRS_CSV = (DATA / f"M09e-{n}.csv" for n in ("runs", "scores", "pairs"))
RESULT = RESULTS / "result.json"

CONTROLLERS = M09E_CONTROLLERS  # static, llm_only_v2, formula, proposed
REPEATS = 2
TOLD_URLLC_BASE_MS = calibration()["base_latency_ms_urllc"]  # 0.592 ms, what the M09 simulator profile tells
# How each controller is started: the runner's arguments (RUNNER_ARGS of the rl-xapp container) and the agent's mode.
RUNNER_BASELINE = {"static": "--baseline equal", "llm_only_v2": "--baseline equal", "formula": "--baseline formula", "proposed": ""}
AGENT_ENV = {"llm_only_v2": {"RIC_CONTROLLER": "llm_only", "RIC_LLM_ONLY_PROMPT": "v2", "RIC_LLM_ONLY_JOINS": "1"},
             "formula": {"RIC_CONTROLLER": "proposed"}, "proposed": {"RIC_CONTROLLER": "proposed"}}
# The pass rule (pre-registered): the testbed's best controller in the predicted "best" set in >= BEST_MIN of 8
# instances, and >= PAIRS_MIN of the 24 pairs (LLM + RL minus each other) confirmed. A predicted tie is confirmed
# within CELL_TOLERANCE (M09a's 0.15), a predicted +/- by the same sign beyond TIE (0.05).
BEST_MIN, PAIRS_MIN = 6, 0.75
# The RF simulator has no clock of its own: it runs as fast as the host lets it. On 2026-11-11 it ran at 0.47x real
# time (host CPU throttled) and the cell carried 37 Mb/s per unit share instead of M08d's 76, with nothing in the
# validity rules to notice. So each run measures it from the gNB's frame counter (10 ms per frame).
GNB = "rfsim5g-oai-gnb"
MIN_RFSIM_SPEED = 0.95  # idle, before a run (healthy host: 0.99-1.06)
MIN_RFSIM_SPEED_RUN = 0.75  # under traffic, over the scenario (healthy host on 2026-11-13: 0.83-0.90)
SPEED_WINDOW_S = 15
FRAME_S, FRAMES_PER_WRAP = 0.010, 1024


@dataclass(frozen=True)
class Piece:
    """One iperf3 client: a slice at one rate, from start_s for `seconds`. Its port is its slice's port plus
    SEGMENT_PORT_STEP x index, so a client still draining its queue never holds the port of the next one."""

    slice: str
    index: int
    start_s: float
    seconds: int
    rate_mbps: float

    @property
    def port(self) -> int:
        return traffic.SLICES[self.slice]["port"] + traffic.SEGMENT_PORT_STEP * self.index


def pieces(steps: dict, until_s: float = SECONDS) -> list[Piece]:
    """The instance's traffic as iperf3 clients: per slice, one client per run of equal rate (two steps at the same
    rate stay one client), cut at until_s; in start order."""
    out = []
    for s, levels in steps.items():
        merged: list[list[float]] = []
        for at, rate in levels:
            if at >= until_s:
                break
            if merged and merged[-1][1] == rate:
                continue
            merged.append([float(at), float(rate)])
        for k, (at, rate) in enumerate(merged):
            end = merged[k + 1][0] if k + 1 < len(merged) else until_s
            out.append(Piece(s, k, at, int(round(end - at)), rate))
    return sorted(out, key=lambda p: (p.start_s, p.slice))


def order(repeat: int, names: list[str]) -> list[tuple[str, str]]:
    """All (instance, controller) pairs, rotated per repeat as in M08d, so slow drift doesn't always fall on the same
    run. With 8 instances x 4 controllers = 32 pairs, a shift of 5 (coprime with 32) moves every pair."""
    pairs = [(n, c) for n in names for c in CONTROLLERS]
    shift = 5 * (repeat - 1) % len(pairs)
    return pairs[shift:] + pairs[:shift]


def runner_args(controller: str) -> str:
    return f"{RUNNER_BASELINE[controller]} --urllc-base-latency-ms {TOLD_URLLC_BASE_MS:g}".strip()


def controller_env(controller: str) -> dict[str, str]:
    return {"RIC_MODE": "eval", "RIC_CONTROLLER": "proposed", "RUNNER_ARGS": runner_args(controller), **AGENT_ENV.get(controller, {})}


# ---------------------------------------------------------------- the traffic


def helper(name: str) -> str:
    return f"traffic-{name}"


def start_helpers() -> None:
    """One idle netshoot container per slice in that UE's network namespace; every client of the run is a
    `docker exec` into it."""
    stop_helpers()
    for name, spec in traffic.SLICES.items():
        out = subprocess.run(["docker", "run", "-d", "--rm", "--name", helper(name), "--net", f"container:{spec['container']}",
                              "nicolaka/netshoot", "sleep", "infinity"], capture_output=True, text=True, check=False)
        if out.returncode != 0:
            raise RuntimeError(f"could not start {helper(name)}: {out.stderr.strip()}")


def require_testbed() -> None:
    """--dev doesn't restart the testbed (--final does): stop here, with the reason, if it isn't running."""
    running = set(subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=False).stdout.split())
    missing = [name for name in T.WATCHED if name not in running]
    if missing:
        raise SystemExit(f"the testbed isn't up ({', '.join(missing)} not running): sudo modprobe sctp, then testbed/up.sh")


FRAME_MARK = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z .*Frame\.Slot (\d+)\.0\b", re.MULTILINE)


def frame_marks(log: str) -> list[tuple[float, int]]:
    """(wall time, frame) of each `Frame.Slot F.0` line of `docker logs --timestamps` output. The gNB's MAC prints
    one every 128 frames. Docker gives nanoseconds; Python parses microseconds, so the fraction is cut to 6 digits."""
    marks = []
    for date, fraction, frame in FRAME_MARK.findall(log):
        stamp = datetime.fromisoformat(f"{date}{(fraction or '.0')[:7]}+00:00")
        marks.append((stamp.timestamp(), int(frame)))
    return marks


def rfsim_speed(marks: list[tuple[float, int]]) -> float | None:
    """Simulated seconds per wall second: 1.0 is real time. Frames wrap at 1024; consecutive marks are 128 apart,
    so the forward step modulo 1024 is the true one. None with fewer than two marks (nothing to measure)."""
    if len(marks) < 2 or marks[-1][0] <= marks[0][0]:
        return None
    frames = sum((b - a) % FRAMES_PER_WRAP for (_, a), (_, b) in zip(marks, marks[1:]))
    return round(frames * FRAME_S / (marks[-1][0] - marks[0][0]), 3)


def measure_rfsim_speed(since: float, until: float | None = None) -> float | None:
    """The speed over the gNB's log between two wall times (Unix seconds)."""
    window = ["--since", f"{since:.3f}"] + ([] if until is None else ["--until", f"{until:.3f}"])
    out = subprocess.run(["docker", "logs", "--timestamps", *window, GNB], capture_output=True, text=True, check=False)
    return rfsim_speed(frame_marks(out.stdout + out.stderr))


def require_real_time() -> float:
    """Before a run: refuse to start on a slow simulator (the matrix resumes where it stopped)."""
    speed = measure_rfsim_speed(time.time() - SPEED_WINDOW_S)
    if speed is None or speed < MIN_RFSIM_SPEED:
        raise SystemExit(f"the RF simulator runs at {speed}x real time over the last {SPEED_WINDOW_S} s "
                         f"(needs >= {MIN_RFSIM_SPEED}): check the host's power mode, then run again")
    return speed


def speed_problems(speed: float | None) -> list[str]:
    """During a run: a slow or unmeasured simulator makes the run invalid (replayed once, as any invalid run)."""
    if speed is None:
        return ["RF simulator speed not measurable (no gNB frame marks)"]
    if speed < MIN_RFSIM_SPEED_RUN:
        return [f"RF simulator at {speed}x real time (< {MIN_RFSIM_SPEED_RUN})"]
    return []


def stop_helpers() -> None:
    subprocess.run(["docker", "rm", "-f", *(helper(n) for n in traffic.SLICES)], capture_output=True, check=False)


def start_servers(plan: list[Piece]) -> None:
    """One iperf3 server per client port on ext-dn, in one exec. A port already served fails to bind: harmless."""
    ports = " ".join(str(p.port) for p in plan)
    traffic.docker("exec", traffic.EXT_DN, "sh", "-c", f"for p in {ports}; do iperf3 -s -D -p $p; done", check=False)
    time.sleep(1)


def client(piece: Piece, ip: str, deadline_s: float) -> list[str]:
    return ["docker", "exec", helper(piece.slice), "timeout", "-s", "TERM", str(math.ceil(deadline_s)),
            "iperf3", "-c", traffic.EXT_DN_IP, "-p", str(piece.port), "-B", ip, "-u", "-b", f"{piece.rate_mbps:g}M", "-R",
            "-t", str(piece.seconds), "-J"]


def play_traffic(plan: list[Piece], ips: dict[str, str], work: Path, errors: list[str], lags: list[float], t0: float,
                 deadline: float) -> None:
    """One thread, on the scenario's clock. At each start time: write the offered load the controllers read
    (traffic_target, all slices), then start the clients that begin then. Records each client's start lag.
    The UE addresses are looked up before t0 (three docker execs made the dry run's t=0 clients 0.6 s late)."""
    conn = db.connect(Path(T.TESTBED_DB))
    rates: dict[str, float] = {}
    procs, outputs = [], []
    for at in sorted({p.start_s for p in plan}):
        time.sleep(max(0.0, t0 + at - time.time()))
        group = [p for p in plan if p.start_s == at]
        rates |= {p.slice: p.rate_mbps for p in group}
        testbed.write_traffic_target(conn, rates)
        for p in group:
            out = work / f"iperf_{p.slice}_{p.index:02d}.json"
            procs.append(subprocess.Popen(client(p, ips[p.slice], deadline - time.time()), stdout=out.open("w"),
                                          stderr=subprocess.DEVNULL, text=True))
            lags.append(time.time() - (t0 + at))
            outputs.append(out)
            log_event("scenario", "segment_start", data={"slice": p.slice, "piece": p.index, "rate_mbps": p.rate_mbps,
                                                         "seconds": p.seconds, "wall": time.time(), "planned_s": at})
    conn.close()
    for proc in procs:
        proc.wait()
    errors += [e for path in outputs if (e := T.iperf_failure(path))]


# ---------------------------------------------------------------- one run


def work_of(name: str, controller: str, repeat: int, study: str = STUDY) -> Path:
    return RESULTS / study / name / controller / f"repeat{repeat}"


def play(inst: Instance, controller: str, repeat: int, study: str = STUDY, until_s: float = SECONDS) -> Path:
    """One run, from a clean store; M08d's sequence (eval/testbed_control.play) with this module's traffic.
    Writes run.json last: without it the run didn't finish and is played again."""
    work = work_of(inst.name, controller, repeat, study)
    speed_before = require_real_time()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    os.environ["RIC_DB"] = T.TESTBED_DB
    env = {**os.environ, **controller_env(controller)}
    uses_agent = controller in AGENT_ENV
    scenario = inst.scenario()
    seconds = min(until_s, scenario.seconds)
    plan = pieces(inst.steps, seconds)
    py = sys.executable
    conn = db.connect(Path(T.TESTBED_DB))
    clear_run_state(conn)
    conn.execute("DELETE FROM kpm_latest")
    conn.execute("DELETE FROM traffic_target")
    started, began = time.time(), time.monotonic()
    before = T.restart_counts()
    processes = [start([py, "-m", "xapps.kpm_monitor", "--for-runner"], env, work / "kpm_monitor.out")]
    errors: list[str] = []
    lags: list[float] = []
    t0 = time.time()
    try:
        start_helpers()
        start_servers(plan)
        time.sleep(3)
        subprocess.run([*T.COMPOSE, "up", "-d", "rl-xapp"], env=env, check=True, capture_output=True)
        T.wait_ready(conn, started, work)
        if uses_agent:
            log_event("scenario", "llm_warmup", data={"run": str(work), "latency_s": round(T.warm_up_llm(), 2)})
            processes.append(start([py, "-m", "uvicorn", "agent.service:app", "--port", str(T.PORT)], env, work / "service.out"))
            processes.append(start([py, "-m", "agent.worker"], env, work / "worker.out"))
            wait_for_service(T.PORT)
        ips = {s: traffic.ue_ip(spec["container"]) for s, spec in traffic.SLICES.items()}
        t0 = time.time()
        deadline = t0 + seconds + T.DRAIN_S
        ping_procs = [traffic.ping(s, int(seconds), work / f"ping_{s}.txt", deadline_s=deadline - time.time()) for s in traffic.SLICES]
        log_event("scenario", "scenario_start", data={"run": str(work), "t0": t0, "instance": inst.name, "controller": controller})
        player = threading.Thread(target=play_traffic, args=(plan, ips, work, errors, lags, t0, deadline), daemon=True)
        player.start()
        if uses_agent:
            for intent in scenario.intents:  # on their own clock: a late client start can't delay them
                if intent.at_s >= seconds:
                    break
                time.sleep(max(0.0, t0 + intent.at_s - time.time()))
                T.submit(intent.text)
        time.sleep(max(0.0, t0 + seconds - time.time()))
        player.join(timeout=max(0.0, deadline + T.STOP_GRACE_S - time.time()))
        if player.is_alive():
            errors.append(f"a traffic process was still running {T.STOP_GRACE_S:g} s after its deadline")
        for proc in ping_procs:
            proc.wait(timeout=max(1.0, deadline + T.STOP_GRACE_S - time.time()))
        conn.execute("DELETE FROM traffic_target")
        stopped = sorted(p.name for p in work.glob("iperf_*.json") if T.stopped_by_deadline(p))
        drained = T.wait_quiet(conn)
        log_event("scenario", "scenario_stop", data={"run": str(work), "drained": drained, "stopped_at_deadline": stopped,
                                                     "closed_after_s": round(time.time() - (t0 + seconds), 1)})
    finally:
        subprocess.run([*T.COMPOSE, "stop", "rl-xapp"], env=env, check=False, capture_output=True)
        logs = subprocess.run(["docker", "logs", "rl-xapp"], capture_output=True, text=True, check=False)
        (work / "rl_xapp.out").write_text(logs.stdout[-20000:] + logs.stderr[-20000:])
        subprocess.run([*T.COMPOSE, "rm", "-f", "rl-xapp"], env=env, check=False, capture_output=True)
        stop_helpers()
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(timeout=20)
    events = T.export(conn, work, started, t0, seconds)
    conn.close()
    speed_run = measure_rfsim_speed(t0, t0 + seconds)
    invalid = T.problems(events, before, T.restart_counts(), errors) + speed_problems(speed_run)
    (work / "run.json").write_text(json.dumps(
        {"study": study, "instance": inst.name, "family": inst.family, "seed": inst.seed, "controller": controller,
         "repeat": repeat, "alpha": SERVED_SMOOTHING_ALPHA, "told_urllc_base_ms": TOLD_URLLC_BASE_MS,
         "runner_args": runner_args(controller), "commit": T.commit(), "t0": t0, "seconds": seconds,
         "clients": len(plan), "max_start_lag_s": round(max(lags), 2) if lags else None,
         "rfsim_speed_before": speed_before, "rfsim_speed_run": speed_run,
         "wall_s": round(time.monotonic() - began, 1), "valid": not invalid, "problems": invalid}, indent=1))
    return work


# ---------------------------------------------------------------- the matrix


def check_image() -> None:
    """The container must run this code: M08d's check, plus the two runner options M09e uses."""
    T.check_image()
    out = subprocess.run(["docker", "run", "--rm", "--entrypoint", "/opt/venv/bin/python", "-w", "/app", T.RL_IMAGE,
                          "-m", "rl.runner", "--help"], capture_output=True, text=True, check=False)
    if "formula" not in out.stdout or "--urllc-base-latency-ms" not in out.stdout:
        raise SystemExit(f"{T.RL_IMAGE} has no --baseline formula / --urllc-base-latency-ms: run xapps/rl/build.sh")


def sample() -> list[Instance]:
    return [draw(f, i, "test") for f, i in m09e_sample()]


def report(work: Path) -> str:
    spec = json.loads((work / "run.json").read_text())
    return f"{spec['wall_s']:.0f} s, start lag max {spec['max_start_lag_s']} s, valid {spec['valid']} {spec['problems']}"


def play_matrix() -> None:
    instances = {i.name: i for i in sample()}
    for repeat in range(1, REPEATS + 1):
        missing = [(n, c) for n, c in order(repeat, list(instances)) if not (work_of(n, c, repeat) / "run.json").exists()]
        if not missing:
            continue
        T.restart_testbed()
        for k, (name, controller) in enumerate(missing, 1):
            work = play(instances[name], controller, repeat)
            print(f"[repeat {repeat}, {k}/{len(missing)}] {name} {controller}: {report(work)}", flush=True)
    replay_invalid(instances)


def replay_invalid(instances: dict[str, Instance]) -> None:
    """Once only, as M08d: the invalid run stays on disk as repeatN_invalid, and its replay counts whatever it shows."""
    for path in sorted(RESULTS.glob(f"{STUDY}/*/*/repeat?/run.json")):
        spec = json.loads(path.read_text())
        work = path.parent
        if spec["valid"] or work.with_name(work.name + "_invalid").exists():
            continue
        work.rename(work.with_name(work.name + "_invalid"))
        T.restart_testbed()
        replayed = play(instances[spec["instance"]], spec["controller"], spec["repeat"])
        print(f"replayed {work}: {report(replayed)}", flush=True)


# ---------------------------------------------------------------- the pass rule


def run_rows(instances: dict[str, Instance]) -> list[dict]:
    """One row per run (not the _invalid copies) and intent, measured like every simulator run (eval.metrics)."""
    rows = []
    for path in sorted(RESULTS.glob(f"{STUDY}/*/*/repeat?/run.json")):
        spec = json.loads(path.read_text())
        inst = instances[spec["instance"]]
        measured = measure_run(path.parent, inst.scenario())
        rows += [{"instance": inst.name, "family": inst.family, "controller": spec["controller"], "repeat": spec["repeat"],
                  "valid": spec["valid"], "problems": "; ".join(spec["problems"]), "max_start_lag_s": spec["max_start_lag_s"],
                  **intent, **measured["run"]} for intent in measured["intents"]]
    return rows


def scores(rows: list[dict]) -> dict[tuple[str, str], float]:
    """Per instance and controller: the mean over repeats of the run's mean intent met (M09d's prediction unit)."""
    per_run: dict[tuple[str, str, int], list[float]] = {}
    for r in rows:
        per_run.setdefault((r["instance"], r["controller"], r["repeat"]), []).append(float(r["intent_met"]))
    per_pair: dict[tuple[str, str], list[float]] = {}
    for (name, controller, _), met in per_run.items():
        per_pair.setdefault((name, controller), []).append(float(np.mean(met)))
    return {k: float(np.mean(v)) for k, v in per_pair.items()}


def confirmed(predicted: str, difference: float) -> bool:
    """A predicted tie holds within CELL_TOLERANCE; a predicted sign holds beyond TIE, in that direction."""
    if predicted == "tie":
        return abs(difference) <= CELL_TOLERANCE
    return difference > TIE if predicted == "+" else difference < -TIE


def judge(measured: dict[tuple[str, str], float], predictions: list[dict]) -> dict:
    """The pre-registered pass rule. `predictions` are the rows of M09d-predictions.csv."""
    names = list(dict.fromkeys(p["instance"] for p in predictions))
    best_rows, pair_rows = [], []
    for name in names:
        mine = [p for p in predictions if p["instance"] == name]
        predicted_best = {p["controller"] for p in mine if p["best"] in ("True", True)}
        top = max(measured[(name, c)] for c in CONTROLLERS)
        testbed_best = {c for c in CONTROLLERS if measured[(name, c)] >= top - 1e-9}
        best_rows.append({"instance": name, "predicted_best": " ".join(sorted(predicted_best)),
                          "testbed_best": " ".join(sorted(testbed_best)), "match": bool(predicted_best & testbed_best),
                          **{c: round(measured[(name, c)], 3) for c in CONTROLLERS}})
        for p in mine:
            if p["controller"] == "proposed":
                continue
            difference = measured[(name, "proposed")] - measured[(name, p["controller"])]
            pair_rows.append({"instance": name, "other": p["controller"], "predicted": p["proposed_minus_this"],
                              "testbed_difference": round(difference, 3), "confirmed": confirmed(p["proposed_minus_this"], difference),
                              "weak": p["controller"] == "llm_only_v2"})
    n_best = sum(r["match"] for r in best_rows)
    share = float(np.mean([r["confirmed"] for r in pair_rows]))
    strong = [r["confirmed"] for r in pair_rows if not r["weak"]]
    return {"instances": best_rows, "pairs": pair_rows, "best_matches": n_best, "pairs_confirmed": round(share, 4),
            "pairs_confirmed_without_llm_only": round(float(np.mean(strong)), 4),
            "pass": n_best >= BEST_MIN and share >= PAIRS_MIN}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def score(final: bool, register: Path | None = None) -> dict:
    if not final:
        raise SystemExit("the testbed sample is scored once: pass --final when you mean it")
    revision = guard(register, SPENT_NAME)
    instances = {i.name: i for i in sample()}
    missing = [str(work_of(n, c, r)) for n in instances for c in CONTROLLERS for r in range(1, REPEATS + 1)
               if not (work_of(n, c, r) / "run.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} runs missing (first: {missing[0]}); finish --final first")
    rows = run_rows(instances)
    write_csv(RUNS_CSV, rows)
    with PREDICTIONS_CSV.open() as fh:
        predictions = list(csv.DictReader(fh))
    verdict = judge(scores(rows), predictions)
    write_csv(SCORES_CSV, verdict["instances"])
    write_csv(PAIRS_CSV, verdict["pairs"])
    invalid = sum(1 for _ in RESULTS.glob(f"{STUDY}/*/*/repeat?_invalid"))
    result = {"commit_score": revision, "pass": verdict["pass"], "best_matches": verdict["best_matches"],
              "pairs_confirmed": verdict["pairs_confirmed"], "pairs_confirmed_without_llm_only": verdict["pairs_confirmed_without_llm_only"],
              "still_invalid_after_replay": sorted({f"{r['instance']} {r['controller']} {r['repeat']}" for r in rows if not r["valid"]}),
              "replayed": invalid, "told_urllc_base_ms": TOLD_URLLC_BASE_MS}
    RESULT.write_text(json.dumps(result, indent=1))
    mark_spent(SPENT_NAME, RESULT, f"M09e: pass {verdict['pass']}, best {verdict['best_matches']}/8, "
               f"pairs {verdict['pairs_confirmed']:.2f}", register)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev", action="store_true", help="dev instances, never in the table; the testbed must be up")
    parser.add_argument("--final", action="store_true", help="the matrix (or, with --score, the pass rule)")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--instances", nargs="+", default=["f1_00", "f3_00"], help="--dev: dev instances, e.g. f3_00")
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS))
    parser.add_argument("--until", type=float, default=120.0, help="--dev: play only the first seconds")
    args = parser.parse_args()
    if args.score:
        print(json.dumps(score(args.final), indent=1))
        return
    if args.dev == args.final:
        raise SystemExit("choose --dev or --final")
    if args.final:
        guard(None, SPENT_NAME)
    check_image()
    if args.dev:
        require_testbed()
        for name in args.instances:
            family, index = name.split("_")
            for controller in args.controllers:
                work = play(generate(family, int(index)), controller, 1, DEV_STUDY, args.until)
                print(f"{name} {controller}: {report(work)}", flush=True)
        return
    play_matrix()
    subprocess.run(["testbed/down.sh"], check=False, capture_output=True)
    print("done: now uv run python -m eval.m09e --score --final", flush=True)


if __name__ == "__main__":
    main()
