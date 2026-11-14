"""The control comparison on the real cell: the four controllers, at real speed, as separate processes (M08c, M08d).

  uv run python -m eval.testbed_control --dev [--scale 0.2] [--scenarios s3] [--controllers proposed] [--what-if]
  uv run python -m eval.testbed_control --final      3 repeats of the 12 runs, then the S3 what-if; about 4.5 hours; resumes if stopped

--final restarts the testbed at the start of each repeat and stops it at the end. It refuses uncommitted code and
an RL xApp image older than this code. Every run is measured by eval.metrics, like the simulator's: its logs are
rebased so the scenario starts at t=0. A run is invalid only if a container restarted, the xApp exited, KPM went
stale or the traffic machinery failed (an iperf3 error other than its own deadline, no iperf3 result, or a traffic
process still running after its deadline); a bad score never invalidates a run. Invalid runs are kept and replayed
once at the end.

After M08d's matrix (fixes recorded in docs/DECISIONS.md): every traffic process stops itself DRAIN_S after the
scenario's end, so a slow queue is a result, not a failure, and no process outlives its run; only play() clears
traffic_target; and the model is warmed up before an LLM run's clock starts.

The what-if is exploratory and stays out of the main comparison: LLM + RL on S3 with the verifier told URLLC's base
latency measured on this cell, so the second intent is accepted, which asks whether the cell can deliver all three goals.

Raw logs go to eval/results/m08d/ (git-ignored), the table to docs/learn/data/M08d-runs.csv. M08c's results, made
at 4bdfba4, stay in eval/results/m08c/ and docs/learn/data/M08c-runs.csv.
"""

import argparse
import csv
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "testbed"))  # for traffic.py
import traffic  # noqa: E402

from eval.interface import wait_for_service  # noqa: E402
from eval.metrics import load_run, measure_run, over_bound_after  # noqa: E402
from eval.realtime_check import start  # noqa: E402
from eval.scenarios import Scenario  # noqa: E402
from eval.testbed_scenarios import SCENARIOS, SEGMENTS, TESTBED_CAPACITY_MBPS  # noqa: E402
from rl.runner import SERVED_SMOOTHING_ALPHA, clear_run_state  # noqa: E402
from store import db, kpi, testbed  # noqa: E402
from store.events import log_event, read_events  # noqa: E402

RESULTS = Path("eval/results/m08d")
TABLE = Path("docs/learn/data/M08d-runs.csv")
TESTBED_DB = "var/ric-testbed.db"
COMPOSE = ["docker", "compose", "-f", "testbed/rfsim-flexric/docker-compose.yml", "--profile", "control"]
RL_IMAGE = "agentic-ric/rl-xapp:latest"
CONTROLLERS = ("static", "rl_only", "llm_only", "proposed")
USES_AGENT = ("llm_only", "proposed")
NO_POLICY = ("static", "llm_only")  # the runner's baseline: an equal split, or the LLM's when it sets one
REPEATS = 3
PORT = 8000
CODE_DIRS = ("agent", "backends", "eval", "gateway", "rl", "sim", "store", "testbed/traffic.py", "verifier", "xapps")
WATCHED = ("rfsim5g-oai-gnb", "nearRT-RIC", "xapp-kpm-slices", "rfsim5g-oai-nr-ue-embb", "rfsim5g-oai-nr-ue-urllc",
           "rfsim5g-oai-nr-ue-be", "oai-upf", "oai-amf")
READY_TIMEOUT_S = 60.0
DRAIN_S = 60.0  # after the scenario's end, each traffic process gets this long to close normally, then stops itself
STOP_GRACE_S = 30.0  # a traffic process still running this long after its deadline is stuck: the run is invalid
DEADLINE_STOP = "interrupt - the client has terminated"  # what iperf3 writes when its deadline (SIGTERM) stops it
MAX_SEGMENTS = 3  # each segment of a run has its own iperf3 servers
WHAT_IF = "what_if"
WHAT_IF_BASE_LATENCY_MS = 0.6  # URLLC's RLC delay under capacity, M08c; the verifier otherwise uses the simulator's 2 ms
QUIET_MBPS = 1.0  # KPM throughput under which a slice's queue counts as drained
QUIET_TIMEOUT_S = 90.0


@dataclasses.dataclass(frozen=True)
class TestbedRun:
    study: str  # testbed, what_if or dev
    scenario: str
    controller: str
    repeat: int

    @property
    def run_id(self) -> str:
        return f"{self.study}/{self.scenario}/{self.controller}/repeat{self.repeat}"


# ---------------------------------------------------------------- pure parts


def commit() -> str:
    run = lambda *args: subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()  # noqa: E731
    return run("rev-parse", "--short", "HEAD") + ("+uncommitted" if run("status", "--porcelain", "--", *CODE_DIRS) else "")


def guard_final(revision: str) -> None:
    """The rules come first and the history shows it (M08a did not; M08b did)."""
    if revision.endswith("+uncommitted"):
        raise SystemExit("commit the code and the protocol first: the matrix must come after them in the history")


def order(repeat: int) -> list[tuple[str, str]]:
    """All 12 (scenario, controller) pairs, rotated per repeat, so that slow drift during a repeat doesn't
    always fall on the same run."""
    pairs = [(s, c) for s in SCENARIOS for c in CONTROLLERS]
    shift = 5 * (repeat - 1) % len(pairs)  # 5 is coprime with 12: every pair changes position and neighbour
    return pairs[shift:] + pairs[:shift]


def runner_args(controller: str) -> str:
    return "--baseline equal" if controller in NO_POLICY else ""


def rebase_events(events: list[dict], t0: float) -> list[dict]:
    """The same events on the scenario's clock, which is what eval.metrics reads."""
    return [{**e, "ts_backend": e["ts_backend"] - t0} if e.get("ts_backend") is not None else e for e in events]


def problems(events: list[dict], restarts_before: dict[str, int], restarts_after: dict[str, int], iperf_errors: list[str]) -> list[str]:
    """Why a run is invalid; empty if it is valid. Only the machinery counts here, never a score."""
    found = []
    if any(e["event"] == "xapp_exited" for e in events):
        found.append("the slice-control xApp exited")
    if any(e["event"] == "kpm_stale" for e in events):
        found.append("KPM reports went stale for over 5 s")
    for name, n in restarts_after.items():
        if n != restarts_before.get(name, n):  # -1: not running (restart_counts), e.g. a UE that crashed and stayed down
            found.append(f"{name} stopped" if n == NOT_RUNNING else f"{name} restarted")
    return found + iperf_errors


def iperf_failure(path: Path) -> str | None:
    """A finished iperf3 run writes JSON with an end section; a failed one doesn't, or carries an error. Being stopped
    by its own deadline isn't a failure: it means the slice's queue was still draining, which is the controller's result."""
    try:
        result = json.loads(path.read_text())
    except (ValueError, OSError):
        return f"{path.name}: no iperf3 result"
    error = result.get("error")
    return f"{path.name}: {error}" if error and not error.startswith(DEADLINE_STOP) else None


def stopped_by_deadline(path: Path) -> bool:
    try:
        return str(json.loads(path.read_text()).get("error", "")).startswith(DEADLINE_STOP)
    except (ValueError, OSError):
        return False


def iperf_mbps(paths: list[Path]) -> float:
    """What iperf3 received, over all of a slice's segments: KPM's check, from outside the RAN."""
    bits = seconds = 0.0
    for path in paths:
        result = json.loads(path.read_text())
        if stopped_by_deadline(path):  # a stopped client's end section is empty: add up what its intervals received
            parts = [i["sum"] for i in result.get("intervals", [])]
            bits += sum(i["bits_per_second"] * i["seconds"] for i in parts)
            seconds += sum(i["seconds"] for i in parts)
            continue
        end = result.get("end", {}).get("sum", {})
        bits += end.get("bits_per_second", 0.0) * (1 - end.get("lost_percent", 0.0) / 100) * end.get("seconds", 0.0)
        seconds += end.get("seconds", 0.0)
    return bits / seconds / 1e6 if seconds else float("nan")


def ping_rtts(path: Path) -> list[float]:
    import re

    return [float(x) for x in re.findall(r"time=([\d.]+) ms", path.read_text())] if path.exists() else []


# ---------------------------------------------------------------- the stack


NOT_RUNNING = -1


def restart_counts() -> dict[str, int]:
    """Each watched container's restart count, or NOT_RUNNING if it is gone or stopped. A container that exits without a
    restart policy keeps its count (M09e, 2026-11-14: the BE UE segfaulted mid-run and the count alone missed it)."""
    counts = {}
    for name in WATCHED:
        out = subprocess.run(["docker", "inspect", "-f", "{{.RestartCount}} {{.State.Running}}", name],
                             capture_output=True, text=True, check=False)
        fields = out.stdout.split()
        running = out.returncode == 0 and len(fields) == 2 and fields[1] == "true"
        counts[name] = int(fields[0]) if running else NOT_RUNNING
    return counts


def segment_seconds(scenario: str, scale: float) -> list[int]:
    return [max(5, round(seconds * scale)) for seconds, _ in SEGMENTS[scenario]]


def play_segments(run: TestbedRun, work: Path, scale: float, conn_path: str, errors: list[str], t0: float) -> None:
    """One thread: the traffic, on the scenario's clock. M08c waited for each iperf3 to close, which takes as long as the
    queues take to drain, so a controller that built a queue delayed its own surge by up to 28 s (M08c's note).

    Every process gets the same deadline, DRAIN_S after the scenario's end, and stops itself there. In M08d a process
    waiting on a 95 s queue outlived its run, and this thread then cleared the next run's traffic_target; now nothing
    here outlives the deadline, and only play() clears traffic_target."""
    conn = db.connect(Path(conn_path))
    lengths = segment_seconds(run.scenario, scale)
    deadline = t0 + sum(lengths) + DRAIN_S
    procs, results = [], []
    at = 0.0
    for i, ((_, rates), seconds) in enumerate(zip(SEGMENTS[run.scenario], lengths, strict=True)):
        time.sleep(max(0.0, t0 + at - time.time()))
        testbed.write_traffic_target(conn, rates)
        log_event("scenario", "segment_start", data={"run": run.run_id, "segment": i, "rates_mbps": rates, "seconds": seconds, "wall": time.time()})
        left = deadline - time.time()
        procs += [traffic.iperf(s, f"{rate}M", seconds, work / f"seg{i}_{s}.json", segment=i, deadline_s=left) for s, rate in rates.items()]
        procs += [traffic.ping(s, seconds, work / f"seg{i}_ping_{s}.txt", deadline_s=left) for s in rates]
        results += [work / f"seg{i}_{s}.json" for s in rates]
        at += seconds
    conn.close()
    for proc in procs:
        proc.wait()
    errors += [e for path in results if (e := iperf_failure(path))]


def warm_up_llm() -> float:
    """One short request before an LLM run's clock starts, so the first parse doesn't include loading the model into
    VRAM (55.7 s in M08d's first LLM run). Same client and settings as the agent, so Ollama doesn't reload the model."""
    from agent.llm import OllamaClient

    return OllamaClient().chat([{"role": "user", "content": "Reply with OK."}]).latency_s


def wait_ready(conn, started: float, work: Path) -> None:
    """Before any traffic: the xApp answered and fresh KPIs are arriving. Otherwise stop here rather than record a broken run."""
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        ready = any(e["event"] == "xapp_ready" and e["ts_wall"] >= started for e in read_events())
        latest = kpi.latest_ts(conn) or 0.0
        if ready and time.time() - latest < 3.0:
            return
        time.sleep(0.5)
    raise RuntimeError(f"the RL xApp or KPM isn't ready after {READY_TIMEOUT_S:g} s: see {work}")


def wait_quiet(conn, timeout_s: float = QUIET_TIMEOUT_S) -> bool:
    """One run's backlog must not leak into the next: wait until every slice has stopped carrying traffic."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        reports = testbed.read_latest(conn)
        if reports and all(r.dl_thr_mbps < QUIET_MBPS for r in reports.values()):
            return True
        time.sleep(1.0)
    return False


def submit(text: str) -> None:
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/intents", json.dumps({"text": text}).encode(), {"content-type": "application/json"})
    urllib.request.urlopen(request, timeout=10).read()


def export(conn, work: Path, started: float, t0: float, seconds: float) -> list[dict]:
    """KPIs and events of this run, on the scenario's clock; rows from before t=0 or after the end are left out."""
    rows = conn.execute("SELECT ts, slice, dl_thr_mbps, offered_load_mbps, latency_ms, prb_share_alloc, policy_id FROM kpi "
                        "WHERE ts > ? AND ts <= ? ORDER BY ts, slice", (t0, t0 + seconds)).fetchall()
    with (work / "kpi.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "slice", "dl_thr_mbps", "offered_load_mbps", "latency_ms", "prb_share_alloc", "policy_id"])
        writer.writerows([r[0] - t0, *tuple(r)[1:]] for r in rows)
    events = rebase_events([e for e in read_events() if e["ts_wall"] >= started], t0)
    (work / "events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events))
    return events


def play(run: TestbedRun, scale: float = 1.0) -> Path:
    """One run, from a clean store. Writes run.json last: without it the run didn't finish and is played again."""
    work = RESULTS / run.run_id
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    os.environ["RIC_DB"] = TESTBED_DB
    env = {**os.environ, "RIC_MODE": "eval", "RIC_CONTROLLER": "proposed" if run.controller not in USES_AGENT else run.controller,
           "RUNNER_ARGS": runner_args(run.controller)}
    py = sys.executable
    conn = db.connect(Path(TESTBED_DB))
    clear_run_state(conn)
    conn.execute("DELETE FROM kpm_latest")
    conn.execute("DELETE FROM traffic_target")
    traffic.remove_leftovers(MAX_SEGMENTS)
    started = time.time()
    began = time.monotonic()
    before = restart_counts()
    processes = [start([py, "-m", "xapps.kpm_monitor", "--for-runner"], env, work / "kpm_monitor.out")]
    errors: list[str] = []
    scenario = SCENARIOS[run.scenario]
    seconds = scenario.seconds * scale
    try:
        time.sleep(3)
        subprocess.run([*COMPOSE, "up", "-d", "rl-xapp"], env=env, check=True, capture_output=True)
        wait_ready(conn, started, work)
        if run.study.startswith(WHAT_IF):  # after the runner's own write, before the agent starts
            info, served = kpi.read_backend_info(conn)
            kpi.write_backend_info(conn, dataclasses.replace(info, urllc_base_latency_ms=WHAT_IF_BASE_LATENCY_MS), served)
        if run.controller in USES_AGENT:
            log_event("scenario", "llm_warmup", data={"run": run.run_id, "latency_s": round(warm_up_llm(), 2)})
            processes.append(start([py, "-m", "uvicorn", "agent.service:app", "--port", str(PORT)], env, work / "service.out"))
            processes.append(start([py, "-m", "agent.worker"], env, work / "worker.out"))
            wait_for_service(PORT)
        traffic.start_servers(MAX_SEGMENTS)
        t0 = time.time()
        log_event("scenario", "scenario_start", data={"run": run.run_id, "t0": t0})
        player = threading.Thread(target=play_segments, args=(run, work, scale, TESTBED_DB, errors, t0), daemon=True)
        player.start()
        if run.controller in USES_AGENT:
            for intent in scenario.intents:  # on their own clock: a segment running late can't delay them
                time.sleep(max(0.0, t0 + intent.at_s * scale - time.time()))
                submit(intent.text)
        time.sleep(max(0.0, t0 + seconds - time.time()))
        player.join(timeout=max(0.0, t0 + sum(segment_seconds(run.scenario, scale)) + DRAIN_S + STOP_GRACE_S - time.time()))
        if player.is_alive():
            errors.append(f"a traffic process was still running {STOP_GRACE_S:g} s after its deadline")
        conn.execute("DELETE FROM traffic_target")  # here, not in the thread: a late thread can't reach the next run
        closed_after_s = round(time.time() - (t0 + seconds), 1)  # information only: a long queue is a result, not a failure
        stopped = sorted(p.name for p in work.glob("seg*_*.json") if stopped_by_deadline(p))
        drained = wait_quiet(conn)
        log_event("scenario", "scenario_stop", data={"run": run.run_id, "drained": drained, "closed_after_s": closed_after_s,
                                                     "stopped_at_deadline": stopped})
    finally:
        subprocess.run([*COMPOSE, "stop", "rl-xapp"], env=env, check=False, capture_output=True)
        logs = subprocess.run(["docker", "logs", "rl-xapp"], capture_output=True, text=True, check=False)
        (work / "rl_xapp.out").write_text(logs.stdout[-20000:] + logs.stderr[-20000:])
        subprocess.run([*COMPOSE, "rm", "-f", "rl-xapp"], env=env, check=False, capture_output=True)
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(timeout=20)
    events = export(conn, work, started, t0, seconds)
    conn.close()
    invalid = problems(events, before, restart_counts(), errors)
    (work / "run.json").write_text(json.dumps(
        {**dataclasses.asdict(run), "seed": run.repeat, "alpha": SERVED_SMOOTHING_ALPHA, "delay_ticks": 0, "importance": None, "commit": commit(),
         "scale": scale, "t0": t0, "wall_s": round(time.monotonic() - began, 1), "valid": not invalid, "problems": invalid}, indent=1))
    return work


# ---------------------------------------------------------------- the table


def scenario_as_played(name: str, events: list[dict]) -> Scenario:
    """S1's reaction clock starts when the spike really started, from the log; everything else is as stated."""
    t0 = next((e["data"]["t0"] for e in events if e["event"] == "scenario_start"), None)
    spike = next((e["data"]["wall"] for e in events if e["event"] == "segment_start" and e["data"]["segment"] == 1), None)
    scenario = SCENARIOS[name]
    if name != "s1" or t0 is None or spike is None:
        return scenario
    return dataclasses.replace(scenario, intents=(dataclasses.replace(scenario.intents[0], reaction_from_s=spike - t0),))


def write_table(results: Path = RESULTS, table: Path = TABLE) -> int:
    rows = []
    paths = [p for study in ("testbed", WHAT_IF) for p in sorted(results.glob(f"{study}/*/*/repeat*/run.json"))]
    for path in paths:
        spec = json.loads(path.read_text())
        work = path.parent
        played = scenario_as_played(spec["scenario"], read_events(work / "events.jsonl"))
        measured = measure_run(work, played)
        kpis = load_run(work)[0]
        after = [over_bound_after(kpis, intent) for intent in played.intents]
        outside = {}
        for s in ("embb", "urllc", "be"):
            outside[f"iperf_{s}_mbps"] = iperf_mbps(sorted(work.glob(f"seg*_{s}.json")))
            rtts = [x for p in sorted(work.glob(f"seg*_ping_{s}.txt")) for x in ping_rtts(p)]
            outside[f"ping_{s}_median_ms"] = float(np.median(rtts)) if rtts else ""
            outside[f"ping_{s}_p95_ms"] = float(np.percentile(rtts, 95)) if rtts else ""
        base = {"study": spec["study"], "scenario": spec["scenario"], "controller": spec["controller"], "seed": spec["repeat"], "valid": spec["valid"],
                "problems": "; ".join(spec["problems"]), "invalid_copy": work.name.endswith("_invalid"), "alpha": spec["alpha"], "delay_ticks": 0, "importance": ""}
        rows += [{**base, **intent, "over_bound_after_s": "" if seconds is None else seconds, **measured["run"], **outside, "wall_s": spec["wall_s"]}
                 for intent, seconds in zip(measured["intents"], after, strict=True)]
    if rows:
        table.parent.mkdir(parents=True, exist_ok=True)
        with table.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------- the matrix


def check_image() -> None:
    """The container must run this code: the right capacity, the baseline and the served filter."""
    code = ("from xapps.kpm_monitor import cell_info; from rl.baselines import EqualSplitPolicy; from rl.runner import SERVED_SMOOTHING_ALPHA as a; "
            "print(cell_info().capacity_mbps, a)")
    out = subprocess.run(["docker", "run", "--rm", "--entrypoint", "/opt/venv/bin/python", "-w", "/app", RL_IMAGE, "-c", code], capture_output=True, text=True, check=False)
    expected = f"{TESTBED_CAPACITY_MBPS} {SERVED_SMOOTHING_ALPHA}"
    if expected not in out.stdout:
        raise SystemExit(f"{RL_IMAGE} is older than this code (expected '{expected}', got '{out.stdout.strip()}' {out.stderr[-200:]}): run xapps/rl/build.sh")


def restart_testbed() -> None:
    subprocess.run(["testbed/down.sh"], check=False, capture_output=True)
    subprocess.run(["testbed/up.sh"], check=True)


def play_matrix(scale: float = 1.0) -> None:
    for repeat in range(1, REPEATS + 1):
        missing = [(s, c) for s, c in order(repeat) if not (RESULTS / TestbedRun("testbed", s, c, repeat).run_id / "run.json").exists()]
        if not missing:
            continue
        restart_testbed()
        for i, (scenario, controller) in enumerate(missing, 1):
            run = TestbedRun("testbed", scenario, controller, repeat)
            work = play(run, scale)
            spec = json.loads((work / "run.json").read_text())
            print(f"[repeat {repeat}, {i}/{len(missing)}] {run.run_id}: {spec['wall_s']:.0f} s, valid {spec['valid']} {spec['problems']}", flush=True)
    replay_invalid(scale)
    play_what_if(scale)


def play_what_if(scale: float) -> None:
    missing = [r for r in range(1, REPEATS + 1) if not (RESULTS / TestbedRun(WHAT_IF, "s3", "proposed", r).run_id / "run.json").exists()]
    if missing:
        restart_testbed()
    for repeat in missing:
        work = play(TestbedRun(WHAT_IF, "s3", "proposed", repeat), scale)
        spec = json.loads((work / "run.json").read_text())
        print(f"[what-if {repeat}] {spec['wall_s']:.0f} s, valid {spec['valid']} {spec['problems']}", flush=True)


def replay_invalid(scale: float) -> None:
    """Once only: the invalid run stays on disk under another name, and its replay counts whatever it shows."""
    invalid = [p.parent for p in sorted(RESULTS.glob("testbed/*/*/repeat?/run.json")) if not json.loads(p.read_text())["valid"]]
    for work in invalid:
        if work.with_name(work.name + "_invalid").exists():
            continue
        spec = json.loads((work / "run.json").read_text())
        work.rename(work.with_name(work.name + "_invalid"))
        restart_testbed()
        play(TestbedRun("testbed", spec["scenario"], spec["controller"], spec["repeat"]), scale)
        print(f"replayed {work}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final", action="store_true", help="the whole matrix; refuses uncommitted code")
    parser.add_argument("--dev", action="store_true", help="never in the table; the testbed must be up")
    parser.add_argument("--what-if", action="store_true", help="--dev: tell the verifier the measured URLLC base latency")
    parser.add_argument("--scale", type=float, default=0.2, help="--dev: multiply the scenario's times")
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=["s3"])
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS))
    args = parser.parse_args()
    if args.final == args.dev:
        raise SystemExit("choose --dev or --final")
    if args.final:
        guard_final(commit())
    check_image()
    if args.dev:
        for scenario in args.scenarios:
            for controller in args.controllers:
                work = play(TestbedRun("what_if_dev" if args.what_if else "dev", scenario, controller, 1), args.scale)
                print(work, json.loads((work / "run.json").read_text())["problems"], flush=True)
        return
    play_matrix()
    print(f"{write_table()} rows in {TABLE}", flush=True)
    subprocess.run(["testbed/down.sh"], check=False, capture_output=True)


if __name__ == "__main__":
    main()
