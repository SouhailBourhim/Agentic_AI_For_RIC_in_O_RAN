"""The whole proposed path in one command, on the simulator, with no GPU and no LLM (M12).

  uv run python -m demo                  about 30 s; prints the URLLC slice before and after one intent
  uv run python -m demo --keep --speed 2  leaves everything running (and the Prometheus exporter) for Grafana
  uv run python -m demo --executor formula --speed 30     what tests/test_demo.py runs

What runs, each as its own process, on a temporary database and event log (var/ric.db is not touched):
1. the RL runner on the `demo` scenario: URLLC surges from 10 % to 40 % of the cell at t=120 s;
2. the agent service (HTTP) and the agent worker, with the rule-based parser instead of the LLM;
3. the MCP gateway, started by the MCP client below over stdio, as Hermes would reach it.
The demo then plays the operator, through the three MCP tools only: it reads the KPIs during the surge
(get_kpis), submits the README's intent (submit_intent), follows it to `applied` (get_status), and reads the
KPIs again.

Two differences from the live demo, both printed:
- RIC_MODE=eval: the intent is confirmed automatically. In demo mode it would wait for `python -m console confirm`.
- The executor is the served RL champion if this machine has one (var/policy_cache, from training or MLflow);
  a clean clone has none, and then the formula executor (rl.baselines.FormulaPolicy, M09b) stands in.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from mcp import Client
from mcp.client.stdio import stdio_client

from gateway.client import FINAL_STATES, gateway_server

INTENT = "garantir 20 Mb/s au slice URLLC avec une latence de 10 ms pendant 5 minutes"
BEFORE_AT_S = 150.0  # backend clock: 30 s into the URLLC surge
AFTER_S = 40.0  # backend seconds after the intent is applied: the served filter needs about 20 to settle
WINDOW_S = 20.0  # KPI window read before and after
SERVED = Path("var/policy_cache/served.json")
PY = sys.executable


def executor_args(choice: str) -> tuple[list[str], str]:
    """The runner's arguments for the executor, and a line saying which one runs."""
    if choice == "rl" or (choice == "auto" and SERVED.exists()):
        served = json.loads(SERVED.read_text())
        return ["--policy-file", str(SERVED.parent / served["file"]), "--model-label", served["label"]], \
            f"executor: RL champion {served['label']}"
    return ["--baseline", "formula"], ("executor: no trained policy on this machine, the formula executor (M09b) "
                                       "stands in; see docs/CLAIMS.md for how the two compare")


def start(command: list[str], env: dict, log: Path) -> subprocess.Popen:
    return subprocess.Popen(command, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT)


def wait_for(url: str, process: subprocess.Popen, log: Path, timeout_s: float = 60.0) -> None:
    """Until the URL answers; a process that died meanwhile is reported with its log."""
    deadline = time.monotonic() + timeout_s
    while True:
        if process.poll() is not None:
            raise SystemExit(f"a process stopped while starting; see {log}:\n{log.read_text()[-2000:]}")
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if time.monotonic() > deadline:
                raise SystemExit(f"{url} did not answer in {timeout_s:g} s; see {log}") from None
            time.sleep(0.3)


async def call(client: Client, tool: str, arguments: dict) -> dict:
    """One MCP tool call; its JSON body. A tool error stops the demo with the gateway's message."""
    result = await client.call_tool(tool, arguments)
    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
    if result.is_error:
        raise SystemExit(f"{tool} failed: {text}")
    return json.loads(text)


async def urllc_window(client: Client) -> tuple[float, dict]:
    """The backend clock and URLLC's mean throughput, mean and p95 latency, and mean share over WINDOW_S."""
    body = await call(client, "get_kpis", {"slice": "urllc", "window_s": WINDOW_S})
    stats = body["slices"].get("urllc", {})
    return body["ts"] or 0.0, {
        "throughput (Mb/s)": stats["dl_thr_mbps"]["mean"],
        "latency mean (ms)": stats["latency_ms"]["mean"],
        "latency p95 (ms)": stats["latency_ms"]["p95"],
        "PRB share": stats["prb_share_alloc"]["mean"],
    }


async def wait_clock(client: Client, at_s: float, speed: float) -> None:
    """Until the cell's clock reaches at_s; the time allowed follows the speed (plus a minute for start-up)."""
    now = (await call(client, "get_kpis", {"window_s": 1.0}))["ts"] or 0.0
    timeout_s = max(at_s - now, 0.0) / speed + 60.0
    deadline = time.monotonic() + timeout_s
    while now < at_s:
        if time.monotonic() > deadline:
            raise SystemExit(f"the cell clock did not reach {at_s:g} s in {timeout_s:.0f} s")
        await asyncio.sleep(0.2)
        now = (await call(client, "get_kpis", {"window_s": 1.0}))["ts"] or 0.0


async def operator(gateway_log: Path, speed: float) -> dict:
    """The operator's side, through the three MCP tools only. Returns what the demo shows (tests read it).

    The gateway runs over stdio, started by this client as Hermes' would be; its own log goes to gateway_log."""
    with gateway_log.open("w") as errlog:
        async with Client(stdio_client(gateway_server(None), errlog=errlog)) as client:
            return await play(client, speed)


async def follow(client: Client, intent_id: str, timeout_s: float = 60.0) -> dict:
    """get_status until a final state, printing each new state; the last status."""
    state, status = None, {}
    deadline = time.monotonic() + timeout_s
    while state not in FINAL_STATES:
        status = await call(client, "get_status", {"intent_id": intent_id})
        if status["state"] != state:
            state = status["state"]
            reason = status["verifier"]["reason"] if state == "applied" and status.get("verifier") else None
            print(f"  {intent_id}: {state}" + (f"  (verifier: {reason})" if reason else ""))
        if time.monotonic() > deadline:
            raise SystemExit(f"{intent_id} still {state} after {timeout_s:g} s")
        await asyncio.sleep(0.2)
    return status


async def play(client: Client, speed: float) -> dict:
    """KPIs during the surge, the intent, its states, KPIs once the executor has settled."""
    await wait_clock(client, BEFORE_AT_S, speed)
    clock, before = await urllc_window(client)
    print(f"\nt={clock:g} s, URLLC surge under the default objective; the operator types:\n  \"{INTENT}\"")
    intent_id = (await call(client, "submit_intent", {"text": INTENT}))["intent_id"]
    status = await follow(client, intent_id)
    if status["state"] != "applied":
        raise SystemExit(f"{intent_id} ended {status['state']}: {status.get('reason')}")
    await wait_clock(client, status["backend_now"] + AFTER_S, speed)
    clock_after, after = await urllc_window(client)
    print(f"\nURLLC, mean over {WINDOW_S:g} s     before (t={clock:g} s)   after (t={clock_after:g} s)")
    for name in before:
        print(f"  {name:20s} {before[name]:12.2f}   {after[name]:12.2f}")
    print("  target: >= 20 Mb/s or the whole load if less, latency < 10 ms")
    return {"intent_id": intent_id, "state": status["state"], "verifier": status.get("verifier"), "before": before,
            "after": after}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m demo", description=__doc__.splitlines()[0])
    parser.add_argument("--executor", choices=["auto", "rl", "formula"], default="auto",
                        help="auto: the RL champion if this machine has one, else the formula")
    parser.add_argument("--speed", type=float, default=10.0, help="cell seconds per wall second")
    parser.add_argument("--port", type=int, default=8765, help="the agent service's port (127.0.0.1)")
    parser.add_argument("--keep", action="store_true", help="leave everything running, plus the exporter on :9108")
    parser.add_argument("--json", type=Path, default=None, help="also write what is shown to this file (tests)")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="ric-demo-"))
    runner_args, executor_line = executor_args(args.executor)
    env = {**os.environ, "RIC_DB": str(work / "ric.db"), "RIC_EVENTS": str(work / "events.jsonl"),
           "RIC_MODE": "eval", "RIC_AGENT": "rules", "RIC_OBSERVE_S": "0",
           "RIC_AGENT_URL": f"http://127.0.0.1:{args.port}"}
    os.environ.update({k: env[k] for k in ("RIC_DB", "RIC_EVENTS", "RIC_AGENT_URL")})  # the stdio gateway inherits RIC_*
    print(f"{executor_line}\nconfirmation: automatic (RIC_MODE=eval; the live demo waits for `python -m console confirm`)"
          f"\nagent: rule-based parser, no LLM\nwork dir: {work}")

    started = time.monotonic()
    processes: list[subprocess.Popen] = []
    try:
        processes.append(start([PY, "-m", "rl.runner", "--scenario", "demo", "--speed", str(args.speed), *runner_args],
                               env, work / "runner.out"))
        service_log = work / "service.out"
        processes.append(start([PY, "-m", "uvicorn", "agent.service:app", "--host", "127.0.0.1", "--port",
                                str(args.port)], env, service_log))
        processes.append(start([PY, "-m", "agent.worker"], env, work / "worker.out"))
        if args.keep:
            processes.append(start([PY, "-m", "monitoring.exporter"], env, work / "exporter.out"))
        wait_for(f"http://127.0.0.1:{args.port}/status", processes[1], service_log)
        shown = asyncio.run(operator(work / "gateway.out", args.speed))
        if args.json is not None:
            args.json.write_text(json.dumps(shown, indent=1))
        print(f"\ndone in {time.monotonic() - started:.0f} s; logs and events in {work}")
        if args.keep:
            pids = " ".join(str(p.pid) for p in processes)
            print(f"still running (exporter on :9108 for Grafana); stop with: kill {pids}")
            return
    finally:
        if not args.keep:
            for process in processes:
                process.terminate()
            for process in processes:
                process.wait(timeout=20)


if __name__ == "__main__":
    main()
