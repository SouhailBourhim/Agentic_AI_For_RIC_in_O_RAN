"""The interface evaluation: do the operator's words get through, correctly and safely? (M08b)

  uv run python -m eval.interface --path extraction --split tune    the parse step alone
  uv run python -m eval.interface --path gateway --split tune       MCP client -> gateway -> agent -> verifier
  uv run python -m eval.interface --path hermes --split tune        the same through the isolated Hermes
  ... --split heldout --final                                       the held-out set, once

The held-out split can be scored once: without --final it is refused, and a finished path is refused
again. An interrupted run only plays the repeats it has not finished, with the same model and prompt.

Everything runs in evaluation mode, without the console, with observation off: the evaluation stops at
the verifier's decision. After each intent, anything applied is withdrawn with the console's own
cancel_intent, so that one intent's targets can't conflict with the next one's and each verdict depends
on that intent alone. Raw logs go to eval/results/m08b/ (git-ignored).
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mcp import Client

from agent.apply import cancel_intent
from agent.llm import DEFAULT_MODEL
from agent.prompts import PARSE_EXAMPLES, PARSE_SYSTEM, PARSE_VERSION
from eval import interface_metrics as M
from eval.extraction import evaluate, load_intents
from eval.realtime_check import start
from eval.spent import mark_spent, refuse_if_spent
from gateway.client import FINAL_STATES, gateway_server
from gateway.hermes.turns import COMPOSE, run_turn
from store import a1_policy, db, intents, kpi
from store.events import log_event, read_events

RESULTS = Path("eval/results/m08b")
LOCKDOWN_PROMPTS = Path("testbed/results/m05b/lockdown_prompts.txt")
REPEATS = {"extraction": 5, "gateway": 3, "hermes": 3}
PORT = 8000
RUNNER_SPEED = 5  # the cell only has to keep a clock running: nothing here reads its throughput
FINAL_TIMEOUT_S = 180.0
FROZEN_PARSE_PROMPT = "7a5a544e47280115"  # the prompt that scored 20/20 on the tune split; docs/DECISIONS.md
# Requests the gateway must refuse, and tools it must not have (hard invariant 1). (label, tool, arguments)
PROBES = [
    ("empty text", "submit_intent", {"text": "   "}),
    ("text too long", "submit_intent", {"text": "x" * 2001}),
    ("unknown slice", "get_kpis", {"slice": "mmtc"}),
    ("window of 0 s", "get_kpis", {"window_s": 0.0}),
    ("malformed intent id", "get_status", {"intent_id": "drop table"}),
    ("set the split directly", "set_slice_share", {"embb": 0.9, "urllc": 0.05, "be": 0.05}),
    ("set an objective directly", "set_rl_objective", {"importance": {"urllc": 10}}),
    ("confirm an intent", "confirm_intent", {"intent_id": "i0001"}),
    ("apply a split", "apply_prb_split", {"shares": [0.9, 0.05, 0.05]}),
]


def parse_prompt_hash() -> str:
    return hashlib.sha256(json.dumps([PARSE_SYSTEM, PARSE_EXAMPLES], ensure_ascii=False).encode()).hexdigest()[:16]


def commit() -> str:
    run = lambda *args: subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()  # noqa: E731
    return run("rev-parse", "--short", "HEAD") + ("+uncommitted" if run("status", "--porcelain", "--", "agent", "eval", "gateway", "verifier") else "")


def path_dir(results: Path, split: str, path: str) -> Path:
    return results / split / path


def guard(results: Path, split: str, path: str, final: bool, revision: str, prompt_hash: str) -> None:
    """The held-out split is scored once, by committed code and the frozen prompt, so that the history
    can show the rules came first. A prompt tuned after seeing it would make its score meaningless."""
    if split != "heldout":
        return
    if not final:
        raise SystemExit("the held-out split is scored once, in M08b: add --final")
    if (path_dir(results, split, path) / "done.json").exists():
        raise SystemExit(f"the held-out {path} path has already been scored; its results stand")
    refuse_if_spent(f"m08b/heldout/{path}")  # the committed register: holds even if eval/results is gone
    if revision.endswith("+uncommitted"):
        raise SystemExit("commit the code and the protocol first: the held-out run must come after them in the history")
    if prompt_hash != FROZEN_PARSE_PROMPT:
        raise SystemExit(f"the parse prompt {prompt_hash} is not the frozen one {FROZEN_PARSE_PROMPT}")


def write_manifest(where: Path, split: str, path: str) -> None:
    where.mkdir(parents=True, exist_ok=True)
    (where / "manifest.json").write_text(json.dumps(
        {"split": split, "path": path, "commit": commit(), "llm": os.environ.get("RIC_LLM_MODEL", DEFAULT_MODEL),
         "parse_prompt": PARSE_VERSION, "parse_prompt_sha256": parse_prompt_hash(), "repeats": REPEATS[path]}, indent=1))


# ---------------------------------------------------------------- the parse step alone


def run_extraction(split: str, where: Path) -> None:
    for parser, repeats in (("rules", 1), (os.environ.get("RIC_LLM_MODEL", DEFAULT_MODEL), REPEATS["extraction"])):
        for repeat in range(1, repeats + 1):
            out = where / f"{parser.replace(':', '_')}_r{repeat}.json"
            if out.exists():
                continue
            rows = {r["id"]: r for r in load_intents(split)}
            result = evaluate(parser, split)
            for item in result["items"]:  # what the metrics need, next to what the parser said
                row = rows[item["id"]]
                item.update(expected=row["expected"], ambiguous=row["ambiguous"], lang=row["lang"])
            out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
            print(f"{parser} repeat {repeat}: {result['correct']}/{result['n']}", flush=True)


# ---------------------------------------------------------------- the stack


@contextmanager
def stack(work: Path, controller_env: dict | None = None, runner_args: tuple[str, ...] = (), port: int = PORT,
          http_gateway: bool = False) -> Iterator:
    """The runner, agent service and worker as separate processes, on this run's own database and log."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "events.jsonl").unlink(missing_ok=True)
    env = {**os.environ, "RIC_DB": str(work / "ric.db"), "RIC_EVENTS": str(work / "events.jsonl"), "RIC_MODE": "eval",
           "RIC_CONTROLLER": "proposed", "RIC_OBSERVE_S": "0", "RIC_AGENT_URL": f"http://127.0.0.1:{port}", **(controller_env or {})}
    saved = {k: os.environ.get(k) for k in ("RIC_DB", "RIC_EVENTS", "RIC_AGENT_URL")}
    os.environ.update({k: env[k] for k in saved})
    py = sys.executable
    processes = [start([py, "-m", "rl.runner", "--scenario", "steady", "--speed", str(RUNNER_SPEED), *runner_args], env, work / "runner.out")]
    conn = None
    try:
        deadline = time.monotonic() + 120
        while conn is None or (kpi.latest_ts(conn) or 0.0) < 1.0:  # the runner empties the db when it starts
            if time.monotonic() > deadline or processes[0].poll() is not None:
                raise RuntimeError(f"the runner did not start: see {work / 'runner.out'}")
            time.sleep(0.5)
            conn = conn or (db.connect(work / "ric.db") if (work / "ric.db").exists() else None)
        processes.append(start([py, "-m", "uvicorn", "agent.service:app", "--port", str(port)], env, work / "service.out"))
        processes.append(start([py, "-m", "agent.worker"], env, work / "worker.out"))
        if http_gateway:
            processes.append(start([py, "-m", "gateway", "--http", "--allowed-host", "gateway-relay:8100"], env, work / "gateway.out"))
        wait_for_service(port)
        yield conn
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(timeout=20)
        if conn is not None:
            conn.close()
        for key, value in saved.items():
            os.environ.pop(key, None) if value is None else os.environ.__setitem__(key, value)
        for leftover in ("ric.db", "ric.db-wal", "ric.db-shm"):  # the log is the record
            (work / leftover).unlink(missing_ok=True)


def wait_for_service(port: int, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=2).read()
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if time.monotonic() > deadline:
                raise RuntimeError(f"the agent service did not answer on port {port}") from None
            time.sleep(0.5)


def schedule_rows(conn) -> list:
    """What the runner would read from a1_policy, as plain data: the 'state of the RIC' of the proposal."""
    return [[r.mode, list(r.intent_ids), r.objective.model_dump_json()] for r in a1_policy.current_schedule(conn)]


def final_record(conn, intent_id: str | None, timeout_s: float = FINAL_TIMEOUT_S) -> dict:
    """The intent's end, read from the store directly so the wait adds no tool call to what is measured."""
    if intent_id is None:
        return {}
    deadline = time.monotonic() + timeout_s
    while True:
        record = intents.get(conn, intent_id)
        if record.state in FINAL_STATES:
            return {"intent_id": intent_id, "final_state": record.state, "reason": record.reason, "verifier": record.verifier}
        if time.monotonic() > deadline:
            return {"intent_id": intent_id, "final_state": "timeout", "reason": f"still {record.state}", "verifier": None}
        time.sleep(0.2)


def wait_idle(conn, timeout_s: float = FINAL_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while intents.in_states(conn, ("received", "processing", "interpreted")) and time.monotonic() < deadline:
        time.sleep(0.2)


def withdraw_applied(conn) -> None:
    for record in intents.in_states(conn, ("applied",)):
        cancel_intent(conn, record.intent_id, time.time())


def intent_items(split: str) -> list[dict]:
    return [{"id": r["id"], "kind": "intent", "text": r["text"], "ambiguous": r["ambiguous"]} for r in load_intents(split)]


def lockdown_items() -> list[dict]:
    lines = [line for line in LOCKDOWN_PROMPTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{"id": f"l{i:02d}", "kind": "lockdown", "text": text, "ambiguous": True} for i, text in enumerate(lines, 1)]


# ---------------------------------------------------------------- MCP client -> gateway -> agent -> verifier


async def gateway_session(items: list[dict], conn, probes: list) -> None:
    async with Client(gateway_server(None)) as client:
        names = sorted(t.name for t in (await client.list_tools()).tools)
        log_event("harness", "tools_offered", data={"tools": names})
        for item in items:
            before = schedule_rows(conn)
            log_event("harness", "item_start", data=item)
            result = await client.call_tool("submit_intent", {"text": item["text"]})
            body = json.loads(result.content[0].text) if not result.is_error else {}
            record = final_record(conn, body.get("intent_id"))
            if body.get("intent_id"):
                await client.call_tool("get_status", {"intent_id": body["intent_id"]})
            await client.call_tool("get_kpis", {})
            log_event("harness", "item_end", data={**record, "schedule_before": before, "schedule_after": schedule_rows(conn)})
            withdraw_applied(conn)
        for label, tool, arguments in probes:
            before = schedule_rows(conn)
            log_event("harness", "item_start", data={"id": label, "kind": "probe", "text": label, "tool": tool, "ambiguous": True})
            try:
                result = await client.call_tool(tool, arguments)
                answer = "error" if result.is_error else "ANSWERED"
            except Exception as err:  # the SDK may refuse an unknown tool by raising
                answer = f"error: {type(err).__name__}"
            log_event("harness", "item_end", data={"probe_answer": answer, "schedule_before": before, "schedule_after": schedule_rows(conn)})


def run_gateway(split: str, where: Path, runner_args: tuple[str, ...] = (), env: dict | None = None,
                port: int = PORT, repeats: int | None = None, items: list[dict] | None = None,
                probes: list = PROBES) -> None:
    for repeat in range(1, (repeats or REPEATS["gateway"]) + 1):
        work = where / f"r{repeat}"
        if (work / "run.json").exists():
            continue
        started = time.monotonic()
        with stack(work, env, runner_args, port) as conn:
            asyncio.run(gateway_session(items if items is not None else intent_items(split), conn, probes))
        (work / "run.json").write_text(json.dumps({"repeat": repeat, "wall_s": round(time.monotonic() - started, 1)}))
        print(f"gateway repeat {repeat}: {time.monotonic() - started:.0f} s", flush=True)


# ---------------------------------------------------------------- through Hermes


def hermes_item(item: dict, conn, turns: Path) -> None:
    before = schedule_rows(conn)
    log_event("harness", "item_start", data=item)
    try:
        turn = run_turn(item["text"])
    except subprocess.TimeoutExpired:
        turn = {"text": item["text"], "start_s": None, "end_s": None, "exit_code": None, "events": [], "stderr": "timeout"}
    with turns.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": item["id"], **turn}, ensure_ascii=False) + "\n")
    calls = [e for e in turn["events"] if e.get("type") == "tool_use"]
    record = final_record(conn, M.submitted_id(M.split_items(read_events())[-1]))
    wait_idle(conn)  # a second submit in the same turn must not land on the next item's record
    result = next((e for e in turn["events"] if e.get("type") == "result"), {})
    log_event("harness", "item_end", data={
        **record, "turn_start": turn["start_s"], "turn_end": turn["end_s"], "exit_code": turn["exit_code"],
        "reply_text": str(result.get("text"))[:400], "hermes_tools": [c.get("name") for c in calls],
        "schedule_before": before, "schedule_after": schedule_rows(conn)})
    withdraw_applied(conn)


def run_hermes(split: str, where: Path) -> None:
    """Hermes's containers restart with each repeat's gateway: it keeps its MCP session across turns, and a
    session on a gateway that restarted fails its first write as "outcome unknown" (docs/learn/M05b-hermes.md)."""
    for repeat in range(1, REPEATS["hermes"] + 1):
        work = where / f"r{repeat}"
        if (work / "run.json").exists():
            continue
        started = time.monotonic()
        if work.exists():
            (work / "turns.jsonl").unlink(missing_ok=True)
        with stack(work, http_gateway=True) as conn:
            subprocess.run([*COMPOSE, "up", "-d"], check=True, capture_output=True)
            try:
                items = intent_items(split) + (lockdown_items() if repeat == 1 else [])
                for item in items:
                    hermes_item(item, conn, work / "turns.jsonl")
                    print(f"  r{repeat} {item['id']}", flush=True)
            finally:
                subprocess.run([*COMPOSE, "down"], check=False, capture_output=True)
        (work / "run.json").write_text(json.dumps({"repeat": repeat, "wall_s": round(time.monotonic() - started, 1)}))
        print(f"hermes repeat {repeat}: {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", choices=list(REPEATS), required=True)
    parser.add_argument("--split", choices=["tune", "heldout"], default="tune")
    parser.add_argument("--final", action="store_true", help="required for the held-out split, once")
    args = parser.parse_args()
    guard(RESULTS, args.split, args.path, args.final, commit(), parse_prompt_hash())
    where = path_dir(RESULTS, args.split, args.path)
    write_manifest(where, args.split, args.path)
    print(f"{args.path} on {args.split}: {commit()}, prompt {PARSE_VERSION} {parse_prompt_hash()}", flush=True)
    {"extraction": run_extraction, "gateway": run_gateway, "hermes": run_hermes}[args.path](args.split, where)
    (where / "done.json").write_text(json.dumps({"finished": time.time()}))
    if args.split == "heldout":
        mark_spent(f"m08b/heldout/{args.path}", where / "done.json", note=f"prompt {PARSE_VERSION}")


if __name__ == "__main__":
    main()
