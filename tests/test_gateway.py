"""M05a: the MCP gateway. Hard invariants 1 and 2, the gateway log, and the demo scenario of the
proposal (§5.4) replayed through the gateway with the rule-based agent.

1: the gateway exposes exactly submit_intent, get_kpis and get_status (no other tool, resource or
   prompt), and submit_intent only creates a request.
2 (imports and addresses): the gateway imports nothing that reaches the RAN or the SQLite store,
   and it only ever calls the agent service's four endpoints at the one configured address.
   The network part (no route to FlexRIC/OAI) is checked on the testbed in M07b.
"""

import ast
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import Client

from agent.apply import apply_intent
from agent.service import app
from agent.tools import AgentContext, Settings
from agent.worker import process_one
from conftest import start_cell
from gateway import server
from store import a1_policy
from store.events import read_events

REPO = Path(__file__).resolve().parent.parent
GATEWAY_DIR = REPO / "gateway"
GATEWAY_FILES = ["__init__.py", "__main__.py", "server.py", "client.py", "plot_demo.py"]
THE_THREE_TOOLS = {"submit_intent", "get_kpis", "get_status"}
# Top-level packages no gateway file may import: the RAN side, the store's database, the agent's
# internals (it is reached over HTTP only), and LLM clients.
FORBIDDEN = {"backends", "xapps", "rl", "sim", "verifier", "agent", "console", "monitoring", "sqlite3",
             "store", "ollama", "langgraph", "langchain", "openai", "anthropic"}
ALLOWED_FROM_FORBIDDEN = {"store.events"}  # JSONL append only: json, os, time, pathlib
# The endpoints of agent/service.py, the only requests the gateway may make.
ENDPOINTS = {("POST", "/intents"), ("GET", "/intents/{id}"), ("GET", "/kpis"), ("GET", "/status")}


def call(tool: str, arguments: dict | None = None):
    """Call one gateway tool through a real MCP session (in-process transport)."""
    async def go():
        async with Client(server.mcp) as client:
            return await client.call_tool(tool, arguments or {})
    return asyncio.run(go())


def body(result) -> dict:
    """The JSON a tool returned (the test fails on a tool error, with its message)."""
    assert not result.is_error, result.content[0].text
    return json.loads(result.content[0].text)


def gateway_events() -> list[dict]:
    return [e for e in read_events() if e["source"] == "gateway"]


@pytest.fixture
def events_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("RIC_EVENTS", str(path))
    return path


@pytest.fixture
def fake_agent(monkeypatch: pytest.MonkeyPatch, events_file: Path) -> list[httpx.Request]:
    """Replace the gateway's HTTP client by one that records every request and answers 200."""
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201 if request.method == "POST" else 200,
                              json={"intent_id": "i0001", "state": "received"})

    client = httpx.Client(base_url=server.agent_url(), transport=httpx.MockTransport(answer))
    monkeypatch.setattr(server, "http", client)
    return seen


# ------------------------------------------------------------------ invariant 1: exactly three tools


def test_exactly_three_tools_and_nothing_else() -> None:
    async def listing():
        async with Client(server.mcp) as client:
            return await client.list_tools(), await client.list_resources(), await client.list_prompts()
    tools, resources, prompts = asyncio.run(listing())
    assert {tool.name for tool in tools.tools} == THE_THREE_TOOLS
    assert resources.resources == [] and prompts.prompts == []  # no second way in


def test_submit_intent_only_creates_a_request(fake_agent) -> None:
    result = call("submit_intent", {"text": "Garantis 10 Mb/s au slice URLLC pendant 10 minutes."})
    assert not result.is_error
    assert [(r.method, r.url.path) for r in fake_agent] == [("POST", "/intents")]
    assert fake_agent[0].content == b'{"text":"Garantis 10 Mb/s au slice URLLC pendant 10 minutes."}'


def test_text_is_forwarded_verbatim(fake_agent) -> None:
    """H2: the gateway does not trim or rewrite what it receives."""
    text = "  prioritise eMBB\tpour 30 min  "
    call("submit_intent", {"text": text})
    assert json.loads(fake_agent[0].content) == {"text": text}
    assert gateway_events()[0]["data"]["arguments"] == {"text": text}


# ------------------------------------------------------------------ invariant 2: imports and addresses


def imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


def test_every_gateway_file_is_scanned() -> None:
    assert {p.name for p in GATEWAY_DIR.glob("*.py")} == set(GATEWAY_FILES)


@pytest.mark.parametrize("module", GATEWAY_FILES)
def test_gateway_imports_nothing_that_reaches_the_ran_or_the_store(module: str) -> None:
    imports = imported_modules(GATEWAY_DIR / module)
    refused = {name for name in imports
               if name.split(".")[0] in FORBIDDEN and name not in ALLOWED_FROM_FORBIDDEN}
    assert not refused, f"gateway/{module} imports {sorted(refused)}"


def test_nothing_forbidden_is_loaded_even_indirectly() -> None:
    """The static scan sees our files only; this sees everything the import pulls in (the SDK too)."""
    probe = ("import sys, gateway.server, gateway.client, gateway.__main__\n"
             "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
             f"{sorted(FORBIDDEN | {'_sqlite3'})!r} and m not in {sorted(ALLOWED_FROM_FORBIDDEN | {'store'})!r})\n"
             "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", probe], cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_every_request_goes_to_the_one_address_and_known_endpoints(fake_agent) -> None:
    call("submit_intent", {"text": "prioritise eMBB"})
    call("get_status", {"intent_id": "i0001"})
    call("get_status")
    call("get_kpis")
    call("get_kpis", {"slice": "urllc", "window_s": 30})
    base = httpx.URL(server.agent_url())
    assert {(r.url.scheme, r.url.host, r.url.port) for r in fake_agent} == {(base.scheme, base.host, base.port)}
    shapes = {(r.method, "/intents/{id}" if r.url.path.startswith("/intents/") else r.url.path) for r in fake_agent}
    assert shapes == ENDPOINTS


# ------------------------------------------------------------------ argument checks and the log


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("submit_intent", {"text": "   "}),
        ("submit_intent", {"text": "x" * 2001}),
        ("get_kpis", {"slice": "mmtc"}),
        ("get_kpis", {"window_s": 0}),
        ("get_kpis", {"window_s": 7200}),
        ("get_status", {"intent_id": "../kpis"}),
    ],
)
def test_bad_arguments_are_refused_and_logged_without_any_request(fake_agent, tool, arguments) -> None:
    result = call(tool, arguments)
    assert result.is_error and "invalid arguments" in result.content[0].text
    assert fake_agent == []
    [event] = gateway_events()
    assert event["data"]["tool"] == tool and event["data"]["outcome"] == "invalid_args"


def test_one_logged_event_per_call(fake_agent) -> None:
    before = time.time()
    call("get_kpis")
    call("get_status", {"intent_id": "i0001"})
    events = gateway_events()
    assert [(e["event"], e["data"]["tool"], e["data"]["outcome"]) for e in events] == [
        ("tool_call", "get_kpis", "ok"), ("tool_call", "get_status", "ok")]
    assert all(e["ts_wall"] >= before and e["data"]["duration_ms"] >= 0 for e in events)
    assert events[1]["intent_id"] == "i0001"


def test_unreachable_agent_is_an_error_not_a_crash(monkeypatch, events_file) -> None:
    def refuse_connection(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)
    monkeypatch.setattr(server, "http", httpx.Client(base_url=server.agent_url(),
                                                     transport=httpx.MockTransport(refuse_connection)))
    result = call("get_status")
    assert result.is_error and "unreachable" in result.content[0].text
    assert gateway_events()[0]["data"]["outcome"] == "unreachable"


# ------------------------------------------------------------------ the §5.4 demo, through the gateway


def test_demo_scenario_through_the_gateway(ric_db, tiny_policy_file, monkeypatch) -> None:
    """Accepted with confirmation, rejected out of bounds, and the out-of-scope request that
    changes nothing. The real agent service answers the gateway (in-process); the rule-based
    agent stands in for the LLM; the console's confirm is called directly."""
    monkeypatch.setattr(server, "http", TestClient(app, base_url=server.agent_url()))
    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(15)
    ctx = AgentContext(conn=ric_db, llm=None,
                       settings=Settings(mode="demo", observe_s=0.0, max_revisions=0, wait_for_confirmation=False))

    def submit_and_process(text: str) -> dict:
        created = body(call("submit_intent", {"text": text}))
        assert created["state"] == "received"
        assert process_one(ctx) == created["intent_id"]  # the worker's job, done here in-process
        return body(call("get_status", {"intent_id": created["intent_id"]}))

    # 1. accepted: waits for the console, nothing applied until it confirms
    accepted = submit_and_process("Garantis 10 Mb/s au slice URLLC pendant 10 minutes.")
    assert accepted["state"] == "awaiting_confirmation"
    assert a1_policy.current_schedule(ric_db) == []
    apply_intent(ric_db, accepted["intent_id"], now_wall=time.time(), confirmed_by="console")
    runner.run(2)
    applied = body(call("get_status", {"intent_id": accepted["intent_id"]}))
    assert applied["state"] == "applied" and applied["in_effect"]

    # 2. rejected by the verifier, with the reason visible in get_status
    schedule_before = a1_policy.current_schedule(ric_db)
    rejected = submit_and_process("Garantis 500 Mb/s au slice URLLC.")
    assert rejected["state"] == "rejected" and rejected["reason"]

    # 3. out of scope: no tool can set the split, and nothing changes
    out_of_scope = submit_and_process("Fixe le partage des PRB à 90/5/5.")
    assert out_of_scope["state"] == "rejected"
    assert a1_policy.current_schedule(ric_db) == schedule_before
    runner.run(2)
    assert runner.row is not None and runner.row.intent_ids == (accepted["intent_id"],)

    # The gateway log names only the three tools.
    assert {e["data"]["tool"] for e in gateway_events()} <= THE_THREE_TOOLS

