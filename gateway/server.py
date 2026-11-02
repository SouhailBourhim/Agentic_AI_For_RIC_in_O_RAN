"""The MCP gateway: exactly three tools, each one a single call to the agent's HTTP API.

  submit_intent(text)               POST /intents          creates a request, applies nothing
  get_kpis(slice, window_s)         GET  /kpis             read only
  get_status(intent_id)             GET  /intents/{id}     read only (no id: GET /status)

Why it is safe by construction (hard invariants 1 and 2):
- There is no fourth tool, no MCP resource and no MCP prompt, so a client (Hermes, the CLI in
  gateway/client.py, any LLM) can do nothing else through this server.
- The gateway knows one address, RIC_AGENT_URL, and that service has no endpoint to confirm,
  apply or set a PRB split (tests/test_agent.py). It imports nothing that reaches the RAN or the
  SQLite store: the one module it uses from store/ is events.py, which only appends JSONL.

Every call, accepted or refused, is one 'tool_call' event in var/events.jsonl (source 'gateway').
M08b computes the interface metrics from it: calls per tool and outcome, timestamps, and the text
received (to compare with what the operator typed).

Arguments are checked here, not only by the agent service, so that a refusal is logged with the
reason 'invalid_args'. That is why the parameters are plain str/float and not Literal types: the
SDK would refuse a bad Literal before our code runs, and the refusal would leave no trace.
"""

import math
import os
import re
import time
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from store.events import log_event

DEFAULT_AGENT_URL = "http://127.0.0.1:8000"
SLICES = ("embb", "urllc", "be")
INTENT_ID = re.compile(r"^i\d{4}$")
MAX_TEXT = 2000  # the agent service refuses longer texts too (agent/service.py, IntentIn)
MAX_WINDOW_S = 3600.0
TIMEOUT_S = 10.0  # submit only stores the intent, so the agent service answers fast


def agent_url() -> str:
    return os.environ.get("RIC_AGENT_URL", DEFAULT_AGENT_URL)


# The one HTTP client of the gateway, bound to the one address it knows. Tests replace it.
http: httpx.Client = httpx.Client(base_url=agent_url(), timeout=TIMEOUT_S)

mcp = MCPServer(
    "agentic-ric-gateway",
    instructions=(
        "Gateway to an agentic RAN intelligent controller. Three tools only. submit_intent sends "
        "the operator's request word for word and only creates a request: the agent interprets "
        "it, a deterministic verifier checks it, and an operator confirms it in a separate "
        "console before anything changes. get_status and get_kpis are read only. Nothing here "
        "can set the PRB split or send a command to the radio."
    ),
)


def log_call(tool: str, arguments: dict[str, Any], outcome: str, started: float,
             intent_id: str | None = None, detail: str | None = None) -> None:
    data = {"tool": tool, "arguments": arguments, "outcome": outcome,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1)}
    if detail is not None:
        data["detail"] = detail
    log_event("gateway", "tool_call", intent_id, data=data)


def refuse(tool: str, arguments: dict[str, Any], reason: str) -> ToolError:
    """Log a refused call (no HTTP request is made) and return the error to raise."""
    log_call(tool, arguments, "invalid_args", time.perf_counter(), detail=reason)
    return ToolError(f"invalid arguments: {reason}")


def call_agent(tool: str, arguments: dict[str, Any], method: str, path: str, **request: Any) -> dict:
    """One HTTP request to the agent service, logged as one tool_call event."""
    started = time.perf_counter()
    try:
        response = http.request(method, path, **request)
    except httpx.HTTPError as err:
        log_call(tool, arguments, "unreachable", started, detail=str(err))
        raise ToolError(f"the agent service at {agent_url()} is unreachable: {err}") from err
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text}
    if not response.is_success:
        detail = str(body.get("detail", body))
        log_call(tool, arguments, f"http_{response.status_code}", started, detail=detail)
        raise ToolError(f"agent service answered HTTP {response.status_code}: {detail}")
    log_call(tool, arguments, "ok", started, intent_id=body.get("intent_id"))
    return body


@mcp.tool()
def submit_intent(text: str) -> dict:
    """Submit the operator's intent, word for word (French or English), for example
    "Garantis 10 Mb/s au slice URLLC pendant 10 minutes."

    This only creates a request; it changes nothing on the network. The agent interprets it, the
    deterministic verifier checks it, and in demo mode an operator confirms it in a separate
    console. Returns the intent id and its first state ('received'): follow it with get_status.
    """
    arguments = {"text": text}
    if not text.strip():
        raise refuse("submit_intent", arguments, "text is empty")
    if len(text) > MAX_TEXT:
        raise refuse("submit_intent", arguments, f"text is longer than {MAX_TEXT} characters")
    return call_agent("submit_intent", arguments, "POST", "/intents", json={"text": text})


@mcp.tool()
def get_kpis(slice: str | None = None, window_s: float = 10.0) -> dict:
    """Recent KPIs per slice (throughput, PRB share, latency): mean, p95 and last value over the
    last window_s seconds of the cell's clock. slice is 'embb', 'urllc' or 'be'; omit it for all
    three. Read only.
    """
    arguments = {"slice": slice, "window_s": window_s}
    if slice is not None and slice not in SLICES:
        raise refuse("get_kpis", arguments, f"slice must be one of {', '.join(SLICES)}")
    if not (math.isfinite(window_s) and 0.0 < window_s <= MAX_WINDOW_S):
        raise refuse("get_kpis", arguments, f"window_s must be in (0, {MAX_WINDOW_S:g}] seconds")
    params: dict[str, Any] = {"window_s": window_s}
    if slice is not None:
        params["slice"] = slice
    return call_agent("get_kpis", arguments, "GET", "/kpis", params=params)


@mcp.tool()
def get_status(intent_id: str | None = None) -> dict:
    """Where an intent stands: received, processing, interpreted, rejected (with the reason),
    awaiting_confirmation, applied, expired or cancelled; the objective it carries and whether it
    is in force. Without intent_id: the objective timetable and the recent intents. Read only.
    """
    arguments = {"intent_id": intent_id}
    if intent_id is None:
        return call_agent("get_status", arguments, "GET", "/status")
    if not INTENT_ID.match(intent_id):
        raise refuse("get_status", arguments, "intent_id looks like 'i0001'")
    return call_agent("get_status", arguments, "GET", f"/intents/{intent_id}")
