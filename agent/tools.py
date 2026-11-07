"""The agent's internal tools, and which ones each controller mode may use (hard invariant 4).

| tool              | proposed (LLM + RL) | llm_only (baseline) |
|-------------------|---------------------|---------------------|
| get_kpis          | yes                 | yes                 |
| wait_and_observe  | yes                 | yes                 |
| set_rl_objective  | yes                 | no                  |
| set_slice_share   | NO                  | yes                 |

tools_for_mode() is the only source of these lists; set_slice_share also refuses to run in the
proposed mode even if called directly. These are internal tools: the MCP gateway (M05a) exposes
only submit_intent, get_kpis and get_status, never these.
"""

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from agent.apply import ApplyError, applied_intents, apply_shares, backend_now, decision_dict, verifier_state
from agent.schema import objective_from_decision
from store import kpi
from verifier.objective import SLICES, RLObjective
from verifier.verify import Decision, verify

TOOLS_BY_MODE = {
    "proposed": ("get_kpis", "wait_and_observe", "set_rl_objective"),
    "llm_only": ("get_kpis", "wait_and_observe", "set_slice_share"),
}


class ToolNotAllowed(Exception):
    """A tool was called in a mode that does not have it."""


@dataclass
class Settings:
    """How the agent runs. Read from the environment by agent/worker.py."""

    mode: str = "demo"  # 'demo': the console confirms; 'eval': applied without confirmation
    controller: str = "proposed"  # 'proposed' (LLM + RL) or 'llm_only' (the M08a baseline)
    confirm_timeout_s: float = 120.0  # wall clock
    observe_s: float = 20.0  # backend seconds observed after an apply (0 = do not observe)
    max_steps: int = 8  # tool calls per intent (proposed mode)
    max_rejections: int = 2  # verifier rejections that still lead to a new plan
    max_revisions: int = 1  # revisions of an applied objective per intent
    good_enough: float = 0.9  # compliance at which no revision is made, whatever the model says
    wait_for_confirmation: bool = True  # False: stop at awaiting_confirmation (tests)
    poll_s: float = 0.5  # wall seconds between two reads while waiting for the console


@dataclass
class AgentContext:
    """Everything a tool or a node needs, injectable so tests need no clock, GPU or runner process."""

    conn: sqlite3.Connection
    settings: Settings
    llm: object | None  # an LLMClient, or None for the rule-based agent
    wait_backend: Callable[[float], None] | None = None  # let N backend seconds pass (default: poll ric.db)
    sleep: Callable[[float], None] = time.sleep  # wall-clock sleep while waiting for the console
    now_wall: Callable[[], float] = time.time
    trace: list[dict] = field(default_factory=list)  # the steps of the current intent, for get_status


def tools_for_mode(controller: str) -> tuple[str, ...]:
    if controller not in TOOLS_BY_MODE:
        raise ValueError(f"unknown controller mode {controller!r}")
    return TOOLS_BY_MODE[controller]


# ------------------------------------------------------------------ definitions given to the LLM

SLICE_NUMBERS = {"type": "object", "properties": {s: {"type": ["number", "null"]} for s in SLICES}}
DEFINITIONS = {
    "get_kpis": {"description": "Recent KPIs per slice (mean throughput, offered load, p95 latency, PRB share).",
                 "parameters": {"type": "object", "properties": {
                     "window_s": {"type": "number", "description": "seconds of backend time, default 10"}}}},
    "wait_and_observe": {"description": "Let the network run, then return the KPIs of that period.",
                         "parameters": {"type": "object", "properties": {"seconds": {"type": "number"}},
                                        "required": ["seconds"]}},
    "set_rl_objective": {
        "description": "Give the RL controller a new objective. Checked by the verifier before anything applies.",
        "parameters": {"type": "object", "properties": {
            "importance": {"type": "object", "description": "0-10 per slice",
                           "properties": {s: {"type": "number"} for s in SLICES}},
            "min_thr_mbps": {**SLICE_NUMBERS, "description": "minimum throughput per slice, Mb/s, or null"},
            "max_latency_ms": {"type": ["number", "null"], "description": "URLLC latency bound, ms"},
            "duration_s": {"type": ["number", "null"], "description": "seconds; null = 600"}},
            "required": ["importance"]}},
    "set_slice_share": {
        "description": "Set the PRB split directly (embb, urllc, be sum to 1, each >= 0.05). Verified first.",
        "parameters": {"type": "object", "properties": {s: {"type": "number"} for s in SLICES},
                       "required": list(SLICES)}},
    "finish": {"description": "Stop working on this intent, with a one-sentence reason.",
               "parameters": {"type": "object", "properties": {"reason": {"type": "string"}},
                              "required": ["reason"]}},
}


def definitions(controller: str) -> list[dict]:
    """The tool list sent to the LLM: the mode's tools plus finish."""
    names = [*tools_for_mode(controller), "finish"]
    return [{"type": "function", "function": {"name": n, **DEFINITIONS[n]}} for n in names]


# ------------------------------------------------------------------ implementations


def get_kpis(ctx: AgentContext, window_s: float = 10.0) -> dict:
    """A compact summary (small enough for an 8B model's context): per slice, means over the window."""
    raw = kpi.get_kpis(ctx.conn, None, float(window_s or 10.0))
    summary: dict = {"backend_t": raw["ts"], "window_s": raw["window_s"]}
    for s, stats in raw["slices"].items():
        summary[s] = {"thr_mbps": round(stats["dl_thr_mbps"]["mean"], 1),
                      "load_mbps": round(stats.get("offered_load_mbps", {}).get("mean", 0.0), 1),
                      "latency_p95_ms": round(stats.get("latency_ms", {}).get("p95", 0.0), 1),
                      "prb_share": round(stats["prb_share_alloc"]["mean"], 3)}
    return summary


def wait_backend_default(ctx: AgentContext, seconds: float) -> None:
    """Wait until the runner's clock (latest KPI timestamp) has moved by `seconds`.

    A wall-clock safety limit stops the wait if the runner is not running (3 x seconds + 30 s).
    """
    start = backend_now(ctx.conn)
    deadline = time.monotonic() + 3 * seconds + 30
    while backend_now(ctx.conn) < start + seconds:
        if time.monotonic() > deadline:
            raise TimeoutError(f"backend clock stuck at {backend_now(ctx.conn):g} s: is rl.runner running?")
        time.sleep(0.2)


def wait_and_observe(ctx: AgentContext, seconds: float) -> dict:
    seconds = max(1.0, float(seconds))
    (ctx.wait_backend or (lambda s: wait_backend_default(ctx, s)))(seconds)
    return get_kpis(ctx, seconds)


def propose_objective(ctx: AgentContext, intent_id: str, supersedes: str | None, importance: dict,
                      min_thr_mbps: dict | None = None, max_latency_ms: float | None = None,
                      duration_s: float | None = None) -> tuple[Decision, RLObjective | None]:
    """set_rl_objective, verification part: build the objective and verify it. Applies nothing.

    A revision is verified as replacing the intent it supersedes (verify(..., replacing=...)).
    """
    now = backend_now(ctx.conn)
    try:
        objective = objective_from_decision(importance or {}, min_thr_mbps, max_latency_ms, duration_s, now)
    except ValueError as err:
        return Decision(accepted=False, check="schema", reason=f"invalid objective: {err}"), None
    label = intent_id if supersedes is None else f"revision of {supersedes}"
    state = verifier_state(ctx.conn, applied_intents(ctx.conn))  # the parent included: the step is from it
    return verify(objective, state, new_intent_id=label, replacing=supersedes), objective


def set_slice_share(ctx: AgentContext, intent_id: str, shares: dict, valid_s: float) -> Decision:
    """The LLM-only baseline: a direct PRB split, verified by verify_shares, applied by the runner."""
    if "set_slice_share" not in tools_for_mode(ctx.settings.controller):
        raise ToolNotAllowed(f"set_slice_share does not exist in the {ctx.settings.controller!r} mode")
    if ctx.settings.mode != "eval":
        raise ApplyError("the LLM-only baseline runs in evaluation mode only (no confirmation per split)")
    values = [shares.get(s) for s in SLICES]
    return apply_shares(ctx.conn, intent_id, values, valid_s, ctx.now_wall())


def compliance(ctx: AgentContext, objective: RLObjective, window_s: float) -> dict[str, float]:
    """Share of the last window's ticks in which each stated target and the latency bound were met.

    Deterministic; the reflection reads it. A throughput target counts as met when the slice got
    min(target, offered load): a slice asking for less than its target cannot miss it.
    """
    end = backend_now(ctx.conn)
    rows = ctx.conn.execute("SELECT slice, dl_thr_mbps, offered_load_mbps, latency_ms FROM kpi WHERE ts > ?",
                            (end - window_s,)).fetchall()
    result = {}
    for s in SLICES:
        target = getattr(objective.min_thr_mbps, s)
        mine = [r for r in rows if r[0] == s]
        if target > 0 and mine:
            ok = [r[1] >= min(target, r[2] if r[2] is not None else target) - 1e-9 for r in mine]
            result[f"{s}_min_thr"] = sum(ok) / len(ok)
    bound = objective.max_latency_ms
    urllc = [r for r in rows if r[0] == "urllc" and r[3] is not None]
    if bound is not None and urllc:
        result["urllc_latency"] = sum(r[3] <= bound for r in urllc) / len(urllc)
    return result


__all__ = ["AgentContext", "Settings", "ToolNotAllowed", "compliance", "decision_dict", "definitions",
           "get_kpis", "propose_objective", "set_slice_share", "tools_for_mode", "wait_and_observe"]
