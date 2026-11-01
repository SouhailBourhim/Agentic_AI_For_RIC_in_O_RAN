"""The agent loop of the proposal's figure 3, as a LangGraph state graph.

    parse ─> plan ─> act ─┬─ get_kpis / wait_and_observe ───────────────> plan
      │                    ├─ set_rl_objective ─> verifier ─ reject ──────> plan   (at most 2)
      v                    │                        └ accept ─> confirm ─> observe ─> reflect
    end (clarify)          ├─ set_slice_share (llm_only) ─> verifier ─> observe ─> plan
                           └─ finish ─> end                       reflect: done ─> end
                                                                  reflect: revise ─> plan (at most 1)

Each node is a plain function of the state, returning the fields it changes; `route` names
the next node. The LLM is called in three places only: parse (structured JSON), plan (one tool
call) and reflect (structured JSON). With ctx.llm None the same graph runs as the rule-based
agent (regex parse, fixed decision rule), used in CI and as a fallback.

Every node logs an `agent_step` event; nothing here touches the RAN: configuration changes go
through propose_objective / set_slice_share, i.e. the verifier, then agent/apply.py.
"""

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from agent import tools
from agent.apply import apply_intent, backend_now, decision_dict
from agent.intents import expire_pending
from agent.parser import parse_facts
from agent.prompts import PARSE_VERSION, PLAN_SYSTEM, REFLECT_SYSTEM, parse_messages
from agent.schema import DEFAULT_DURATION_S, ParsedIntent
from agent.tools import AgentContext
from store import intents
from store.events import log_event
from verifier.objective import SLICES, RLObjective

SCRATCH_LINES = 8  # the plan step sees only the latest tool results: small context for an 8B model
REFLECT_SCHEMA = {"type": "object", "properties": {"decision": {"type": "string", "enum": ["done", "revise"]},
                                                   "reason": {"type": "string"}},
                  "required": ["decision", "reason"]}


class AgentState(TypedDict, total=False):
    intent_id: str  # the operator's intent
    active_id: str | None  # the intent whose objective is applied (the original or its revision)
    waiting_id: str | None  # the intent waiting for the console
    text: str
    facts: dict | None  # ParsedIntent
    steps: int
    rejections: int
    revisions: int
    scratch: list[str]  # tool results and feedback shown to the plan step
    rejected_calls: list[str]  # proposals the verifier rejected (JSON), never sent again
    pending: dict | None  # the tool call chosen by plan: {"name", "arguments"}
    objective: dict | None  # the applied objective of active_id
    compliance: dict | None
    end_backend: float | None  # llm_only: when the intent's duration is over
    route: str
    outcome: str  # final: applied, rejected, awaiting_confirmation, expired, failed
    reason: str


def llm_parse(llm, text: str) -> tuple[ParsedIntent | None, float, int]:
    """The parse step with an LLM: (facts or None, total latency, attempts). One retry on an
    invalid reply. Shared by the graph and the extraction score (eval/extraction.py)."""
    latency = 0.0
    for attempt in (1, 2):
        reply = llm.chat(parse_messages(text), schema=ParsedIntent.model_json_schema())
        latency += reply.latency_s
        try:
            return ParsedIntent.model_validate_json(reply.content), latency, attempt
        except ValidationError:
            continue
    return None, latency, 2


def run_intent(ctx: AgentContext, intent_id: str) -> AgentState:
    """Run the loop for one intent (already claimed by the worker, state 'processing')."""
    record = intents.get(ctx.conn, intent_id)
    graph = build_graph(ctx)
    limit = 50 if ctx.settings.controller == "proposed" else 2000  # LangGraph's recursion guard
    state = graph.invoke({"intent_id": intent_id, "active_id": None, "waiting_id": None, "text": record.text,
                          "steps": 0, "rejections": 0, "revisions": 0, "scratch": [], "rejected_calls": []},
                         {"recursion_limit": limit})
    finish_record(ctx, state)
    return state


def build_graph(ctx: AgentContext):  # noqa: C901 (one closure per node, kept together on purpose)
    s = ctx.settings
    conn = ctx.conn

    def step(state: AgentState, node: str, **data: Any) -> None:
        entry = {"node": node, "backend_t": backend_now(conn), **data}
        ctx.trace.append(entry)
        log_event("agent", "agent_step", state["intent_id"], entry["backend_t"], entry)

    def note(state: AgentState, line: str) -> list[str]:
        return [*state.get("scratch", []), line][-SCRATCH_LINES:]

    def reject(state: AgentState, reason: str) -> dict:
        """End the intent as rejected (only if nothing of it is applied)."""
        current = intents.get(conn, state["intent_id"])
        if current.state in ("processing", "interpreted"):
            intents.set_state(conn, state["intent_id"], "rejected", ctx.now_wall(), reason=reason)
        return {"route": "end", "outcome": "rejected", "reason": reason}

    def done(state: AgentState, reason: str) -> dict:
        return {"route": "end", "outcome": "applied", "reason": reason}

    # ------------------------------------------------------------------ parse
    def parse(state: AgentState) -> dict:
        if ctx.llm is None:
            facts, latency, attempts = parse_facts(state["text"]), 0.0, 0
        else:
            facts, latency, attempts = llm_parse(ctx.llm, state["text"])
        step(state, "parse", prompt=PARSE_VERSION, latency_s=round(latency, 2), attempts=attempts,
             facts=facts.model_dump() if facts else None)
        if facts is None:
            return reject(state, "agent: the model returned no valid parse (twice)")
        if not facts.understood:
            return reject(state, f"needs clarification: {facts.clarification or 'the request is ambiguous'}")
        intents.set_state(conn, state["intent_id"], "processing", ctx.now_wall(), parsed=facts.model_dump())
        duration = facts.duration_s if facts.duration_s is not None else DEFAULT_DURATION_S
        return {"facts": facts.model_dump(), "end_backend": backend_now(conn) + duration, "route": "plan"}

    # ------------------------------------------------------------------ plan
    def plan(state: AgentState) -> dict:
        if s.controller == "proposed" and state["steps"] >= s.max_steps:
            reason = f"step limit ({s.max_steps} tool calls) reached"
            return done(state, reason) if state.get("active_id") else reject(state, f"agent: {reason}")
        if ctx.llm is None:
            call, latency = rule_decision(state), 0.0
        else:
            call, latency = llm_decision(state)
        step(state, "plan", latency_s=round(latency, 2), tool=call["name"] if call else None,
             arguments=call["arguments"] if call else None)
        if call is None:
            reason = "the model did not choose a tool (twice)"
            return done(state, reason) if state.get("active_id") else reject(state, f"agent: {reason}")
        return {"pending": call, "steps": state["steps"] + 1, "route": "act"}

    def llm_decision(state: AgentState) -> tuple[dict | None, float]:
        user = (f"Operator request: {state['text']}\nExtracted facts: {json.dumps(state['facts'])}\n"
                f"Backend time: {backend_now(conn):g} s\n" + "\n".join(state.get("scratch", [])) +
                "\nChoose exactly one tool.")
        messages = [{"role": "system", "content": PLAN_SYSTEM[s.controller]}, {"role": "user", "content": user}]
        latency = 0.0
        for _ in (1, 2):
            reply = ctx.llm.chat(messages, tools=tools.definitions(s.controller))
            latency += reply.latency_s
            if reply.tool_calls:
                first = reply.tool_calls[0]
                return {"name": first.name, "arguments": dict(first.arguments)}, latency
            messages.append({"role": "user", "content": "You must call exactly one of the tools."})
        return None, latency

    def rule_decision(state: AgentState) -> dict | None:
        """The rule-based agent: one fixed translation of the facts, no correction after a rejection."""
        if s.controller != "proposed":
            raise ValueError("the rule-based agent only runs the proposed controller")
        if state["rejections"] > 0:
            return {"name": "finish", "arguments": {"reason": "the rule-based agent cannot correct a rejection"}}
        facts = state["facts"]
        targets = {k: v for k, v in facts["min_thr_mbps"].items() if v is not None}
        favoured = facts["priority"] or (next(iter(targets)) if len(targets) == 1 else None)
        high = 10 if state["revisions"] else 6  # a revision pushes the favoured slice harder
        importance = {sl: (high if sl == favoured else 2) for sl in SLICES} if favoured else {}
        return {"name": "set_rl_objective",
                "arguments": {"importance": importance, "min_thr_mbps": facts["min_thr_mbps"],
                              "max_latency_ms": facts["max_latency_ms"], "duration_s": facts["duration_s"]}}

    # ------------------------------------------------------------------ act
    def act(state: AgentState) -> dict:
        call = state["pending"]
        name, args = call["name"], call["arguments"] or {}
        if name != "finish" and name not in tools.tools_for_mode(s.controller):
            step(state, "act", tool=name, result="not available in this mode")
            return {"scratch": note(state, f"Tool {name} does not exist here."), "route": "plan"}
        if name == "finish":
            reason = str(args.get("reason") or "finished")
            step(state, "act", tool="finish", reason=reason)
            return done(state, reason) if state.get("active_id") else reject(state, f"agent: {reason}")
        if name == "get_kpis":
            result = tools.get_kpis(ctx, args.get("window_s") or 10.0)
            step(state, "act", tool=name, result=result)
            return {"scratch": note(state, f"get_kpis -> {json.dumps(result)}"), "route": "plan"}
        if name == "wait_and_observe":
            result = tools.wait_and_observe(ctx, args.get("seconds") or s.observe_s)
            step(state, "act", tool=name, result=result)
            return {"scratch": note(state, f"wait_and_observe -> {json.dumps(result)}"), "route": "plan"}
        key = json.dumps({"tool": name, **args}, sort_keys=True)
        if key in state.get("rejected_calls", []):  # the same proposal again: no new information
            step(state, "act", tool=name, result="identical to a rejected proposal")
            reason = "the model repeated a proposal the verifier had already rejected"
            last = next((line for line in reversed(state.get("scratch", [])) if line.startswith("VERIFIER")), "")
            final = f"{reason}; {last.removeprefix('VERIFIER REJECTED: ')}".rstrip("; ")
            return done(state, final) if state.get("active_id") else reject(state, final)
        if name == "set_slice_share":
            decision = tools.set_slice_share(ctx, state["intent_id"], args, valid_s=s.observe_s + 5.0)
            step(state, "act", tool=name, arguments=args, verifier=decision_dict(decision))
            if not decision.accepted:
                return {**rejected_action(state, decision.reason), "rejected_calls": [*state["rejected_calls"], key]}
            return {"active_id": state["intent_id"], "route": "observe",
                    "scratch": note(state, f"set_slice_share accepted: {json.dumps(args)}")}
        return set_objective(state, args, key)

    def rejected_action(state: AgentState, reason: str) -> dict:
        count = state["rejections"] + 1
        if count > s.max_rejections:
            final = f"verifier rejected {count} proposals; last: {reason}"
            return done(state, final) if state.get("active_id") else reject(state, final)
        return {"rejections": count, "route": "plan", "scratch": note(state, f"VERIFIER REJECTED: {reason}")}

    def set_objective(state: AgentState, args: dict, key: str) -> dict:
        revision_of = state.get("active_id")  # an applied objective exists: this is a revision
        decision, objective = tools.propose_objective(
            ctx, state["intent_id"], revision_of, args.get("importance") or {}, args.get("min_thr_mbps"),
            args.get("max_latency_ms"), args.get("duration_s"))
        step(state, "act", tool="set_rl_objective", arguments=args, verifier=decision_dict(decision))
        if not decision.accepted:
            return {**rejected_action(state, decision.reason), "rejected_calls": [*state["rejected_calls"], key]}
        now_wall = ctx.now_wall()
        if revision_of is None:
            target = state["intent_id"]
        else:
            target = intents.create(conn, state["text"], now_wall, backend_now(conn), supersedes=revision_of,
                                    state="interpreted").intent_id
        intents.set_state(conn, target, "interpreted", now_wall, objective=objective,
                          verifier=decision_dict(decision), parsed=state["facts"])
        if s.mode == "eval":
            record = apply_intent(conn, target, now_wall, confirmed_by="auto-eval")
            if record.state != "applied":
                return rejected_action(state, record.reason or "re-verification failed")
            return {"active_id": target, "objective": objective.model_dump(), "route": "observe"}
        intents.set_state(conn, target, "awaiting_confirmation", now_wall, reason=decision.reason,
                          confirm_deadline=now_wall + s.confirm_timeout_s)
        log_event("agent", "awaiting_confirmation", target, backend_now(conn),
                  {"deadline_wall": now_wall + s.confirm_timeout_s, "revision_of": revision_of})
        if not s.wait_for_confirmation:
            return {"waiting_id": target, "route": "end", "outcome": "awaiting_confirmation",
                    "reason": "waiting for the console"}
        return {"waiting_id": target, "route": "confirm"}

    # ------------------------------------------------------------------ confirm (demo mode)
    def confirm(state: AgentState) -> dict:
        target = state["waiting_id"]
        while True:
            expire_pending(conn, ctx.now_wall())
            record = intents.get(conn, target)
            if record.state == "applied":
                step(state, "confirm", intent=target, result="applied")
                return {"active_id": target, "objective": record.objective.model_dump(), "waiting_id": None,
                        "route": "observe"}
            if record.state in ("rejected", "expired"):
                step(state, "confirm", intent=target, result=record.state)
                if state.get("active_id"):  # a refused revision: the applied objective stays
                    return done(state, f"revision {target} {record.state}; the previous objective stays")
                return {"route": "end", "outcome": record.state, "reason": record.reason or record.state}
            ctx.sleep(s.poll_s)

    # ------------------------------------------------------------------ observe and reflect
    def observe(state: AgentState) -> dict:
        if s.observe_s <= 0:
            return done(state, "applied (no observation configured)")
        result = tools.wait_and_observe(ctx, s.observe_s)
        if s.controller == "llm_only":
            step(state, "observe", result=result)
            if backend_now(conn) >= (state.get("end_backend") or 0.0):
                return done(state, "the intent's duration is over")
            return {"scratch": note(state, f"KPIs since your last split: {json.dumps(result)}"), "route": "plan"}
        objective = RLObjective.model_validate(state["objective"])
        comp = tools.compliance(ctx, objective, s.observe_s)
        step(state, "observe", result=result, compliance=comp)
        return {"compliance": comp, "route": "reflect",
                "scratch": note(state, f"KPIs after applying: {json.dumps(result)}; compliance {json.dumps(comp)}")}

    def reflect(state: AgentState) -> dict:
        comp = state.get("compliance") or {}
        if state["revisions"] >= s.max_revisions:
            decision, reason, latency = "done", "revision limit reached", 0.0
        elif ctx.llm is None:
            missed = [k for k, v in comp.items() if v < 0.9]
            decision = "revise" if missed else "done"
            reason, latency = (f"missed: {', '.join(missed)}" if missed else "every target met"), 0.0
        else:
            user = f"Operator request: {state['text']}\nCompliance over the last {s.observe_s:g} s: {json.dumps(comp)}"
            reply = ctx.llm.chat([{"role": "system", "content": REFLECT_SYSTEM}, {"role": "user", "content": user}],
                                 schema=REFLECT_SCHEMA)
            latency = reply.latency_s
            try:
                parsed = json.loads(reply.content)
                decision, reason = parsed["decision"], str(parsed["reason"])
            except (ValueError, KeyError, TypeError):
                decision, reason = "done", "no valid reflection; keeping the objective"
            if decision not in ("done", "revise"):
                decision = "done"
        step(state, "reflect", decision=decision, reason=reason, compliance=comp, latency_s=round(latency, 2))
        if decision == "revise":
            return {"revisions": state["revisions"] + 1, "route": "plan",
                    "scratch": note(state, f"REFLECTION: revise the objective ({reason})")}
        return done(state, f"reflection: {reason}")

    graph = StateGraph(AgentState)
    for name, fn in [("parse", parse), ("plan", plan), ("act", act), ("confirm", confirm),
                     ("observe", observe), ("reflect", reflect)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "parse")
    routes = {"plan": "plan", "act": "act", "confirm": "confirm", "observe": "observe", "reflect": "reflect",
              "end": END}
    for name in ("parse", "plan", "act", "confirm", "observe", "reflect"):
        graph.add_conditional_edges(name, lambda st: st["route"], routes)
    return graph.compile()


def finish_record(ctx: AgentContext, state: AgentState) -> None:
    """Close the loop in the record: log the outcome; an intent left half-way is rejected."""
    record = intents.get(ctx.conn, state["intent_id"])
    if record.state in ("processing", "interpreted"):
        intents.set_state(ctx.conn, state["intent_id"], "rejected", ctx.now_wall(),
                          reason=state.get("reason") or "agent stopped")
    log_event("agent", "agent_done", state["intent_id"], backend_now(ctx.conn),
              {"outcome": state.get("outcome"), "reason": state.get("reason"), "steps": state.get("steps"),
               "revisions": state.get("revisions"), "active_id": state.get("active_id")})
