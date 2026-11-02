"""The intent state machine: from operator text to an applied (or rejected) objective.

    received ─worker─> processing ─parse/plan─> interpreted ─verify─> awaiting_confirmation ─console─> applied
                           │                         │                     │        │
                           └─ rejected (clarify)     └─ rejected           │        └─ rejected (refused / re-check)
                                                        (verifier)         └─ expired (no answer in time)

M04b: the HTTP service only stores the intent ('received', submit below: hard invariant 1).
The agent worker (agent/worker.py) takes it ('processing') and runs the LangGraph loop
(agent/graph.py). In evaluation mode there is no operator: interpreted ──> applied
(confirmed_by='auto-eval'). An applied intent may later be cancelled from the console, or
superseded by its revision (a new intent created by the agent, at most one per intent).

Expiry of a pending confirmation is lazy: no background task, every read compares the
deadline with the wall clock first (expire_pending). The deadline is wall-clock time because
it measures how long a human takes to answer; the objective's own window is backend time.
Every function takes the wall clock as an argument, so tests can move time by hand.
"""

import sqlite3

from agent.apply import applied_intents, backend_now, verifier_state
from store import a1_policy, intents
from store.events import log_event, read_events
from store.intents import IntentRecord


def submit(conn: sqlite3.Connection, text: str, now_wall: float) -> IntentRecord:
    """Store the operator's text as a new intent, state 'received'. Parses, decides and applies
    nothing: the agent worker does that (hard invariant 1: submit_intent only creates a request)."""
    state = verifier_state(conn)  # raises ApplyError if no runner has published the cell yet
    record = intents.create(conn, text, now_wall, issued_backend=state.now_s)
    log_event("agent", "received", record.intent_id, state.now_s, {"text": text})
    return record


def expire_pending(conn: sqlite3.Connection, now_wall: float) -> list[str]:
    """Mark every intent whose confirmation deadline has passed as expired. Returns their ids."""
    expired = []
    for record in intents.in_states(conn, ("awaiting_confirmation",)):
        if record.confirm_deadline is not None and now_wall >= record.confirm_deadline:
            intents.set_state(conn, record.intent_id, "expired", now_wall,
                              reason="not confirmed before the deadline", confirm_deadline=None)
            log_event("agent", "expired", record.intent_id, backend_now(conn))
            expired.append(record.intent_id)
    return expired


def refuse(conn: sqlite3.Connection, intent_id: str, now_wall: float) -> IntentRecord:
    """The operator says no (console). Only a pending intent can be refused."""
    expire_pending(conn, now_wall)
    record = intents.get(conn, intent_id)
    if record is None or record.state != "awaiting_confirmation":
        raise ValueError(f"{intent_id} is not awaiting confirmation")
    log_event("console", "refused", intent_id, backend_now(conn))
    return intents.set_state(conn, intent_id, "rejected", now_wall, reason="refused by the operator",
                             confirm_deadline=None)


def status(conn: sqlite3.Connection, intent_id: str, now_wall: float) -> dict | None:
    """get_status: the intent record, plus whether it is in force right now and merged with what.

    An older intent stays 'applied' while a newer one runs; in_effect and merged_with show it
    is still part of the merged objective (M03 rule 4).
    """
    expire_pending(conn, now_wall)
    record = intents.get(conn, intent_id)
    if record is None:
        return None
    now = backend_now(conn)
    row = a1_policy.in_force(conn, now)
    in_effect = row is not None and intent_id in row.intent_ids
    result = {
        "intent_id": record.intent_id,
        "text": record.text,
        "state": record.state,
        "reason": record.reason,
        "objective": record.objective.model_dump() if record.objective else None,
        "verifier": record.verifier,
        "confirmed_by": record.confirmed_by,
        "confirm_deadline_wall": record.confirm_deadline,
        "state_times_wall": record.state_times,
        "backend_now": now,
        "in_effect": in_effect,
        "merged_with": [i for i in row.intent_ids if i != intent_id] if in_effect else [],
        "supersedes": record.supersedes,
        "agent_steps": agent_steps(record),
    }
    if record.state == "applied" and not in_effect and record.objective is not None:
        until = record.objective.valid_until
        result["note"] = "ended" if until is not None and now >= until else "scheduled, not started yet"
    return result


def agent_steps(record: intents.IntentRecord) -> list[dict]:
    """The agent's steps for this intent, from the event log.

    The log keeps every run, and ids restart at i0001 when the runner empties ric.db: only events
    written since this intent was created belong to it (seen in the M04b live demo, where a new
    i0001 showed the steps of the previous run's i0001).
    """
    created = min(record.state_times.values(), default=0.0)
    return [e["data"] for e in read_events() if e["intent_id"] == record.intent_id
            and e["event"] in ("agent_step", "agent_done") and e["ts_wall"] >= created]


def overview(conn: sqlite3.Connection, now_wall: float) -> dict:
    """get_status without an id: the policy in force, the timetable and the recent intents."""
    expire_pending(conn, now_wall)
    now = backend_now(conn)
    row = a1_policy.in_force(conn, now)
    return {
        "backend_now": now,
        "in_force": None if row is None else {
            "policy_id": row.policy_id, "intent_ids": list(row.intent_ids),
            "objective": row.objective.model_dump(), "expires_at": row.expires_at},
        "timetable": [{"policy_id": r.policy_id, "valid_from": r.valid_from, "expires_at": r.expires_at,
                       "intent_ids": list(r.intent_ids)} for r in a1_policy.current_schedule(conn)],
        "applied": [i.intent_id for i in applied_intents(conn)],
        "recent": [{"intent_id": r.intent_id, "state": r.state, "text": r.text, "reason": r.reason}
                   for r in intents.recent(conn)],
    }
