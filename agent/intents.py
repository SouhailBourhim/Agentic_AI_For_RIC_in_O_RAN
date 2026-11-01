"""The intent state machine: from operator text to an applied (or rejected) objective.

    received ──parse──> interpreted ──verify──> awaiting_confirmation ──console──> applied
        │                    │                        │       │
        └─ rejected          └─ rejected              │       └─ rejected (refused / re-check)
           (not understood)     (verifier, reason)    └─ expired (no answer before the deadline)

In evaluation mode there is no operator: interpreted ──> applied (confirmed_by='auto-eval').
Later an applied intent may be cancelled from the console.

Expiry of a pending confirmation is lazy: no background task, every read compares the
deadline with the wall clock first (expire_pending). The deadline is wall-clock time because
it measures how long a human takes to answer; the objective's own window is backend time.
Every function takes the wall clock as an argument, so tests can move time by hand.
"""

import sqlite3
from dataclasses import dataclass

from agent.apply import applied_intents, apply_intent, backend_now, decision_dict, verifier_state
from agent.parser import parse_intent
from store import a1_policy, intents
from store.events import log_event
from store.intents import IntentRecord
from verifier.verify import verify


@dataclass(frozen=True)
class Settings:
    mode: str = "demo"  # 'demo': the console confirms; 'eval': applied without confirmation
    confirm_timeout_s: float = 120.0  # wall clock


def submit(conn: sqlite3.Connection, text: str, settings: Settings, now_wall: float) -> IntentRecord:
    """Store, parse and verify an intent. Never applies anything in demo mode (invariant 1)."""
    state = verifier_state(conn)  # raises ApplyError if no runner has published the cell yet
    record = intents.create(conn, text, now_wall, issued_backend=state.now_s)
    log_event("agent", "received", record.intent_id, state.now_s, {"text": text})

    result = parse_intent(text, state.now_s)
    if result.objective is None:
        log_event("agent", "not_understood", record.intent_id, state.now_s, {"problem": result.problem})
        return intents.set_state(conn, record.intent_id, "rejected", now_wall, parsed=result.parsed,
                                 reason=f"parser: {result.problem}")
    intents.set_state(conn, record.intent_id, "interpreted", now_wall, parsed=result.parsed,
                      objective=result.objective)
    log_event("agent", "interpreted", record.intent_id, state.now_s,
              {"parsed": result.parsed, "objective": result.objective.model_dump()})

    decision = verify(result.objective, state, new_intent_id=record.intent_id)
    log_event("verifier", "verified", record.intent_id, state.now_s, decision_dict(decision))
    if not decision.accepted:
        return intents.set_state(conn, record.intent_id, "rejected", now_wall, verifier=decision_dict(decision),
                                 reason=f"verifier ({decision.check}): {decision.reason}")
    if settings.mode == "eval":
        intents.set_state(conn, record.intent_id, "interpreted", now_wall, verifier=decision_dict(decision))
        return apply_intent(conn, record.intent_id, now_wall, confirmed_by="auto-eval")
    log_event("agent", "awaiting_confirmation", record.intent_id, state.now_s,
              {"deadline_wall": now_wall + settings.confirm_timeout_s})
    return intents.set_state(conn, record.intent_id, "awaiting_confirmation", now_wall,
                             verifier=decision_dict(decision), reason=decision.reason,
                             confirm_deadline=now_wall + settings.confirm_timeout_s)


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
    }
    if record.state == "applied" and not in_effect and record.objective is not None:
        until = record.objective.valid_until
        result["note"] = "ended" if until is not None and now >= until else "scheduled, not started yet"
    return result


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
