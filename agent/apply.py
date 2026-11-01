"""The apply step: THE ONLY WRITER OF a1_policy.

Used by the agent service (evaluation mode, no confirmation) and by the operator console
(demo mode, after confirmation). Both go through the same two functions:

- apply_intent: re-verify the intent against the intents applied *now* (the set may have
  changed since it was submitted), then write the new timetable;
- cancel_intent: withdraw an applied intent and write the timetable without it.

Each runs inside one database transaction, so two processes applying at the same moment
cannot both read the same "applied" set and overwrite each other's timetable.

The timetable: the merged objective (verifier/merge.py) changes only when an intent starts
or ends, so we cut time at those instants and write one a1_policy row per segment that has at
least one intent. Segments with none get no row: the runner uses the default objective.
"""

import sqlite3

from store import a1_policy, intents, kpi
from store.events import log_event
from store.intents import IntentRecord
from verifier.merge import ActiveIntent, merge
from verifier.objective import RLObjective
from verifier.verify import Decision, VerifierState, reverify_after_removal, verify


class ApplyError(Exception):
    """The intent cannot be applied or cancelled in its current state."""


def backend_now(conn: sqlite3.Connection) -> float:
    """The backend clock as last published by the runner (0 before its first tick)."""
    return kpi.latest_ts(conn) or 0.0


def applied_intents(conn: sqlite3.Connection) -> list[ActiveIntent]:
    """Every applied intent, as the verifier and merge() see them. 'Newest' = applied last."""
    return [
        ActiveIntent(intent_id=r.intent_id, issued_at=r.applied_backend or 0.0, objective=r.objective)
        for r in intents.in_states(conn, ("applied",))
        if r.objective is not None
    ]


def verifier_state(conn: sqlite3.Connection, active: list[ActiveIntent] | None = None) -> VerifierState:
    """What the verifier needs: the clock and cell published by the runner, and the applied intents."""
    found = kpi.read_backend_info(conn)
    if found is None:
        raise ApplyError("no RL runner has published the cell yet: start rl.runner first")
    info, _ = found
    now = backend_now(conn)
    active = applied_intents(conn) if active is None else active
    still_running = tuple(
        i for i in active if i.objective.valid_until is None or i.objective.valid_until > now
    )
    return VerifierState(now_s=now, capacity_mbps=info.capacity_mbps,
                         urllc_base_latency_ms=info.urllc_base_latency_ms, active=still_running)


def decision_dict(decision: Decision) -> dict:
    return {"accepted": decision.accepted, "check": decision.check, "reason": decision.reason}


def build_timetable(active: list[ActiveIntent], now: float) -> list[tuple[float, float | None, RLObjective, tuple[str, ...]]]:
    """(valid_from, expires_at, merged objective, intent_ids) for every segment from now on.

    Cut points: now, and every start or end of an intent that is still ahead. Between two cut
    points the set of active intents does not change, so neither does the merge.
    """
    cuts = {now}
    for i in active:
        for t in (i.objective.valid_from, i.objective.valid_until):
            if t is not None and t > now:
                cuts.add(t)
    ordered = sorted(cuts)
    segments = []
    for start, end in zip(ordered, [*ordered[1:], None]):
        merged = merge(active, start)
        if not merged.intent_ids:
            continue  # nothing active: no row, the runner falls back to the default objective
        objective = RLObjective(**{**merged.objective.model_dump(), "valid_from": start, "valid_until": end})
        segments.append((start, end, objective, merged.intent_ids))
    return segments


def write_timetable(
    conn: sqlite3.Connection, active: list[ActiveIntent], state: VerifierState, confirmed_by: str, reason: str
) -> Decision:
    """Verify the whole timetable (every start instant), then write it. Called inside a transaction."""
    decision = reverify_after_removal(state.model_copy(update={"active": tuple(active)}))
    segments = [
        a1_policy.Segment(objective=obj, intent_ids=ids, verifier_decision=decision_dict(decision),
                          confirmed_by=confirmed_by, valid_from=start, expires_at=end)
        for start, end, obj, ids in build_timetable(active, state.now_s)
    ]
    schedule_id = a1_policy.write_schedule(conn, segments, reason, state.now_s)
    log_event("agent", "timetable_written", ts_backend=state.now_s,
              data={"schedule_id": schedule_id, "reason": reason, "verifier": decision_dict(decision),
                    "segments": [{"valid_from": s.valid_from, "expires_at": s.expires_at,
                                  "intent_ids": list(s.intent_ids)} for s in segments]})
    return decision


def apply_intent(conn: sqlite3.Connection, intent_id: str, now_wall: float, confirmed_by: str) -> IntentRecord:
    """Re-verify, then apply an intent: new timetable, intent state 'applied'.

    Allowed from 'awaiting_confirmation' (console, before the deadline) and from 'interpreted'
    (evaluation mode, confirmed_by='auto-eval'). If the re-verification fails, the intent is
    rejected and a1_policy is not touched.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        record = intents.get(conn, intent_id)
        if record is None:
            raise ApplyError(f"no intent {intent_id}")
        allowed = "interpreted" if confirmed_by == "auto-eval" else "awaiting_confirmation"
        if record.state != allowed:
            raise ApplyError(f"{intent_id} is {record.state}, not {allowed}")
        if record.confirm_deadline is not None and now_wall >= record.confirm_deadline:
            late_by_s = now_wall - record.confirm_deadline
            record = intents.set_state(conn, intent_id, "expired", now_wall,
                                       reason="not confirmed before the deadline", confirm_deadline=None)
            conn.execute("COMMIT")
            # Distinct from a lazy expiry found by a read: here someone tried to confirm too late.
            log_event("console" if confirmed_by == "console" else "agent", "confirm_after_deadline", intent_id,
                      data={"late_by_s": late_by_s})
            log_event("agent", "expired", intent_id)
            return record

        state = verifier_state(conn)
        decision = verify(record.objective, state, new_intent_id=intent_id)
        log_event("verifier", "reverified", intent_id, state.now_s, decision_dict(decision))
        if not decision.accepted:
            record = intents.set_state(conn, intent_id, "rejected", now_wall, verifier=decision_dict(decision),
                                       reason=f"re-verification at confirmation: {decision.reason}",
                                       confirm_deadline=None)
            conn.execute("COMMIT")
            return record

        new = ActiveIntent(intent_id=intent_id, issued_at=state.now_s, objective=record.objective)
        write_timetable(conn, [*state.active, new], state, confirmed_by, f"apply {intent_id}")
        record = intents.set_state(conn, intent_id, "applied", now_wall, confirmed_by=confirmed_by,
                                   applied_backend=state.now_s, confirm_deadline=None,
                                   reason=decision.reason)
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    log_event("agent", "applied", intent_id, state.now_s, {"confirmed_by": confirmed_by})
    return record


def cancel_intent(conn: sqlite3.Connection, intent_id: str, now_wall: float) -> IntentRecord:
    """Withdraw an applied intent (console). The timetable is rewritten without it; a cancel
    cannot be refused, the verifier's re-check is recorded with the new timetable."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        record = intents.get(conn, intent_id)
        if record is None or record.state != "applied":
            raise ApplyError(f"{intent_id} is not an applied intent")
        remaining = [i for i in applied_intents(conn) if i.intent_id != intent_id]
        state = verifier_state(conn, remaining)
        write_timetable(conn, list(state.active), state, "console", f"cancel {intent_id}")
        record = intents.set_state(conn, intent_id, "cancelled", now_wall, reason="cancelled by the operator")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    log_event("console", "cancelled", intent_id, state.now_s)
    return record
