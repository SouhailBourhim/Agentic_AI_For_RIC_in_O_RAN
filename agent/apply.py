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
from verifier.objective import DEFAULT_OBJECTIVE, SLICES, RLObjective
from verifier.verify import (
    DEFAULT_CONFIG,
    Decision,
    VerifierState,
    reverify_after_removal,
    verify,
    verify_shares,
    with_fair_shares,
)


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


def build_timetable(active: list[ActiveIntent], state: VerifierState) -> list[tuple[float, float | None, RLObjective, tuple[str, ...]]]:
    """(valid_from, expires_at, objective that runs, intent_ids) for every segment from now on.

    Cut points: now, and every start or end of an intent that is still ahead. Between two cut
    points the set of active intents does not change, so neither does the merge. Each segment is the
    merge plus the implied targets of the slices with none (verifier with_fair_shares).
    """
    now = state.now_s
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
        runs = with_fair_shares(merged, state, DEFAULT_CONFIG)
        objective = RLObjective(**{**runs.model_dump(), "valid_from": start, "valid_until": end})
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
        for start, end, obj, ids in build_timetable(active, state)
    ]
    schedule_id = a1_policy.write_schedule(conn, segments, reason, state.now_s)
    log_event("agent", "timetable_written", ts_backend=state.now_s,
              data={"schedule_id": schedule_id, "reason": reason, "verifier": decision_dict(decision),
                    "segments": [{"valid_from": s.valid_from, "expires_at": s.expires_at,
                                  "intent_ids": list(s.intent_ids)} for s in segments]})
    return decision


def starting_at(objective: RLObjective, now_s: float) -> RLObjective:
    """The same objective with its window moved to start at now_s, same length (never earlier)."""
    shift = now_s - objective.valid_from
    if shift <= 0:
        return objective
    until = None if objective.valid_until is None else objective.valid_until + shift
    return objective.model_copy(update={"valid_from": now_s, "valid_until": until})


def apply_intent(conn: sqlite3.Connection, intent_id: str, now_wall: float, confirmed_by: str) -> IntentRecord:
    """Re-verify, then apply an intent: new timetable, intent state 'applied'.

    Allowed from 'awaiting_confirmation' (console, before the deadline) and from 'interpreted'
    (evaluation mode, confirmed_by='auto-eval'). If the re-verification fails, the intent is
    rejected and a1_policy is not touched.

    A revision (record.supersedes set, M04b) is verified as replacing its parent (conflicts
    without it, step from what runs now), and the parent is cancelled in the same transaction:
    at no moment are both, or neither, in the timetable.

    The objective's window starts at the apply, not at the proposal: the time spent waiting for
    the console is not taken out of the duration the operator asked for (seen in the M04b live
    demo: a 5-minute request confirmed after 105 s ran for 195 s). The shifted objective is the
    one re-verified and stored.
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
        objective = starting_at(record.objective, state.now_s)
        decision = verify(objective, state, new_intent_id=intent_id, replacing=record.supersedes)
        log_event("verifier", "reverified", intent_id, state.now_s, decision_dict(decision))
        if not decision.accepted:
            record = intents.set_state(conn, intent_id, "rejected", now_wall, verifier=decision_dict(decision),
                                       reason=f"re-verification at confirmation: {decision.reason}",
                                       confirm_deadline=None)
            conn.execute("COMMIT")
            return record

        new = ActiveIntent(intent_id=intent_id, issued_at=state.now_s, objective=objective)
        staying = [i for i in state.active if i.intent_id != record.supersedes]  # a revision replaces its parent
        write_timetable(conn, [*staying, new], state, confirmed_by, f"apply {intent_id}")
        if record.supersedes is not None:
            intents.set_state(conn, record.supersedes, "cancelled", now_wall,
                              reason=f"superseded by its revision {intent_id}")
        record = intents.set_state(conn, intent_id, "applied", now_wall, confirmed_by=confirmed_by,
                                   applied_backend=state.now_s, confirm_deadline=None,
                                   reason=decision.reason, objective=objective)
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


def apply_shares(conn: sqlite3.Connection, intent_id: str, shares: list[float], valid_s: float,
                 now_wall: float) -> Decision:
    """The "LLM only" baseline (M08a): write a PRB split chosen by the LLM itself, for valid_s.

    Only the llm_only mode's set_slice_share tool calls this (hard invariant 4). The split goes
    through verify_shares (floor, sum, step from the split applied now) and becomes the only
    segment of a new timetable, in direct_share mode: the RL runner applies it as given (still
    with the floor) and stays the only caller of apply_prb_split. Evaluation mode only.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        record = intents.get(conn, intent_id)
        if record is None or record.state not in ("processing", "applied"):
            raise ApplyError(f"{intent_id} cannot take a direct PRB split in state "
                             f"{record.state if record else 'missing'}")
        now = backend_now(conn)
        rows = conn.execute("SELECT slice, prb_share_alloc FROM kpi WHERE ts = ?", (now,)).fetchall()
        applied_now = dict(rows)
        previous = [applied_now[s] for s in SLICES] if len(applied_now) == len(SLICES) else None
        decision = verify_shares(shares, previous)
        log_event("verifier", "verified_shares", intent_id, now, {**decision_dict(decision), "shares": shares})
        if not decision.accepted:
            conn.execute("COMMIT")
            return decision
        holder = RLObjective(weights=DEFAULT_OBJECTIVE.weights, valid_from=now, valid_until=now + valid_s)
        segment = a1_policy.Segment(objective=holder, intent_ids=(intent_id,), verifier_decision=decision_dict(decision),
                                    confirmed_by="auto-eval", valid_from=now, expires_at=now + valid_s,
                                    mode="direct_share", shares=tuple(float(x) for x in shares))
        a1_policy.write_schedule(conn, [segment], f"shares {intent_id}", now)
        if record.state != "applied":
            intents.set_state(conn, intent_id, "applied", now_wall, confirmed_by="auto-eval", applied_backend=now,
                              reason="LLM-only baseline: the LLM sets the PRB split")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    log_event("agent", "shares_applied", intent_id, now, {"shares": shares, "valid_s": valid_s})
    return decision
