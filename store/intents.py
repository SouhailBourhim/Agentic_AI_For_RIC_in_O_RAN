"""The intent records: what the operator asked, and where each request stands.

Storage only. The rules for moving between states live in agent/intents.py; this module just
reads and writes rows. Each state change records its wall-clock time in state_times, so the
whole history of an intent is in its row (and in the event log).
"""

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from verifier.objective import RLObjective

STATES = (
    "received",  # text stored, not parsed yet
    "processing",  # taken by the agent worker (M04b): its LangGraph loop is running
    "interpreted",  # parsed into an objective, not verified yet
    "rejected",  # by the parser, the verifier or the operator; `reason` says which and why
    "awaiting_confirmation",  # accepted by the verifier, waiting for the console (demo mode)
    "applied",  # written into the a1_policy timetable
    "expired",  # nobody confirmed it before confirm_deadline
    "cancelled",  # applied, then withdrawn by the operator from the console
)


@dataclass(frozen=True)
class IntentRecord:
    intent_id: str
    text: str
    state: str
    parsed: dict | None = None
    objective: RLObjective | None = None
    verifier: dict | None = None
    reason: str | None = None
    confirmed_by: str | None = None
    issued_backend: float | None = None
    applied_backend: float | None = None
    supersedes: str | None = None  # a revision: the intent it replaces once applied
    confirm_deadline: float | None = None
    state_times: dict[str, float] = field(default_factory=dict)


def _record(row: sqlite3.Row) -> IntentRecord:
    return IntentRecord(
        intent_id=row["intent_id"],
        text=row["text"],
        state=row["state"],
        parsed=json.loads(row["parsed"]) if row["parsed"] else None,
        objective=RLObjective.model_validate_json(row["objective"]) if row["objective"] else None,
        verifier=json.loads(row["verifier"]) if row["verifier"] else None,
        reason=row["reason"],
        confirmed_by=row["confirmed_by"],
        issued_backend=row["issued_backend"],
        applied_backend=row["applied_backend"],
        supersedes=row["supersedes"],
        confirm_deadline=row["confirm_deadline"],
        state_times=json.loads(row["state_times"]),
    )


def create(conn: sqlite3.Connection, text: str, now_wall: float, issued_backend: float | None,
           supersedes: str | None = None, state: str = "received") -> IntentRecord:
    """Store a new intent (state 'received' by default). Ids are i0001, i0002, ... in arrival order.

    A revision made by the agent is created directly as 'interpreted', with `supersedes` naming
    the intent it replaces, so the worker does not pick it up as a new operator request.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        count = conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0]
        intent_id = f"i{count + 1:04d}"
        conn.execute(
            "INSERT INTO intents (intent_id, text, state, issued_backend, supersedes, state_times)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (intent_id, text, state, issued_backend, supersedes, json.dumps({state: now_wall})),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return get(conn, intent_id)  # type: ignore[return-value]


def get(conn: sqlite3.Connection, intent_id: str) -> IntentRecord | None:
    row = conn.execute("SELECT * FROM intents WHERE intent_id = ?", (intent_id,)).fetchone()
    return _record(row) if row is not None else None


def in_states(conn: sqlite3.Connection, states: tuple[str, ...]) -> list[IntentRecord]:
    """Intents currently in any of the given states, oldest first."""
    marks = ",".join("?" * len(states))
    rows = conn.execute(f"SELECT * FROM intents WHERE state IN ({marks}) ORDER BY intent_id", states)
    return [_record(r) for r in rows]


def recent(conn: sqlite3.Connection, limit: int = 20) -> list[IntentRecord]:
    rows = conn.execute("SELECT * FROM intents ORDER BY intent_id DESC LIMIT ?", (limit,))
    return [_record(r) for r in rows]


def claim(conn: sqlite3.Connection, intent_id: str, from_state: str, to_state: str, now_wall: float) -> bool:
    """Move an intent from one state to another only if it is still in from_state; True if this
    call did it. One UPDATE, so two workers can never both take the same intent."""
    record = get(conn, intent_id)
    if record is None:
        return False
    times = json.dumps({**record.state_times, to_state: now_wall})
    cursor = conn.execute("UPDATE intents SET state = ?, state_times = ? WHERE intent_id = ? AND state = ?",
                          (to_state, times, intent_id, from_state))
    return cursor.rowcount == 1


def set_state(conn: sqlite3.Connection, intent_id: str, state: str, now_wall: float, **fields: Any) -> IntentRecord:
    """Move an intent to `state`, record the time, and update the given columns.

    fields may be: parsed (dict), objective (RLObjective), verifier (dict), reason, confirmed_by,
    confirm_deadline, applied_backend. Objects are stored as JSON.
    """
    if state not in STATES:
        raise ValueError(f"unknown intent state {state!r}")
    current = get(conn, intent_id)
    if current is None:
        raise KeyError(intent_id)
    columns: dict[str, Any] = {"state": state, "state_times": json.dumps({**current.state_times, state: now_wall})}
    for name, value in fields.items():
        if name in ("parsed", "verifier"):
            value = json.dumps(value) if value is not None else None
        elif name == "objective":
            value = value.model_dump_json() if value is not None else None
        elif name not in ("reason", "confirmed_by", "confirm_deadline", "applied_backend"):
            raise ValueError(f"unknown intent field {name!r}")
        columns[name] = value
    assignments = ", ".join(f"{name} = ?" for name in columns)
    conn.execute(f"UPDATE intents SET {assignments} WHERE intent_id = ?", (*columns.values(), intent_id))
    return get(conn, intent_id)  # type: ignore[return-value]
