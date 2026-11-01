"""a1_policy: the channel from the agent to the RL xApp. AN EMULATION OF O-RAN A1.

In O-RAN, the Non-RT RIC sends policies to the Near-RT RIC over the A1 interface. FlexRIC is a
Near-RT RIC only: there is no Non-RT RIC and no A1 in our stack. We emulate the channel with
two SQLite tables, explicitly named after what they stand for (CLAUDE.md, "Note on A1 policy").

The content is a **timetable**. Each time the set of applied intents changes (an intent is
applied or cancelled), the agent's apply step (agent/apply.py, the only writer) computes the
merged objective for every time segment until the last intent ends, has each one verified, and
writes them all at once as a new schedule. The RL runner reads the segment of the newest
schedule that covers "now". So an intent expiring is just the clock moving into the next
segment: no process has to wake up, and fast simulated runs see no gap.

Append-only: old schedules stay in the table as the record of what was in force when.
"""

import json
import sqlite3
import time
from dataclasses import dataclass

from verifier.objective import RLObjective


@dataclass(frozen=True)
class Segment:
    """One row to write: the objective in force from valid_from to expires_at."""

    objective: RLObjective
    intent_ids: tuple[str, ...]
    verifier_decision: dict  # {"accepted": ..., "check": ..., "reason": ...}
    confirmed_by: str  # 'console' | 'auto-eval'
    valid_from: float
    expires_at: float | None
    mode: str = "rl"  # 'rl' | 'direct_share' (LLM-only baseline, M04b)
    shares: tuple[float, ...] | None = None  # only in direct_share mode


@dataclass(frozen=True)
class PolicyRow:
    """One row as read back."""

    policy_id: int
    schedule_id: int
    objective: RLObjective
    mode: str
    shares: tuple[float, ...] | None
    intent_ids: tuple[str, ...]
    valid_from: float
    expires_at: float | None


def write_schedule(conn: sqlite3.Connection, segments: list[Segment], reason: str, now_backend: float) -> int:
    """Write a new timetable in one transaction; it replaces the previous one. Returns its id.

    An empty list is a valid timetable: nothing applied, the runner uses the default objective.
    If the caller already opened a transaction (agent/apply.py does, so that reading the active
    intents and writing the timetable happen together), the rows join it.
    """
    own_transaction = not conn.in_transaction
    if own_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        cursor = conn.execute(
            "INSERT INTO a1_schedule (created_wall, created_backend, reason) VALUES (?, ?, ?)",
            (time.time(), now_backend, reason),
        )
        schedule_id = cursor.lastrowid
        conn.executemany(
            "INSERT INTO a1_policy (schedule_id, objective, mode, shares, intent_ids, verifier_decision,"
            " confirmed_by, issued_at, valid_from, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (schedule_id, s.objective.model_dump_json(), s.mode,
                 json.dumps(list(s.shares)) if s.shares is not None else None,
                 json.dumps(list(s.intent_ids)), json.dumps(s.verifier_decision), s.confirmed_by,
                 now_backend, s.valid_from, s.expires_at)
                for s in segments
            ],
        )
        if own_transaction:
            conn.execute("COMMIT")
    except BaseException:
        if own_transaction:
            conn.execute("ROLLBACK")
        raise
    return schedule_id


def _row(row: sqlite3.Row) -> PolicyRow:
    return PolicyRow(
        policy_id=row["policy_id"],
        schedule_id=row["schedule_id"],
        objective=RLObjective.model_validate_json(row["objective"]),
        mode=row["mode"],
        shares=tuple(json.loads(row["shares"])) if row["shares"] else None,
        intent_ids=tuple(json.loads(row["intent_ids"])),
        valid_from=row["valid_from"],
        expires_at=row["expires_at"],
    )


def latest_schedule_id(conn: sqlite3.Connection) -> int | None:
    return conn.execute("SELECT MAX(schedule_id) FROM a1_schedule").fetchone()[0]


def current_schedule(conn: sqlite3.Connection) -> list[PolicyRow]:
    """All segments of the newest timetable, in time order."""
    rows = conn.execute(
        "SELECT * FROM a1_policy WHERE schedule_id = (SELECT MAX(schedule_id) FROM a1_schedule)"
        " ORDER BY valid_from"
    )
    return [_row(r) for r in rows]


def in_force(conn: sqlite3.Connection, now: float) -> PolicyRow | None:
    """The segment of the newest timetable with valid_from <= now < expires_at, or None.

    None means "use the default objective". This is the only query the RL runner makes.
    """
    row = conn.execute(
        "SELECT * FROM a1_policy WHERE schedule_id = (SELECT MAX(schedule_id) FROM a1_schedule)"
        " AND valid_from <= ? AND (expires_at IS NULL OR ? < expires_at)"
        " ORDER BY valid_from DESC LIMIT 1",
        (now, now),
    ).fetchone()
    return _row(row) if row is not None else None
