"""Opening var/ric.db and creating its tables.

SQLite in WAL mode (write-ahead log): readers never wait for a writer, and one writer at a
time. That is enough for a handful of local processes, needs no server, and the whole state
of a run is one file, readable with Python's sqlite3 module or any SQLite browser.
"""

import os
import sqlite3
from pathlib import Path

DEFAULT_DB = Path("var/ric.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS kpi (           -- one row per slice per tick (docs/PLAN.md section 3)
    ts REAL NOT NULL,                      -- backend clock, s
    source TEXT NOT NULL,                  -- 'sim' | 'flexric'
    cell_id TEXT NOT NULL,
    slice TEXT NOT NULL,                   -- 'embb' | 'urllc' | 'be'
    dl_thr_mbps REAL NOT NULL,
    prb_share_alloc REAL NOT NULL,         -- 0..1, the split applied during the tick
    prb_used REAL NOT NULL,                -- 0..1 of the slice's own PRBs
    n_ue INTEGER NOT NULL,
    latency_ms REAL,
    offered_load_mbps REAL,
    buffer_bytes REAL,
    policy_id INTEGER                      -- a1_policy row in force during the tick (NULL = default)
);
CREATE INDEX IF NOT EXISTS kpi_ts ON kpi (ts);

CREATE TABLE IF NOT EXISTS backend_info (  -- one row, written by the runner at startup
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cell_id TEXT NOT NULL,
    source TEXT NOT NULL,
    slices TEXT NOT NULL,                  -- JSON list
    n_prb INTEGER NOT NULL,
    capacity_mbps REAL NOT NULL,
    urllc_base_latency_ms REAL NOT NULL,
    model_version TEXT NOT NULL,           -- the served policy, e.g. 'slice-policy v3 (td3_...)'
    started_wall REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS a1_schedule (   -- one row per timetable; the newest one is in force
    schedule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_wall REAL NOT NULL,
    created_backend REAL NOT NULL,
    reason TEXT NOT NULL                   -- e.g. 'apply i0003', 'cancel i0002'
);

CREATE TABLE IF NOT EXISTS a1_policy (     -- the segments of each timetable (append-only)
    policy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    schedule_id INTEGER NOT NULL REFERENCES a1_schedule (schedule_id),
    objective TEXT NOT NULL,               -- RLObjective JSON (the merge of intent_ids)
    mode TEXT NOT NULL,                    -- 'rl' | 'direct_share'
    shares TEXT,                           -- JSON, only in direct_share mode
    intent_ids TEXT NOT NULL,              -- JSON list, every intent merged into this segment
    verifier_decision TEXT NOT NULL,       -- JSON {accepted, check, reason}
    confirmed_by TEXT NOT NULL,            -- 'console' | 'auto-eval'
    issued_at REAL NOT NULL,               -- backend clock when written
    valid_from REAL NOT NULL,              -- backend clock
    expires_at REAL                        -- backend clock, NULL = no end
);

CREATE TABLE IF NOT EXISTS kpm_latest (   -- testbed only (M07b): newest KPM values per slice
    slice TEXT PRIMARY KEY,                -- written by `xapps.kpm_monitor --for-runner`,
    report_wall REAL NOT NULL,             -- read by FlexRICBackend.read_kpis, which turns them
    dl_thr_mbps REAL NOT NULL,             -- into KPI records with the split it applied
    prb_cell REAL NOT NULL,                -- 0..1 of the WHOLE cell (sum of RRU.PrbTotDl / 100)
    latency_ms REAL,
    n_ue INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS traffic_target (  -- testbed only (M07b): the iperf3 rate each slice is
    slice TEXT PRIMARY KEY,                  -- being sent, written by the traffic script. A real
    offered_load_mbps REAL NOT NULL,         -- gNB does not know this (KPM does not measure it)
    set_wall REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS intents (
    intent_id TEXT PRIMARY KEY,            -- 'i0001', 'i0002', ...
    text TEXT NOT NULL,                    -- verbatim operator text
    state TEXT NOT NULL,
    parsed TEXT,                           -- JSON, what the parser extracted
    objective TEXT,                        -- RLObjective JSON proposed for this intent
    verifier TEXT,                         -- JSON {accepted, check, reason}
    reason TEXT,                           -- why it is in its state (human readable)
    confirmed_by TEXT,
    issued_backend REAL,                   -- backend clock at submission
    applied_backend REAL,                  -- backend clock when applied: decides 'newest' in the merge
    confirm_deadline REAL,                 -- wall clock, only while awaiting confirmation
    supersedes TEXT,                       -- a revision: the intent it replaces once applied (M04b)
    state_times TEXT NOT NULL              -- JSON {state: wall clock}
);
"""


def db_path() -> Path:
    """RIC_DB overrides the default location (tests use a temporary file)."""
    return Path(os.environ.get("RIC_DB", DEFAULT_DB))


def connect(path: Path | None = None, timeout_s: float = 5.0) -> sqlite3.Connection:
    """Open (and create if needed) the database.

    timeout_s is how long a write waits for another writer. The RL runner passes 0: it must
    never wait on another process (hard invariant 5), so a busy database is an error it handles.
    """
    path = path or db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=timeout_s, isolation_level=None)  # autocommit; explicit BEGIN
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    migrate(conn)
    return conn


# Columns added after a table was first created: (table, column, SQL type). CREATE TABLE IF NOT
# EXISTS does not add columns to an existing table, so they are added here, once.
ADDED_COLUMNS = [("intents", "supersedes", "TEXT")]


def migrate(conn: sqlite3.Connection) -> None:
    for table, column, sql_type in ADDED_COLUMNS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
