"""The decision log: one JSON object per line in var/events.jsonl, the same format for every process.

    {"ts_wall": 1730000000.12, "ts_backend": 412.0, "source": "agent", "event": "verified",
     "intent_id": "i0003", "data": {...}}

source is 'runner' | 'agent' | 'verifier' | 'console' | 'gateway' (M05a: one 'tool_call' per MCP
call). M08b computes its interface metrics from this file, so events are never edited or removed, only appended.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_EVENTS = Path("var/events.jsonl")


def events_path() -> Path:
    """RIC_EVENTS overrides the default location (tests use a temporary file)."""
    return Path(os.environ.get("RIC_EVENTS", DEFAULT_EVENTS))


def log_event(
    source: str,
    event: str,
    intent_id: str | None = None,
    ts_backend: float | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    """Append one event. A single short write per line, so lines from different processes
    do not interleave."""
    record = {"ts_wall": time.time(), "ts_backend": ts_backend, "source": source, "event": event,
              "intent_id": intent_id, "data": data or {}}
    path = events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_events(path: Path | None = None) -> list[dict[str, Any]]:
    """All events, oldest first (for tests and the evaluation)."""
    path = path or events_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
