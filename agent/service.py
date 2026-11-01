"""The agent service (HTTP): the API the MCP gateway will call in M05a.

Three operations, matching the three gateway tools (hard invariant 1):
  POST /intents           submit_intent  store the intent ('received'); the agent worker does the rest
  GET  /intents/{id}      get_status     where an intent stands, and whether it is in force
  GET  /kpis              get_kpis       mean / p95 / last per slice over a window
plus GET /status (the same get_status without an id: the timetable and recent intents).

There is deliberately NO endpoint to confirm, refuse or cancel: that is the operator console's
job (console/), on its own path to the database, so nothing reachable through the gateway can
ever confirm an objective.

Run:  uv run uvicorn agent.service:app --port 8000   (and the worker: uv run python -m agent.worker)
Settings (environment): RIC_DB, RIC_EVENTS, RIC_MODE=demo|eval (shown in /status; used by the worker).
"""

import os
import time
from collections.abc import Iterator

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from agent import intents as machine
from agent.apply import ApplyError
from store import db, kpi

app = FastAPI(title="Agentic RIC: agent service", version="0.2 (M04b, LangGraph worker)")


def connection() -> Iterator:
    """One connection per request: short-lived, so nothing stays open between calls."""
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


class IntentIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)  # the operator's words, verbatim


@app.post("/intents", status_code=201)
def submit_intent(body: IntentIn, conn=Depends(connection)) -> dict:
    try:
        record = machine.submit(conn, body.text, time.time())
    except ApplyError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    return machine.status(conn, record.intent_id, time.time())


@app.get("/intents/{intent_id}")
def get_status(intent_id: str, conn=Depends(connection)) -> dict:
    result = machine.status(conn, intent_id, time.time())
    if result is None:
        raise HTTPException(status_code=404, detail=f"no intent {intent_id}")
    return result


@app.get("/kpis")
def get_kpis(
    slice: str | None = Query(default=None, pattern="^(embb|urllc|be)$"),
    window_s: float = Query(default=10.0, gt=0.0, le=3600.0),
    conn=Depends(connection),
) -> dict:
    return kpi.get_kpis(conn, slice, window_s)


@app.get("/status")
def get_overview(conn=Depends(connection)) -> dict:
    found = kpi.read_backend_info(conn)
    result = machine.overview(conn, time.time())
    result["backend"] = None if found is None else {**found[0].__dict__, "model_version": found[1]}
    result["mode"] = os.environ.get("RIC_MODE", "demo")
    return result
