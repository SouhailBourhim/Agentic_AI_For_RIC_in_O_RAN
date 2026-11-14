"""The agent worker: takes operator intents from ric.db and runs the LangGraph loop on each.

The HTTP service only stores intents ('received'). This process picks the oldest one, claims it
('processing', one atomic UPDATE so it is never taken twice) and runs agent/graph.py to the end:
parse, plan, verify, confirm (demo) or apply (eval), observe, reflect. One intent at a time:
only one LLM fits in the GPU, so while an intent waits for the console or observes, the next
one waits too.

  uv run python -m agent.worker
Environment: RIC_MODE=demo|eval, RIC_CONTROLLER=proposed|llm_only, RIC_AGENT=llm|rules,
RIC_LLM_PROVIDER (ollama | nim, M09i), RIC_LLM_MODEL (Ollama tag or NIM model id), RIC_OBSERVE_S (20), RIC_CONFIRM_TIMEOUT_S (120), RIC_DB, RIC_EVENTS.
M09e, the fairer LLM only on the testbed: RIC_LLM_ONLY_PROMPT=v1|v2 and RIC_LLM_ONLY_JOINS=1 (as eval.harness sets them).
"""

import os
import time

from agent.graph import run_intent
from agent.llm import LLMError, client_from_env
from agent.tools import AgentContext, Settings
from store import db, intents
from store.events import log_event
from store.intents import IntentRecord


def settings_from_env() -> Settings:
    return Settings(
        mode=os.environ.get("RIC_MODE", "demo"),
        controller=os.environ.get("RIC_CONTROLLER", "proposed"),
        confirm_timeout_s=float(os.environ.get("RIC_CONFIRM_TIMEOUT_S", "120")),
        observe_s=float(os.environ.get("RIC_OBSERVE_S", "20")),
        llm_only_prompt=os.environ.get("RIC_LLM_ONLY_PROMPT", "v1"),
        llm_only_joins=os.environ.get("RIC_LLM_ONLY_JOINS", "0") == "1",
    )


def next_intent(conn) -> str | None:
    """Claim the oldest received intent, or None if there is nothing to do."""
    for record in intents.in_states(conn, ("received",)):
        if intents.claim(conn, record.intent_id, "received", "processing", time.time()):
            return record.intent_id
    return None


def process_one(ctx: AgentContext) -> str | None:
    """Claim and run one intent; returns its id. Any failure ends that intent, not the worker."""
    intent_id = next_intent(ctx.conn)
    if intent_id is None:
        return None
    ctx.trace.clear()
    try:
        run_intent(ctx, intent_id)
    except (LLMError, TimeoutError, ValueError) as err:
        record = intents.get(ctx.conn, intent_id)
        if record.state in ("processing", "interpreted"):
            intents.set_state(ctx.conn, intent_id, "rejected", time.time(), reason=f"agent error: {err}")
        log_event("agent", "agent_error", intent_id, data={"error": str(err)})
        print(f"{intent_id}: agent error: {err}")
    return intent_id


def submit_and_run(ctx: AgentContext, text: str) -> IntentRecord:
    """Submit an intent and run the agent on it in this process (tests, figures, evaluation).

    The same steps as the HTTP service + the worker, without the processes in between.
    """
    from agent.intents import submit  # here: the worker process itself never submits

    record = submit(ctx.conn, text, ctx.now_wall())
    intents.claim(ctx.conn, record.intent_id, "received", "processing", ctx.now_wall())
    ctx.trace.clear()
    run_intent(ctx, record.intent_id)
    return intents.get(ctx.conn, record.intent_id)


def main() -> None:
    settings = settings_from_env()
    llm = None if os.environ.get("RIC_AGENT", "llm") == "rules" else client_from_env()
    ctx = AgentContext(conn=db.connect(), settings=settings, llm=llm)
    print(f"agent worker: {settings.controller} controller, {settings.mode} mode, "
          f"{'rules' if llm is None else 'LLM ' + llm.model}, db {db.db_path()}")
    try:
        while True:
            handled = process_one(ctx)
            if handled is None:
                time.sleep(0.5)
            else:
                record = intents.get(ctx.conn, handled)
                print(f"{handled}: {record.state} ({record.reason})")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
