"""M04a, run through the M04b graph with the rule-based agent: parser, state machine, apply, console, HTTP.

Hard invariant 6: in demo mode an accepted objective waits for the console and expires after
the timeout; evaluation mode skips confirmation. Also: nothing rejected ever reaches a1_policy,
and the agent API offers no way to confirm.
"""

import pytest
from fastapi.testclient import TestClient

from agent import intents as machine
from agent.apply import ApplyError, apply_intent, cancel_intent
from agent.parser import parse_facts
from agent.service import app
from conftest import run_agent, start_cell
from store import a1_policy
from store.events import read_events

URLLC_10 = "garantir 10 Mb/s au slice URLLC pendant 10 minutes"

# ------------------------------------------------------------------ rule-based parser (facts only)


@pytest.mark.parametrize(
    ("text", "priority", "targets", "bound", "duration"),
    [
        (URLLC_10, None, [None, 10, None], None, 600),  # a guarantee alone is not a priority
        ("prioritise eMBB for 30 min", "embb", [None, None, None], None, 1800),
        ("prioriser le best effort pendant 90 s", "be", [None, None, None], None, 90),
        ("Garantir 15,5 Mbit/s à l'eMBB", None, [15.5, None, None], None, None),  # no duration stated
        ("URLLC latency below 8 ms for 1 hour", None, [None, None, None], 8.0, 3600),
        ("protéger URLLC, 10 Mb/s, latence 10 ms", "urllc", [None, 10, None], 10.0, None),
        ("guarantee 20 Mbps to embb and 10 mbps to urllc for 2 hours", None, [20, 10, None], None, 7200),
        ("latence URLLC < 5 ms et 20 Mb/s pour le best effort pendant 1 h", None, [None, None, 20], 5.0, 3600),
        ("garantir 60 Mb/s au slice URLLC", None, [None, 60, None], None, None),  # impossible: the verifier's job
    ],
)
def test_rule_parser_extracts_facts(text, priority, targets, bound, duration) -> None:
    facts = parse_facts(text)
    assert facts.understood and facts.clarification is None
    assert facts.priority == priority
    assert [facts.min_thr_mbps.embb, facts.min_thr_mbps.urllc, facts.min_thr_mbps.be] == targets
    assert facts.max_latency_ms == bound
    assert facts.duration_s == duration


@pytest.mark.parametrize(
    ("text", "question"),
    [
        ("make the network better", "what should change"),
        ("give 10 Mb/s", "which slice"),
        ("latence eMBB < 5 ms", "only for URLLC"),
    ],
)
def test_rule_parser_asks_when_it_does_not_understand(text, question) -> None:
    facts = parse_facts(text)
    assert not facts.understood and question in facts.clarification


# ------------------------------------------------------------------ state machine (invariant 6)


def policy_rows(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM a1_policy").fetchone()[0]


def test_submit_without_runner_is_refused(ric_db) -> None:
    with pytest.raises(ApplyError, match="rl.runner"):
        run_agent(ric_db, URLLC_10, 'demo', 0.0)


def test_demo_mode_waits_for_console_then_applies(ric_db, tiny_policy_file) -> None:
    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(5)
    record = run_agent(ric_db, URLLC_10, 'demo', 1000.0)
    assert record.state == "awaiting_confirmation" and record.confirm_deadline == 1120.0
    assert policy_rows(ric_db) == 0  # submit_intent never applies anything (invariant 1)
    record = apply_intent(ric_db, record.intent_id, now_wall=1060.0, confirmed_by="console")
    assert record.state == "applied" and record.confirmed_by == "console"
    runner.run(2)
    assert runner.row is not None and runner.row.intent_ids == ("i0001",)


def test_the_duration_starts_at_the_confirmation(ric_db, tiny_policy_file) -> None:
    """M04b live demo: a 5-minute intent confirmed later must still run 5 minutes."""
    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(5)
    record = run_agent(ric_db, "garantir 10 Mb/s au slice URLLC pendant 5 minutes", "demo", 1000.0)
    assert (record.objective.valid_from, record.objective.valid_until) == (5.0, 305.0)  # as proposed
    runner.run(40)  # the operator takes 40 backend seconds to confirm
    record = apply_intent(ric_db, record.intent_id, now_wall=1040.0, confirmed_by="console")
    assert (record.objective.valid_from, record.objective.valid_until) == (45.0, 345.0)
    segments = [(r.valid_from, r.expires_at) for r in a1_policy.current_schedule(ric_db)]
    assert segments == [(45.0, 345.0)]


def test_pending_intent_expires_and_cannot_be_confirmed(ric_db, tiny_policy_file) -> None:
    start_cell(ric_db, tiny_policy_file).run(5)
    record = run_agent(ric_db, URLLC_10, 'demo', 1000.0)
    assert machine.status(ric_db, record.intent_id, now_wall=1119.0)["state"] == "awaiting_confirmation"
    assert machine.status(ric_db, record.intent_id, now_wall=1120.0)["state"] == "expired"
    with pytest.raises(ApplyError, match="expired"):
        apply_intent(ric_db, record.intent_id, now_wall=1121.0, confirmed_by="console")
    assert policy_rows(ric_db) == 0


def test_confirming_after_deadline_expires_it(ric_db, tiny_policy_file) -> None:
    """Even if nobody read the status in between, the deadline is checked at confirmation."""
    start_cell(ric_db, tiny_policy_file).run(5)
    record = run_agent(ric_db, URLLC_10, 'demo', 1000.0)
    assert apply_intent(ric_db, record.intent_id, now_wall=1200.0, confirmed_by="console").state == "expired"
    assert policy_rows(ric_db) == 0
    late = [e for e in read_events() if e["event"] == "confirm_after_deadline"]
    assert len(late) == 1 and late[0]["source"] == "console" and late[0]["data"]["late_by_s"] == 80.0


def test_eval_mode_skips_confirmation(ric_db, tiny_policy_file) -> None:
    start_cell(ric_db, tiny_policy_file).run(5)
    record = run_agent(ric_db, URLLC_10, 'eval', 0.0)
    assert record.state == "applied" and record.confirmed_by == "auto-eval"
    assert policy_rows(ric_db) == 1


@pytest.mark.parametrize(
    "text",
    [
        "make the network better",  # parser
        "garantir 60 Mb/s au slice URLLC",  # verifier: training range
        "garantir 10 Mb/s au slice URLLC pendant 5 s",  # verifier: window too short
        "latence URLLC < 1 ms",  # verifier: latency physics
    ],
)
@pytest.mark.parametrize("mode", ["demo", "eval"])
def test_rejected_intent_never_reaches_a1_policy(ric_db, tiny_policy_file, text, mode) -> None:
    start_cell(ric_db, tiny_policy_file).run(5)
    record = run_agent(ric_db, text, mode, 0.0)
    assert record.state == "rejected" and record.reason
    assert policy_rows(ric_db) == 0
    with pytest.raises(ApplyError):
        apply_intent(ric_db, record.intent_id, now_wall=1.0, confirmed_by="console")
    assert policy_rows(ric_db) == 0


def test_refused_intent_never_reaches_a1_policy(ric_db, tiny_policy_file) -> None:
    start_cell(ric_db, tiny_policy_file).run(5)
    record = run_agent(ric_db, URLLC_10, 'demo', 0.0)
    assert machine.refuse(ric_db, record.intent_id, now_wall=10.0).reason == "refused by the operator"
    with pytest.raises(ApplyError):
        apply_intent(ric_db, record.intent_id, now_wall=11.0, confirmed_by="console")
    assert policy_rows(ric_db) == 0


def test_confirmation_reverifies_against_what_is_applied_now(ric_db, tiny_policy_file) -> None:
    """Both intents pass alone and wait. After the first is applied the second conflicts
    (BE 27 % + eMBB 29 % + URLLC 22 Mb/s within 5 ms 49 % > 100 %): confirming re-verifies and rejects."""
    start_cell(ric_db, tiny_policy_file).run(5)
    first = run_agent(ric_db, "garantir 20 Mb/s au best effort pendant 10 min", 'demo', 0.0)
    second = run_agent(ric_db, "garantir 22 Mb/s à l'eMBB et 22 Mb/s au slice URLLC avec latence 5 ms", "demo", 0.0)
    assert first.state == second.state == "awaiting_confirmation"
    apply_intent(ric_db, first.intent_id, now_wall=1.0, confirmed_by="console")
    rows = policy_rows(ric_db)
    record = apply_intent(ric_db, second.intent_id, now_wall=2.0, confirmed_by="console")
    assert record.state == "rejected" and "re-verification" in record.reason and "conflict" in record.verifier["check"]
    assert policy_rows(ric_db) == rows


# ------------------------------------------------------------------ timetable, cancel, status


def test_status_ignores_steps_of_a_previous_run_with_the_same_id(ric_db, tiny_policy_file) -> None:
    """The event log keeps every run; ids restart at i0001 when the runner empties ric.db."""
    import json
    import os
    import time

    old = {"ts_wall": time.time() - 3600, "ts_backend": 1.0, "source": "agent", "event": "agent_step",
           "intent_id": "i0001", "data": {"node": "parse", "from": "previous run"}}
    with open(os.environ["RIC_EVENTS"], "a", encoding="utf-8") as f:
        f.write(json.dumps(old) + "\n")
    start_cell(ric_db, tiny_policy_file).run(5)
    record = run_agent(ric_db, URLLC_10, "eval", time.time())
    steps = machine.status(ric_db, record.intent_id, now_wall=time.time())["agent_steps"]
    assert record.intent_id == "i0001" and steps and all(s.get("from") != "previous run" for s in steps)


def test_timetable_merges_and_status_shows_merged_with(ric_db, tiny_policy_file) -> None:
    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(10)  # backend clock 10 s
    a = run_agent(ric_db, "garantir 10 Mb/s au slice URLLC pendant 2 minutes", 'eval', 0.0)
    runner.run(10)  # 20 s
    b = run_agent(ric_db, "garantir 20 Mb/s au best effort pendant 1 min", 'eval', 0.0)
    segments = [(r.valid_from, r.expires_at, r.intent_ids) for r in a1_policy.current_schedule(ric_db)]
    assert segments == [(20.0, 80.0, ("i0001", "i0002")), (80.0, 130.0, ("i0001",))]
    merged = a1_policy.current_schedule(ric_db)[0].objective
    assert merged.min_thr_mbps.as_list() == [0.0, 10.0, 20.0]
    assert merged.weights.as_list() == pytest.approx([0.2, 0.2, 0.6])  # b, applied last
    status = machine.status(ric_db, a.intent_id, now_wall=0.0)
    assert status["in_effect"] and status["merged_with"] == [b.intent_id]


def test_cancel_rewrites_the_timetable(ric_db, tiny_policy_file) -> None:
    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(10)
    run_agent(ric_db, "garantir 10 Mb/s au slice URLLC pendant 2 minutes", 'eval', 0.0)
    run_agent(ric_db, "garantir 20 Mb/s au best effort pendant 1 min", 'eval', 0.0)
    assert cancel_intent(ric_db, "i0001", now_wall=1.0).state == "cancelled"
    assert [r.intent_ids for r in a1_policy.current_schedule(ric_db)] == [("i0002",)]
    with pytest.raises(ApplyError):
        cancel_intent(ric_db, "i0001", now_wall=2.0)  # already cancelled


# ------------------------------------------------------------------ HTTP service


def test_service_endpoints(ric_db, tiny_policy_file, monkeypatch) -> None:
    monkeypatch.setenv("RIC_MODE", "demo")
    client = TestClient(app)
    assert client.post("/intents", json={"text": URLLC_10}).status_code == 503  # no runner yet
    start_cell(ric_db, tiny_policy_file).run(15)
    created = client.post("/intents", json={"text": URLLC_10})
    # Hard invariant 1: submit_intent only creates the request; the worker does the rest.
    assert created.status_code == 201 and created.json()["state"] == "received"
    assert policy_rows(ric_db) == 0
    assert client.get("/intents/i0001").json()["intent_id"] == "i0001"
    assert client.get("/intents/i9999").status_code == 404
    kpis = client.get("/kpis", params={"slice": "urllc", "window_s": 5}).json()
    assert kpis["ts"] == 15.0 and set(kpis["slices"]) == {"urllc"}
    assert client.get("/kpis", params={"slice": "mmtc"}).status_code == 422
    overview = client.get("/status").json()
    assert overview["backend"]["model_version"] == "tiny test policy" and overview["mode"] == "demo"


def test_service_offers_no_way_to_confirm() -> None:
    """Only submit, get_status and get_kpis (+ the status overview). Confirmation is the console's.
    FastAPI's own read-only documentation pages (/docs, /openapi.json...) are left out."""
    docs = {app.openapi_url, app.docs_url, app.redoc_url, app.swagger_ui_oauth2_redirect_url}
    routes = {(route.path, method) for route in app.routes if hasattr(route, "methods") and route.path not in docs
              for method in route.methods if method in ("GET", "POST", "PUT", "PATCH", "DELETE")}
    assert routes == {("/intents", "POST"), ("/intents/{intent_id}", "GET"), ("/kpis", "GET"), ("/status", "GET")}
