"""M04a: rule-based parser, intent state machine, apply step, console paths and HTTP service.

Hard invariant 6: in demo mode an accepted objective waits for the console and expires after
the timeout; evaluation mode skips confirmation. Also: nothing rejected ever reaches a1_policy,
and the agent API offers no way to confirm.
"""

import pytest
from fastapi.testclient import TestClient

from agent import intents as machine
from agent.apply import ApplyError, apply_intent, cancel_intent
from agent.parser import parse_intent
from agent.service import app
from conftest import start_cell
from store import a1_policy
from store.events import read_events

DEMO = machine.Settings(mode="demo", confirm_timeout_s=120.0)
EVAL = machine.Settings(mode="eval")
URLLC_10 = "garantir 10 Mb/s au slice URLLC pendant 10 minutes"

# ------------------------------------------------------------------ parser


@pytest.mark.parametrize(
    ("text", "weights", "targets", "bound", "duration"),
    [
        (URLLC_10, [0.2, 0.6, 0.2], [0, 10, 0], None, 600),
        ("prioritise eMBB for 30 min", [0.6, 0.2, 0.2], [0, 0, 0], None, 1800),
        ("prioriser le best effort pendant 90 s", [0.2, 0.2, 0.6], [0, 0, 0], None, 90),
        ("Garantir 15,5 Mbit/s à l'eMBB", [0.6, 0.2, 0.2], [15.5, 0, 0], None, 600),
        ("URLLC latency below 8 ms for 1 hour", [0.4, 0.3, 0.3], [0, 0, 0], 8.0, 3600),
        ("protéger URLLC, 10 Mb/s, latence 10 ms", [0.2, 0.6, 0.2], [0, 10, 0], 10.0, 600),
        ("guarantee 20 Mbps to embb and 10 mbps to urllc for 2 hours", [0.4, 0.3, 0.3], [20, 10, 0], None, 7200),
        ("latence URLLC < 5 ms et 20 Mb/s pour le best effort pendant 1 h", [0.2, 0.2, 0.6], [0, 0, 20], 5.0, 3600),
        ("garantir 60 Mb/s au slice URLLC", [0.2, 0.6, 0.2], [0, 60, 0], None, 600),  # impossible: the verifier's job
    ],
)
def test_parser_extracts(text, weights, targets, bound, duration) -> None:
    result = parse_intent(text, now_s=100.0)
    obj = result.objective
    assert result.problem is None and obj is not None
    assert obj.weights.as_list() == pytest.approx(weights)
    assert obj.min_thr_mbps.as_list() == pytest.approx(targets)
    assert obj.max_latency_ms == bound
    assert (obj.valid_from, obj.valid_until) == (100.0, 100.0 + duration)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("make the network better", "not understood"),
        ("give 10 Mb/s", "without a clear slice"),
        ("latence eMBB < 5 ms", "only for URLLC"),
    ],
)
def test_parser_says_when_it_does_not_understand(text, problem) -> None:
    result = parse_intent(text, now_s=0.0)
    assert result.objective is None and problem in result.problem


# ------------------------------------------------------------------ state machine (invariant 6)


def policy_rows(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM a1_policy").fetchone()[0]


def test_submit_without_runner_is_refused(ric_db) -> None:
    with pytest.raises(ApplyError, match="rl.runner"):
        machine.submit(ric_db, URLLC_10, DEMO, now_wall=0.0)


def test_demo_mode_waits_for_console_then_applies(ric_db, tiny_policy_file) -> None:
    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(5)
    record = machine.submit(ric_db, URLLC_10, DEMO, now_wall=1000.0)
    assert record.state == "awaiting_confirmation" and record.confirm_deadline == 1120.0
    assert policy_rows(ric_db) == 0  # submit_intent never applies anything (invariant 1)
    record = apply_intent(ric_db, record.intent_id, now_wall=1060.0, confirmed_by="console")
    assert record.state == "applied" and record.confirmed_by == "console"
    runner.run(2)
    assert runner.row is not None and runner.row.intent_ids == ("i0001",)


def test_pending_intent_expires_and_cannot_be_confirmed(ric_db, tiny_policy_file) -> None:
    start_cell(ric_db, tiny_policy_file).run(5)
    record = machine.submit(ric_db, URLLC_10, DEMO, now_wall=1000.0)
    assert machine.status(ric_db, record.intent_id, now_wall=1119.0)["state"] == "awaiting_confirmation"
    assert machine.status(ric_db, record.intent_id, now_wall=1120.0)["state"] == "expired"
    with pytest.raises(ApplyError, match="expired"):
        apply_intent(ric_db, record.intent_id, now_wall=1121.0, confirmed_by="console")
    assert policy_rows(ric_db) == 0


def test_confirming_after_deadline_expires_it(ric_db, tiny_policy_file) -> None:
    """Even if nobody read the status in between, the deadline is checked at confirmation."""
    start_cell(ric_db, tiny_policy_file).run(5)
    record = machine.submit(ric_db, URLLC_10, DEMO, now_wall=1000.0)
    assert apply_intent(ric_db, record.intent_id, now_wall=1200.0, confirmed_by="console").state == "expired"
    assert policy_rows(ric_db) == 0
    late = [e for e in read_events() if e["event"] == "confirm_after_deadline"]
    assert len(late) == 1 and late[0]["source"] == "console" and late[0]["data"]["late_by_s"] == 80.0


def test_eval_mode_skips_confirmation(ric_db, tiny_policy_file) -> None:
    start_cell(ric_db, tiny_policy_file).run(5)
    record = machine.submit(ric_db, URLLC_10, EVAL, now_wall=0.0)
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
@pytest.mark.parametrize("settings", [DEMO, EVAL], ids=["demo", "eval"])
def test_rejected_intent_never_reaches_a1_policy(ric_db, tiny_policy_file, text, settings) -> None:
    start_cell(ric_db, tiny_policy_file).run(5)
    record = machine.submit(ric_db, text, settings, now_wall=0.0)
    assert record.state == "rejected" and record.reason
    assert policy_rows(ric_db) == 0
    with pytest.raises(ApplyError):
        apply_intent(ric_db, record.intent_id, now_wall=1.0, confirmed_by="console")
    assert policy_rows(ric_db) == 0


def test_refused_intent_never_reaches_a1_policy(ric_db, tiny_policy_file) -> None:
    start_cell(ric_db, tiny_policy_file).run(5)
    record = machine.submit(ric_db, URLLC_10, DEMO, now_wall=0.0)
    assert machine.refuse(ric_db, record.intent_id, now_wall=10.0).reason == "refused by the operator"
    with pytest.raises(ApplyError):
        apply_intent(ric_db, record.intent_id, now_wall=11.0, confirmed_by="console")
    assert policy_rows(ric_db) == 0


def test_confirmation_reverifies_against_what_is_applied_now(ric_db, tiny_policy_file) -> None:
    """Both intents pass alone and wait. After the first is applied the second conflicts
    (BE 27 % + eMBB 29 % + URLLC 22 Mb/s within 5 ms 49 % > 100 %): confirming re-verifies and rejects."""
    start_cell(ric_db, tiny_policy_file).run(5)
    first = machine.submit(ric_db, "garantir 20 Mb/s au best effort pendant 10 min", DEMO, now_wall=0.0)
    second = machine.submit(ric_db, "garantir 22 Mb/s à l'eMBB et 22 Mb/s au slice URLLC avec latence 5 ms",
                            DEMO, now_wall=0.0)
    assert first.state == second.state == "awaiting_confirmation"
    apply_intent(ric_db, first.intent_id, now_wall=1.0, confirmed_by="console")
    rows = policy_rows(ric_db)
    record = apply_intent(ric_db, second.intent_id, now_wall=2.0, confirmed_by="console")
    assert record.state == "rejected" and "re-verification" in record.reason and "conflict" in record.verifier["check"]
    assert policy_rows(ric_db) == rows


# ------------------------------------------------------------------ timetable, cancel, status


def test_timetable_merges_and_status_shows_merged_with(ric_db, tiny_policy_file) -> None:
    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(10)  # backend clock 10 s
    a = machine.submit(ric_db, "garantir 10 Mb/s au slice URLLC pendant 2 minutes", EVAL, now_wall=0.0)
    runner.run(10)  # 20 s
    b = machine.submit(ric_db, "garantir 20 Mb/s au best effort pendant 1 min", EVAL, now_wall=0.0)
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
    machine.submit(ric_db, "garantir 10 Mb/s au slice URLLC pendant 2 minutes", EVAL, now_wall=0.0)
    machine.submit(ric_db, "garantir 20 Mb/s au best effort pendant 1 min", EVAL, now_wall=0.0)
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
    assert created.status_code == 201 and created.json()["state"] == "awaiting_confirmation"
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
