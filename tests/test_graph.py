"""M04b: the LangGraph agent, driven by a scripted FakeLLM (no GPU, runs in CI).

Covers the loop of the proposal's figure 3, its limits, demo-mode confirmation, the revision
rule, hard invariant 4 (set_slice_share only in llm_only mode), the worker's claim and the
Ollama request format.
"""

import json
import sqlite3

import pytest

from agent import tools
from agent.apply import apply_intent
from agent.llm import FakeLLM, LLMError, LLMReply, OllamaClient, tool_call
from agent.tools import AgentContext, Settings, ToolNotAllowed
from agent.worker import next_intent, process_one, submit_and_run
from conftest import run_agent, start_cell
from store import a1_policy, intents
from store.events import read_events
from verifier.verify import verify_shares

TEXT = "garantir 10 Mb/s au slice URLLC pendant 10 minutes"


def facts(understood=True, priority=None, urllc=None, embb=None, be=None, lat=None, dur=600.0, question=None):
    return json.dumps({"understood": understood, "clarification": question, "priority": priority,
                       "min_thr_mbps": {"embb": embb, "urllc": urllc, "be": be},
                       "max_latency_ms": lat, "duration_s": dur})


def objective_call(urllc=10.0, embb=None, importance=None, lat=None, dur=600.0) -> LLMReply:
    return tool_call("set_rl_objective", importance=importance or {"embb": 2, "urllc": 8, "be": 2},
                     min_thr_mbps={"embb": embb, "urllc": urllc, "be": None}, max_latency_ms=lat, duration_s=dur)


def reflection(decision="done", reason="targets met") -> str:
    return json.dumps({"decision": decision, "reason": reason})


@pytest.fixture
def cell(ric_db, tiny_policy_file):
    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(10)
    return runner


def agent(ric_db, cell, script, mode="eval", **settings):
    llm = FakeLLM(script)
    options = {"observe_s": 5.0, "max_revisions": 1, **settings}
    return run_agent(ric_db, TEXT, mode, llm=llm, runner=cell, **options), llm


# ------------------------------------------------------------------ the loop


def test_happy_path_parse_plan_verify_apply_observe_reflect(ric_db, cell) -> None:
    record, llm = agent(ric_db, cell, [facts(urllc=10), objective_call(), reflection("done")])
    assert record.state == "applied" and record.confirmed_by == "auto-eval"
    nodes = [e["data"]["node"] for e in read_events() if e["event"] == "agent_step"]
    assert nodes == ["parse", "plan", "act", "observe", "reflect"]
    assert llm.requests[0]["schema"]["title"] == "ParsedIntent"  # structured output for the parse
    assert cell.backend.now() == 15.0  # 5 s observed on the backend clock
    weights = a1_policy.current_schedule(ric_db)[0].objective.weights.as_list()
    assert weights == pytest.approx([2 / 12, 8 / 12, 2 / 12])  # importance 2/8/2 -> weights summing to 1


def test_verifier_rejection_replans(ric_db, cell) -> None:
    too_much = objective_call(urllc=60.0)  # above the training range: rejected
    record, llm = agent(ric_db, cell, [facts(urllc=60), too_much, objective_call(urllc=20.0), reflection()])
    assert record.state == "applied"
    plan_prompt = llm.requests[2]["messages"][1]["content"]
    assert "VERIFIER REJECTED" in plan_prompt and "trained on" in plan_prompt  # the reason reaches the LLM


def test_three_rejections_reject_the_intent(ric_db, cell) -> None:
    script = [facts(urllc=60), objective_call(urllc=60.0), objective_call(urllc=50.0), objective_call(urllc=40.0)]
    record, _ = agent(ric_db, cell, script)
    assert record.state == "rejected" and "verifier rejected 3 proposals" in record.reason
    assert a1_policy.current_schedule(ric_db) == []


def test_repeating_a_rejected_proposal_ends_at_once(ric_db, cell) -> None:
    """Seen live with llama3.1: the same rejected objective proposed again. Not re-verified."""
    bad = objective_call(urllc=60.0)
    record, llm = agent(ric_db, cell, [facts(urllc=60), bad, bad])
    assert record.state == "rejected" and "repeated a proposal" in record.reason and "trained on" in record.reason
    assert len(llm.requests) == 3  # parse + 2 plans: no third plan
    assert sum(1 for e in read_events() if e["event"] == "agent_step" and e["data"].get("verifier")) == 1


def test_invalid_json_is_retried_then_rejected(ric_db, cell) -> None:
    record, _ = agent(ric_db, cell, ["not json", '{"understood": "maybe"}'])
    assert record.state == "rejected" and "no valid parse" in record.reason


def test_invalid_json_once_then_valid_goes_on(ric_db, cell) -> None:
    record, _ = agent(ric_db, cell, ["oops", facts(urllc=10), objective_call(), reflection()])
    assert record.state == "applied"


def test_ambiguous_intent_is_rejected_with_the_question(ric_db, cell) -> None:
    record, _ = agent(ric_db, cell, [facts(understood=False, question="Pour quel slice ?", dur=None)])
    assert record.state == "rejected" and record.reason == "needs clarification: Pour quel slice ?"


def test_one_revision_supersedes_the_applied_objective(ric_db, cell) -> None:
    script = [facts(urllc=10), objective_call(), reflection("revise", "URLLC missed its target"),
              objective_call(importance={"embb": 1, "urllc": 10, "be": 1}), reflection("revise", "still missed")]
    record, _ = agent(ric_db, cell, script)
    assert record.state == "cancelled" and "superseded by its revision i0002" in record.reason
    revision = intents.get(ric_db, "i0002")
    assert revision.state == "applied" and revision.supersedes == "i0001"
    assert [r.intent_ids for r in a1_policy.current_schedule(ric_db)] == [("i0002",)]  # never both
    last = [e["data"] for e in read_events() if e["event"] == "agent_step"][-1]
    assert last["decision"] == "done" and last["reason"] == "revision limit reached"  # no second revision


def test_step_limit(ric_db, cell) -> None:
    record, _ = agent(ric_db, cell, [facts(urllc=10)] + [tool_call("get_kpis")] * 3, max_steps=3)
    assert record.state == "rejected" and "step limit" in record.reason


def test_unknown_tool_is_not_executed(ric_db, cell) -> None:
    script = [facts(urllc=10), tool_call("set_slice_share", embb=0.1, urllc=0.8, be=0.1), objective_call(),
              reflection()]
    record, llm = agent(ric_db, cell, script)
    assert record.state == "applied"
    assert "does not exist here" in llm.requests[2]["messages"][1]["content"]
    assert all(r.mode == "rl" for r in a1_policy.current_schedule(ric_db))


# ------------------------------------------------------------------ demo mode: the console in the loop


def test_demo_mode_waits_for_the_console(ric_db, cell) -> None:
    clock = {"t": 1000.0}

    def operator_confirms(_seconds: float) -> None:  # the "sleep" while waiting: the operator acts
        apply_intent(ric_db, "i0001", clock["t"], confirmed_by="console")

    ctx = AgentContext(conn=ric_db, llm=FakeLLM([facts(urllc=10), objective_call(), reflection()]),
                       settings=Settings(mode="demo", observe_s=5.0), now_wall=lambda: clock["t"],
                       sleep=operator_confirms, wait_backend=lambda s: cell.run(int(s)))
    record = submit_and_run(ctx, TEXT)
    assert record.state == "applied" and record.confirmed_by == "console"
    assert [e["node"] for e in ctx.trace] == ["parse", "plan", "act", "confirm", "observe", "reflect"]


def test_demo_mode_without_answer_expires_and_stops(ric_db, cell) -> None:
    clock = {"t": 1000.0}

    def nobody_answers(seconds: float) -> None:
        clock["t"] += 60.0

    ctx = AgentContext(conn=ric_db, llm=FakeLLM([facts(urllc=10), objective_call()]),
                       settings=Settings(mode="demo", confirm_timeout_s=120.0), now_wall=lambda: clock["t"],
                       sleep=nobody_answers)
    record = submit_and_run(ctx, TEXT)
    assert record.state == "expired"
    assert a1_policy.current_schedule(ric_db) == []


# ------------------------------------------------------------------ hard invariant 4


def test_tool_lists_per_mode() -> None:
    assert "set_slice_share" not in tools.tools_for_mode("proposed")
    assert "set_rl_objective" not in tools.tools_for_mode("llm_only")
    sent = [d["function"]["name"] for d in tools.definitions("proposed")]
    assert sent == ["get_kpis", "wait_and_observe", "set_rl_objective", "finish"]


def test_plan_step_is_never_offered_set_slice_share(ric_db, cell) -> None:
    _, llm = agent(ric_db, cell, [facts(urllc=10), objective_call(), reflection()])
    offered = {d["function"]["name"] for r in llm.requests if r["tools"] for d in r["tools"]}
    assert "set_slice_share" not in offered


def test_set_slice_share_refuses_to_run_in_proposed_mode(ric_db, cell) -> None:
    ctx = AgentContext(conn=ric_db, llm=None, settings=Settings(mode="eval", controller="proposed"))
    with pytest.raises(ToolNotAllowed):
        tools.set_slice_share(ctx, "i0001", {"embb": 0.4, "urllc": 0.4, "be": 0.2}, valid_s=10.0)


def test_llm_only_mode_writes_only_verified_direct_shares(ric_db, cell) -> None:
    split = tool_call("set_slice_share", embb=0.4, urllc=0.45, be=0.15)
    starved = tool_call("set_slice_share", embb=0.5, urllc=0.48, be=0.02)  # below the floor: rejected
    script = [facts(urllc=10, dur=8.0), split, starved, split]
    record, _ = agent(ric_db, cell, script, controller="llm_only")
    assert record.state == "applied"
    rows = ric_db.execute("SELECT mode, shares FROM a1_policy").fetchall()
    assert {r[0] for r in rows} == {"direct_share"} and len(rows) == 2  # the rejected split never written
    assert all(min(json.loads(r[1])) >= 0.05 for r in rows)


@pytest.mark.parametrize(
    ("shares", "previous", "check"),
    [
        ([0.4, 0.4, 0.2], None, "ok"),
        ([0.4, float("nan"), 0.6], None, "schema"),
        ([0.5, 0.5], None, "schema"),
        ([0.5, 0.4, 0.2], None, "budget"),
        ([0.6, 0.38, 0.02], None, "floor"),
        ([0.8, 0.1, 0.1], [0.4, 0.3, 0.3], "step"),
        ([0.6, 0.2, 0.2], [0.3, 0.4, 0.3], "ok"),  # 0.3 exactly: allowed
    ],
)
def test_verify_shares(shares, previous, check) -> None:
    assert verify_shares(shares, previous).check == check


# ------------------------------------------------------------------ worker and client


def test_worker_claims_an_intent_only_once(ric_db, cell) -> None:
    from agent.intents import submit

    submit(ric_db, TEXT, 0.0)
    other = sqlite3.connect(ric_db.execute("PRAGMA database_list").fetchone()[2], isolation_level=None)
    other.row_factory = sqlite3.Row
    assert next_intent(ric_db) == "i0001"
    assert next_intent(other) is None  # already processing


def test_worker_turns_an_llm_failure_into_a_rejection(ric_db, cell) -> None:
    from agent.intents import submit

    class Down:
        model = "down"

        def chat(self, *args, **kwargs):
            raise LLMError("cannot reach Ollama")

    submit(ric_db, TEXT, 0.0)
    process_one(AgentContext(conn=ric_db, llm=Down(), settings=Settings(mode="eval")))
    record = intents.get(ric_db, "i0001")
    assert record.state == "rejected" and "cannot reach Ollama" in record.reason


def test_ollama_request_format(monkeypatch) -> None:
    sent = []

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.body).encode()

    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        sent.append(body)
        call = {"function": {"name": "get_kpis", "arguments": {"window_s": 5}}}
        return Response({"message": {"content": "", "tool_calls": [call]}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    reply = OllamaClient(model="m").chat([{"role": "user", "content": "x"}], tools=tools.definitions("proposed"),
                                         schema={"type": "object"})
    body = sent[0]
    assert body["options"] == {"temperature": 0, "num_ctx": 4096} and body["think"] is False
    assert body["format"] == {"type": "object"} and body["stream"] is False
    assert reply.tool_calls[0].name == "get_kpis" and reply.tool_calls[0].arguments == {"window_s": 5}
