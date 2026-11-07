"""M08a: the control evaluation's harness and measures, with no GPU and no Ollama (FakeLLM)."""

import json

import numpy as np
import pytest

from agent import tools
from agent.llm import FakeLLM, LLMReply, tool_call
from agent.parser import parse_facts
from agent.tools import AgentContext, Settings
from eval import metrics
from eval.control import FIXED_SEEDS, bridge, fixed, headline, lever
from eval.extraction import load_intents, score_one
from eval.harness import CONTROLLERS, ChargedLLM, RunSpec, run_once
from eval.scenarios import DEV_SEED, SCENARIOS, SEEDS, Scenario, ScenarioIntent, facts
from rl.evaluate import EVAL_SEEDS
from rl.goal_test import TEST_REWARD_SEEDS, VALIDATION_REWARD_SEEDS
from store import kpi
from store.kpi import CellInfo
from verifier.merge import ActiveIntent
from verifier.objective import RLObjective, SliceValues
from verifier.verify import VerifierState, verify

URLLC_20 = ScenarioIntent(0.0, "text", facts(4.0, urllc=20.0, max_latency_ms=10.0), reaction_from_s=0.0)
SHORT = Scenario("s2", "short", 40, (ScenarioIntent(5.0, SCENARIOS["s2"].intents[0].text,
                                                    facts(20.0, urllc=18.0, max_latency_ms=5.0), 5.0),))


def columns(thr, load, lat) -> dict:
    n = len(thr)
    other = {"t": np.arange(1.0, n + 1), "thr": np.zeros(n), "load": np.zeros(n), "lat": np.zeros(n),
             "share": np.full(n, 1 / 3), "policy": np.full(n, np.nan)}
    urllc = {**other, "thr": np.array(thr, float), "load": np.array(load, float), "lat": np.array(lat, float)}
    return {"embb": other, "urllc": urllc, "be": other}


# ---------------------------------------------------------------- measures, by hand


def test_a_target_counts_as_met_at_the_offered_load_and_needs_the_latency_bound() -> None:
    kpis = columns(thr=[20, 8, 8, 19], load=[30, 8, 9, 30], lat=[5, 5, 5, 5])
    assert metrics.met_mask(kpis, URLLC_20)[1].tolist() == [True, True, False, False]
    late = columns(thr=[25], load=[25], lat=[11])
    assert metrics.met_mask(late, URLLC_20)[1].tolist() == [False]


def test_the_score_uses_the_same_rule_as_the_agents_compliance(ric_db) -> None:
    """The evaluation must not measure something else than what the agent is told it achieved."""
    kpi.write_backend_info(ric_db, CellInfo("c", "sim", ("embb", "urllc", "be"), 106, 75.0, 2.0), "t")
    values = [(20.0, 30.0, 5.0), (8.0, 8.0, 5.0), (8.0, 9.0, 5.0), (25.0, 30.0, 11.0)]
    for i, (thr, load, lat) in enumerate(values, 1):
        ric_db.execute("INSERT INTO kpi (ts, source, cell_id, slice, dl_thr_mbps, prb_share_alloc, prb_used, n_ue, "
                       "latency_ms, offered_load_mbps) VALUES (?, 'sim', 'c', 'urllc', ?, 0.3, 1, 1, ?, ?)",
                       (float(i), thr, lat, load))
    objective = RLObjective(weights=SliceValues(urllc=1.0), min_thr_mbps=SliceValues(urllc=20.0), max_latency_ms=10.0)
    agent_view = tools.compliance(AgentContext(ric_db, Settings(), None), objective, 4.0)
    kpis = columns([v[0] for v in values], [v[1] for v in values], [v[2] for v in values])
    thr_only = ScenarioIntent(0.0, "", facts(4.0, urllc=20.0), 0.0)
    assert metrics.met_mask(kpis, thr_only)[1].mean() == agent_view["urllc_min_thr"]
    assert (kpis["urllc"]["lat"] <= 10.0).mean() == agent_view["urllc_latency"]


def test_reaction_needs_five_seconds_in_a_row_and_never_reached_stays_none() -> None:
    t = np.arange(1.0, 13.0)
    ok = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0], bool)
    assert metrics.reaction_s(t, ok, 0.0, 12.0) == 4.0  # holds from the 5th second
    assert metrics.reaction_s(t, np.zeros(12, bool), 0.0, 12.0) is None
    assert metrics.reaction_s(t, np.ones(12, bool), 0.0, 12.0) == 0.0  # already there
    assert metrics.reaction_s(t, ok, 0.0, 7.0) is None  # the 5 s do not fit in the window


def test_moved_counts_a_prb_once_and_can_look_only_at_part_of_a_run() -> None:
    kpis = columns([0, 0, 0], [0, 0, 0], [0, 0, 0])
    kpis["embb"] = {**kpis["embb"], "share": np.array([0.5, 0.4, 0.4])}
    kpis["urllc"] = {**kpis["urllc"], "share": np.array([0.3, 0.4, 0.4])}
    kpis["be"] = {**kpis["be"], "share": np.array([0.2, 0.2, 0.2])}
    assert metrics.moved_per_s(kpis) == pytest.approx(0.05)
    assert metrics.moved_per_s(kpis, np.array([False, True, True])) == pytest.approx(0.0)
    assert metrics.moved_per_s(kpis, np.array([True, False, True])) is None


def test_blocked_actions_are_counted_by_check() -> None:
    step = lambda accepted, check: {"event": "agent_step", "data": {"verifier": {"accepted": accepted, "check": check}}}
    events = [step(False, "conflict"), step(False, "conflict"), step(True, "ok"), {"event": "agent_step", "data": {}},
              {"event": "reverified", "data": {"accepted": False, "check": "step"}},
              {"event": "reverified", "data": {"accepted": True, "check": "ok"}}]
    assert metrics.blocked_by_check(events) == (3, {"conflict": 2, "step": 1})


def test_a_revision_taking_a_number_does_not_shift_the_second_intent() -> None:
    first, second = SCENARIOS["s3"].intents
    events = [{"event": "received", "intent_id": "i0001", "data": {"text": first.text}},
              {"event": "received", "intent_id": "i0003", "data": {"text": second.text}}]
    assert metrics.intent_id_of(events, second) == "i0003"
    assert metrics.intent_id_of([], second) is None


def test_a_lowered_target_is_reported_as_relaxed() -> None:
    intent = SCENARIOS["s2"].intents[0]
    accepted = {"tool": "set_rl_objective", "verifier": {"accepted": True}}
    same = {"min_thr_mbps": {"urllc": 18.0}, "max_latency_ms": 5.0, "duration_s": 180.0}
    lower = {**same, "min_thr_mbps": {"urllc": 15.0}}
    assert metrics.relaxed([{"data": {**accepted, "arguments": same}}], intent) == []
    assert "urllc minimum" in metrics.relaxed([{"data": {**accepted, "arguments": lower}}], intent)[0]


# ---------------------------------------------------------------- scenarios and seeds


@pytest.mark.parametrize("intent", [i for s in SCENARIOS.values() for i in s.intents], ids=lambda i: i.text[:25])
def test_scenario_intents_state_their_facts_and_fit_a_run(intent) -> None:
    assert score_one(intent.expected, parse_facts(intent.text))["correct"]


def test_no_scenario_text_comes_from_the_held_out_split() -> None:
    held_out = {r["text"] for r in load_intents("heldout")}
    assert not held_out & {i.text for s in SCENARIOS.values() for i in s.intents}


def test_the_seeds_are_fresh() -> None:
    used = set(EVAL_SEEDS) | set(TEST_REWARD_SEEDS) | set(VALIDATION_REWARD_SEEDS)
    assert not used & set(SEEDS) and DEV_SEED not in used and DEV_SEED not in SEEDS
    assert not (used | set(SEEDS)) & set(FIXED_SEEDS)
    assert len(fixed()) == 3 * 2 * 10 + 3 * 10


def objective_for(intent: ScenarioIntent, now: float) -> RLObjective:
    return tools.objective_from_decision({"embb": 3, "urllc": 3, "be": 3}, intent.expected["min_thr_mbps"],
                                         intent.max_latency_ms, intent.expected["duration_s"], now)


def test_each_scenario_goal_is_acceptable_alone_and_s3_conflicts() -> None:
    """The numbers follow from the verifier's rules: S3's second intent fails on check `conflict`."""
    for scenario in SCENARIOS.values():
        for intent in scenario.intents:
            state = VerifierState(now_s=intent.at_s, capacity_mbps=75.0)
            assert verify(objective_for(intent, intent.at_s), state).accepted, intent.text
    first, second = SCENARIOS["s3"].intents
    active = (ActiveIntent(intent_id="a", issued_at=60.0, objective=objective_for(first, 60.0)),)
    decision = verify(objective_for(second, 120.0), VerifierState(now_s=120.0, capacity_mbps=75.0, active=active))
    assert (decision.accepted, decision.check) == (False, "conflict")


def test_the_matrix_is_the_registered_one() -> None:
    assert len(headline(list(CONTROLLERS), SEEDS)) == 4 * 3 * 10
    assert len(lever()) == 2 * 10 and {s.scenario for s in lever()} == {"s1"}
    specs = bridge()
    assert len({s.run_id for s in specs}) == len(specs) == 3 * 3 * 10 + 3 * 5


# ---------------------------------------------------------------- the harness


def test_an_llm_call_costs_the_cell_its_real_time() -> None:
    passed: list[int] = []
    llm = ChargedLLM(FakeLLM([LLMReply("", latency_s=2.1), LLMReply("", latency_s=0.0)]), passed.append)
    llm.chat([])
    llm.chat([])
    assert passed == [3, 0]  # rounded up: the reply cannot arrive before the second ends


def test_static_never_runs_a_policy_and_ignores_intents(tmp_path) -> None:
    work = run_once(RunSpec("t", "s2", "static", 1), SHORT, tmp_path / "run", None)
    kpis, events = metrics.load_run(work)
    assert not [e for e in events if e["event"] == "received"]
    assert np.allclose(kpis["urllc"]["share"], 1 / 3) and len(kpis["urllc"]["t"]) == SHORT.seconds
    measured = metrics.measure_run(work, SHORT)
    assert measured["intents"][0]["intent_met"] == 0.0  # still scored: 7 ms against a 5 ms bound
    assert json.loads((work / "run.json").read_text())["controller"] == "static"


def test_llm_only_uses_the_verifier_and_the_runner_and_no_rl(tmp_path) -> None:
    parse = json.dumps({**facts(20.0, urllc=18.0, max_latency_ms=5.0)})
    split = tool_call("set_slice_share", embb=0.3, urllc=0.5, be=0.2)
    work = run_once(RunSpec("t", "s2", "llm_only", 1), SHORT, tmp_path / "run", None,
                    FakeLLM([parse, *[split] * 5]))
    kpis, events = metrics.load_run(work)
    assert any(e["event"] == "verified_shares" for e in events)
    # Equal before the first split, then the LLM's: there is no policy file in this run at all.
    assert kpis["urllc"]["share"][0] == pytest.approx(1 / 3)
    assert kpis["urllc"]["share"][10] == pytest.approx(0.5)
    assert metrics.measure_run(work, SHORT)["intents"][0]["parse_ok"] is True


def test_proposed_goes_through_the_verifier_and_a_second_intent_waits_its_turn(tmp_path, tiny_policy_file) -> None:
    two = Scenario("s3", "two", 60, (ScenarioIntent(5.0, "first", facts(30.0, urllc=10.0), 5.0),
                                     ScenarioIntent(6.0, "second", facts(30.0, embb=10.0), 6.0),))
    parse = lambda **k: json.dumps(facts(30.0, **k))
    objective = lambda **k: tool_call("set_rl_objective", importance={"embb": 3, "urllc": 4, "be": 3}, min_thr_mbps=k, duration_s=30.0)
    reflect = json.dumps({"decision": "done", "reason": "ok"})
    llm = FakeLLM([parse(urllc=10.0), objective(urllc=10.0), reflect, parse(embb=10.0), objective(embb=10.0), reflect])
    work = run_once(RunSpec("t", "s3", "proposed", 1), two, tmp_path / "run", tiny_policy_file, llm)
    _, events = metrics.load_run(work)
    applied = [(e["intent_id"], e["ts_backend"]) for e in events if e["event"] == "applied"]
    assert [i for i, _ in applied] == ["i0001", "i0002"]
    assert applied[1][1] >= 25.0  # the second waited for the first's 20 s observation: one worker
    assert metrics.measure_run(work, two)["run"]["proposals"] == 2


def test_the_run_stops_on_time_with_an_agent_still_working(tmp_path) -> None:
    parse = json.dumps(facts(300.0, urllc=18.0, max_latency_ms=5.0))  # far longer than the run
    llm = FakeLLM([parse, *[tool_call("set_slice_share", embb=0.3, urllc=0.5, be=0.2)] * 3])
    scenario = Scenario("s2", "short", 30, (ScenarioIntent(1.0, "x", facts(300.0, urllc=18.0, max_latency_ms=5.0), 1.0),))
    work = run_once(RunSpec("t", "s2", "llm_only", 1), scenario, tmp_path / "run", None, llm)
    kpis, events = metrics.load_run(work)
    assert kpis["urllc"]["t"][-1] == 30.0 and events[-1]["event"] == "run_over"


def test_scripted_importance_goes_through_the_verifier(tmp_path, tiny_policy_file) -> None:
    work = run_once(RunSpec("t", "s1", "scripted", 1, importance=6.0),
                    Scenario("s1", "short", 20, (ScenarioIntent(5.0, "x", facts(10.0, urllc=10.0), 5.0),)),
                    tmp_path / "run", tiny_policy_file)
    _, events = metrics.load_run(work)
    assert [e["event"] for e in events if e["event"] in ("reverified", "applied")] == ["reverified", "applied"]


def test_set_slice_share_is_not_a_tool_of_the_proposed_controller() -> None:
    assert "set_slice_share" not in tools.tools_for_mode("proposed")


def test_the_figures_draw_from_the_table_and_show_never_reached() -> None:
    from eval.plot_control import plot_met, plot_reaction, values

    row = {"study": "headline", "scenario": "s2", "intent": "1", "controller": "proposed", "intent_met": "0.9", "reaction_s": ""}
    rows = [row, {**row, "reaction_s": "3.0"}]
    assert values(rows, "reaction_s", study="headline", scenario="s2", intent=1, controller="proposed") == [3.0]
    assert plot_met(rows).axes and plot_reaction(rows).axes


# ---------------------------------------------------------------- the fixes of 2026-11-07


def scripted_agent(ric_db, tiny_policy_file, llm, text: str, **settings):
    from conftest import run_agent, start_cell

    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(5)
    return run_agent(ric_db, text, mode="eval", llm=llm, runner=runner, **settings)


def test_the_first_objective_keeps_the_default_weights_when_no_priority_was_stated(ric_db, tiny_policy_file) -> None:
    from verifier.objective import DEFAULT_OBJECTIVE

    parse = json.dumps(facts(100.0, urllc=10.0))
    call = tool_call("set_rl_objective", importance={"embb": 2, "urllc": 8, "be": 2}, min_thr_mbps={"urllc": 10.0}, duration_s=100.0)
    record = scripted_agent(ric_db, tiny_policy_file, FakeLLM([parse, call]), "x")
    assert record.state == "applied" and record.objective.weights == DEFAULT_OBJECTIVE.weights


def test_a_stated_priority_keeps_the_models_weights(ric_db, tiny_policy_file) -> None:
    parse = json.dumps({**facts(100.0, urllc=10.0), "priority": "urllc"})
    call = tool_call("set_rl_objective", importance={"embb": 2, "urllc": 8, "be": 2}, min_thr_mbps={"urllc": 10.0}, duration_s=100.0)
    record = scripted_agent(ric_db, tiny_policy_file, FakeLLM([parse, call]), "x")
    assert record.objective.weights.urllc == pytest.approx(8 / 12)


def test_a_revision_may_not_be_asked_for_above_the_good_enough_line(ric_db, tiny_policy_file) -> None:
    """Compliance is computed in code; a small model asked for a revision at 95 % (M08a)."""
    parse = json.dumps(facts(100.0, urllc=1.0))
    call = tool_call("set_rl_objective", importance={"urllc": 5}, min_thr_mbps={"urllc": 1.0}, duration_s=100.0)
    revise = json.dumps({"decision": "revise", "reason": "the model is unsure"})
    record = scripted_agent(ric_db, tiny_policy_file, FakeLLM([parse, call, revise]), "x", observe_s=5.0, max_revisions=1)
    assert record.state == "applied" and record.supersedes is None
