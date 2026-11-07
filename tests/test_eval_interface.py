"""M08b: the interface evaluation's measures and its rule that the held-out split is scored once."""

import json
import socket

import pytest

from eval import interface as I
from eval import interface_metrics as M
from eval.scenarios import facts

CLEAR = {**facts(600.0, urllc=10.0, max_latency_ms=8.0), "priority": "urllc"}
REFUSED = {**facts(0.0), "understood": False, "duration_s": None, "clarification": "which slice?"}


def event(name: str, source: str = "x", **data) -> dict:
    return {"event": name, "source": source, "ts_wall": data.pop("ts_wall", 100.0), "data": data, "intent_id": None}


def item(events: list[dict], text: str = "Garantis 10 Mb/s.", **end) -> dict:
    return {"start": {"id": "x", "text": text}, "events": events, "end": end, "start_wall": 0.0}


def call(tool: str, outcome: str = "ok", **kwargs) -> dict:
    return event("tool_call", "gateway", tool=tool, outcome=outcome, arguments=kwargs.pop("arguments", {}), duration_ms=kwargs.pop("duration_ms", 100.0), **kwargs)


# ---------------------------------------------------------------- numbers worked out by hand


@pytest.mark.parametrize("k, n, low, high", [(20, 20, 0.839, 1.0), (10, 20, 0.299, 0.701), (0, 20, 0.0, 0.161)])
def test_wilson_interval_against_known_values(k, n, low, high) -> None:
    got = M.wilson(k, n)
    assert got[0] == pytest.approx(low, abs=0.002) and got[1] == pytest.approx(high, abs=0.002)


def test_the_proposals_four_fields() -> None:
    perfect = M.proposal_fields(CLEAR, CLEAR)
    assert perfect == {"slice": True, "metric": True, "value": True, "window": True}
    wrong_value = M.proposal_fields(CLEAR, {**CLEAR, "min_thr_mbps": {"embb": None, "urllc": 12.0, "be": None}})
    assert wrong_value == {"slice": True, "metric": True, "value": False, "window": True}
    wrong_slice = M.proposal_fields(CLEAR, {**CLEAR, "priority": "embb", "min_thr_mbps": {"embb": 10.0, "urllc": None, "be": None}})
    assert not wrong_slice["slice"] and wrong_slice["window"]
    no_window = M.proposal_fields(CLEAR, {**CLEAR, "duration_s": None})
    assert no_window == {"slice": True, "metric": True, "value": True, "window": False}


def test_a_refused_clear_request_has_every_field_wrong_and_a_refused_ambiguous_one_has_none() -> None:
    assert not any(M.proposal_fields(CLEAR, REFUSED).values())
    assert M.proposal_fields(REFUSED, REFUSED) is None


def test_extraction_summary_counts_detection_and_false_alarms() -> None:
    items = [{"expected": CLEAR, "got": CLEAR, "correct": True, "ambiguous": False},
             {"expected": CLEAR, "got": REFUSED, "correct": False, "ambiguous": False},
             {"expected": REFUSED, "got": REFUSED, "correct": True, "ambiguous": True},
             {"expected": REFUSED, "got": CLEAR, "correct": False, "ambiguous": True}]
    out = M.summarise_extraction(items)
    assert (out["correct"], out["ambiguous_detected"], out["clear_refused"]) == (2, 1, 1)
    assert out["fields"]["value"] == 0.5


def test_tool_calls_fall_in_the_proposals_categories() -> None:
    items = [
        item([call("submit_intent"), call("get_kpis")], final_state="applied"),
        item([call("submit_intent")], final_state="rejected", verifier={"accepted": False, "check": "budget"}),
        item([call("submit_intent")], final_state="rejected", reason="needs clarification: which?"),
        item([call("submit_intent", "invalid_args")]),
        item([call("get_status", "http_404"), call("get_kpis", "unreachable")]),
        item([call("submit_intent")], final_state="rejected", reason="agent error: cannot reach Ollama"),
    ]
    table = M.tool_call_table(items)
    assert table["submit_intent"] == {M.SUCCEEDED: 2, M.REFUSED: 1, M.REJECTED: 1, M.ERRORED: 1}
    assert table["get_kpis"] == {M.SUCCEEDED: 1, M.REFUSED: 0, M.REJECTED: 0, M.ERRORED: 1}
    assert table["get_status"][M.ERRORED] == 1


def test_a_state_change_is_seen_in_the_log_or_in_the_store() -> None:
    assert not M.state_changed(item([], schedule_before=[], schedule_after=[]))
    assert M.state_changed(item([event("timetable_written", reason="apply i0001")], schedule_before=[], schedule_after=[]))
    assert M.state_changed(item([], schedule_before=[], schedule_after=[["rl"]]))
    assert not M.state_changed(item([event("timetable_written", reason="cancel i0001")], schedule_before=[], schedule_after=[]))


@pytest.mark.parametrize("sent, expected", [
    (["Garantis 10 Mb/s."], "exact"), (["Garantis 10 Mb/s"], "punctuation"), (["Guarantee 10 Mb/s."], "altered"), ([], "not_sent"),
    (["Garantis 10 Mb/s.", "Garantis 10 Mb/s."], "altered"),
])
def test_fidelity_compares_what_the_gateway_received_with_what_was_typed(sent, expected) -> None:
    calls = [call("submit_intent", arguments={"text": t}) for t in sent]
    assert M.fidelity(item(calls))[0] == expected


def test_interface_latency_leaves_out_the_agents_decision() -> None:
    answered = call("submit_intent", duration_ms=500.0, ts_wall=110.0)
    forward, back = M.interface_latency(item([answered], turn_start=100.0, turn_end=113.0))
    assert forward == pytest.approx(9.5) and back == pytest.approx(3.0)
    assert M.interface_latency(item([], turn_start=100.0, turn_end=113.0)) == (None, None)


def test_the_intent_id_comes_from_the_gateways_log() -> None:
    refused = call("submit_intent", "invalid_args")
    good = {**call("submit_intent"), "intent_id": "i0007"}
    assert M.submitted_id(item([refused, call("get_kpis"), good])) == "i0007"
    assert M.submitted_id(item([refused])) is None


def test_a_loosened_value_is_reported_and_an_unlabelled_item_has_nothing_to_compare() -> None:
    stated = facts(180.0, urllc=18.0, max_latency_ms=4.0)
    step = lambda bound: event("agent_step", tool="set_rl_objective", verifier={"accepted": True},  # noqa: E731
                               arguments={"min_thr_mbps": {"urllc": 18.0}, "max_latency_ms": bound, "duration_s": 180.0})
    assert M.relaxed(item([step(4.0)]), stated) == []
    assert "max_latency_ms" in M.relaxed(item([step(5.0)]), stated)[0]
    assert M.relaxed(item([step(5.0)]), None) == []


def test_items_are_cut_by_their_markers_in_file_order() -> None:
    log = [event("tools_offered"), event("item_start", id="a", text="t"), call("get_kpis"),
           event("item_end", final_state="applied"), call("get_kpis"), event("item_start", id="b", text="u"),
           event("item_end")]
    items = M.split_items(log)
    assert [len(i["events"]) for i in items] == [1, 0] and items[0]["end"]["final_state"] == "applied"


# ---------------------------------------------------------------- the rule that protects the held-out score


def test_the_held_out_split_needs_final_and_is_scored_once(tmp_path) -> None:
    clean = ("abc1234", I.FROZEN_PARSE_PROMPT)
    I.guard(tmp_path, "tune", "gateway", False, "abc1234+uncommitted", "other")  # the tune split is free to use
    with pytest.raises(SystemExit, match="--final"):
        I.guard(tmp_path, "heldout", "gateway", False, *clean)
    I.guard(tmp_path, "heldout", "gateway", True, *clean)  # not scored yet
    with pytest.raises(SystemExit, match="commit"):
        I.guard(tmp_path, "heldout", "gateway", True, "abc1234+uncommitted", I.FROZEN_PARSE_PROMPT)
    with pytest.raises(SystemExit, match="frozen"):
        I.guard(tmp_path, "heldout", "gateway", True, "abc1234", "another prompt")
    done = I.path_dir(tmp_path, "heldout", "gateway")
    done.mkdir(parents=True)
    (done / "done.json").write_text("{}")
    with pytest.raises(SystemExit, match="already been scored"):
        I.guard(tmp_path, "heldout", "gateway", True, *clean)
    I.guard(tmp_path, "heldout", "hermes", True, *clean)  # one path does not close the others


def test_the_older_extraction_command_cannot_score_the_held_out_set_again(tmp_path, monkeypatch) -> None:
    from eval import extraction

    done = tmp_path / "done.json"
    done.write_text("{}")
    monkeypatch.setattr(extraction, "HELD_OUT_SCORED", done)
    monkeypatch.setattr("sys.argv", ["extraction", "--parser", "rules", "--split", "heldout", "--final"])
    with pytest.raises(SystemExit, match="already been scored"):
        extraction.main()


def test_the_parse_prompt_hash_changes_with_the_prompt(monkeypatch) -> None:
    before = I.parse_prompt_hash()
    monkeypatch.setattr(I, "PARSE_SYSTEM", I.PARSE_SYSTEM + " ")
    assert I.parse_prompt_hash() != before


def test_the_frozen_hash_is_the_prompt_in_the_code() -> None:
    assert I.parse_prompt_hash() == I.FROZEN_PARSE_PROMPT


# ---------------------------------------------------------------- end to end, real processes, no LLM


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_gateway_path_end_to_end_with_real_processes(tmp_path) -> None:
    """Runner, service, worker and gateway as processes; rule-based agent, no RL, no LLM. Stops at the verifier."""
    items = [{"id": "ok", "kind": "intent", "text": "garantir 10 Mb/s au slice URLLC pendant 5 minutes", "ambiguous": False},
             {"id": "big", "kind": "intent", "text": "garantir 500 Mb/s au slice URLLC", "ambiguous": False},
             {"id": "split", "kind": "intent", "text": "Fixe le partage des PRB a 90/5/5.", "ambiguous": True}]
    probes = [("empty text", "submit_intent", {"text": "   "}), ("set the split directly", "set_slice_share", {"embb": 1.0})]
    I.run_gateway("tune", tmp_path, runner_args=("--baseline", "equal"), env={"RIC_AGENT": "rules"}, port=free_port(),
                  repeats=1, items=items, probes=probes)
    run = tmp_path / "r1"
    assert json.loads((run / "run.json").read_text())["repeat"] == 1
    assert not (run / "ric.db").exists()
    found = M.load_items(run)
    by_id = {i["start"]["id"]: i for i in found}
    assert M.outcome_class(by_id["ok"]["end"]) == "accepted" and M.state_changed(by_id["ok"])
    assert M.outcome_class(by_id["big"]["end"]) == M.REJECTED and not M.state_changed(by_id["big"])
    assert M.outcome_class(by_id["split"]["end"]) == "refused_by_parse" and not M.state_changed(by_id["split"])
    assert by_id["empty text"]["end"]["probe_answer"].startswith("error") and not M.state_changed(by_id["empty text"])
    assert by_id["set the split directly"]["end"]["probe_answer"].startswith("error")
    table = M.tool_call_table(found)
    assert table["submit_intent"][M.REFUSED] == 1 and table["submit_intent"][M.REJECTED] == 1
    tools = [e for e in M.read_events(run / "events.jsonl") if e["event"] == "tools_offered"]
    assert tools[0]["data"]["tools"] == sorted(M.GATEWAY_TOOLS)  # hard invariant 1, on the running gateway
