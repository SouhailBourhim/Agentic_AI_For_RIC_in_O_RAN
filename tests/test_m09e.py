"""M09e: the testbed plays generated instances as the simulator did, each controller is started as on the simulator,
and the pass rule does what the pre-registration says. Nothing here touches Docker or the testbed."""

import json

import pytest

from agent.worker import settings_from_env
from eval import m09e as E
from eval.generator import SECONDS, draw


def test_one_client_per_rate_change_each_on_its_own_port() -> None:
    steps = {"embb": ((0.0, 30.0),), "urllc": ((0.0, 5.0), (90.0, 5.0), (100.0, 30.0), (130.0, 6.0)), "be": ((0.0, 20.0),)}
    plan = E.pieces(steps)
    urllc = [p for p in plan if p.slice == "urllc"]
    assert [(p.start_s, p.seconds, p.rate_mbps) for p in urllc] == [(0.0, 100, 5.0), (100.0, 30, 30.0), (130.0, 230, 6.0)]
    assert len({p.port for p in plan}) == len(plan)
    assert [p.start_s for p in plan] == sorted(p.start_s for p in plan)


def test_every_sampled_instance_is_covered_end_to_end() -> None:
    for inst in E.sample():
        plan = E.pieces(inst.steps)
        for s in ("embb", "urllc", "be"):
            assert sum(p.seconds for p in plan if p.slice == s) == SECONDS
        assert len({p.port for p in plan}) == len(plan)


def test_until_cuts_the_plan() -> None:
    inst = draw("f3", 0, "dev")
    plan = E.pieces(inst.steps, 120.0)
    assert all(p.start_s < 120 for p in plan)
    assert all(sum(p.seconds for p in plan if p.slice == s) == 120 for s in ("embb", "urllc", "be"))


def test_the_order_rotates_and_covers_every_pair() -> None:
    names = [i.name for i in E.sample()]
    one, two = E.order(1, names), E.order(2, names)
    assert sorted(one) == sorted(two) and len(one) == 32 == len(set(one))
    assert all(a != b for a, b in zip(one, two))


def test_each_controller_is_started_as_on_the_simulator() -> None:
    assert E.TOLD_URLLC_BASE_MS == pytest.approx(0.592)
    assert E.runner_args("static") == E.runner_args("llm_only_v2") == "--baseline equal --urllc-base-latency-ms 0.592"
    assert E.runner_args("formula") == "--baseline formula --urllc-base-latency-ms 0.592"
    assert E.runner_args("proposed") == "--urllc-base-latency-ms 0.592"
    env = E.controller_env("llm_only_v2")
    assert env["RIC_CONTROLLER"] == "llm_only" and env["RIC_MODE"] == "eval"
    assert E.controller_env("formula")["RIC_CONTROLLER"] == "proposed"


def test_the_worker_reads_the_fairer_llm_only_settings(monkeypatch) -> None:
    for k, v in E.AGENT_ENV["llm_only_v2"].items():
        monkeypatch.setenv(k, v)
    settings = settings_from_env()
    assert settings.controller == "llm_only" and settings.llm_only_prompt == "v2" and settings.llm_only_joins
    for k in E.AGENT_ENV["llm_only_v2"]:
        monkeypatch.delenv(k)
    default = settings_from_env()
    assert default.llm_only_prompt == "v1" and not default.llm_only_joins


@pytest.mark.parametrize("predicted, difference, expected", [
    ("tie", 0.15, True), ("tie", -0.15, True), ("tie", 0.151, False),
    ("+", 0.051, True), ("+", 0.05, False), ("+", -0.2, False),
    ("-", -0.051, True), ("-", -0.05, False),
])
def test_a_prediction_is_confirmed_by_the_registered_rule(predicted, difference, expected) -> None:
    assert E.confirmed(predicted, difference) is expected


def _prediction(name, controller, best, sign):
    return {"instance": name, "controller": controller, "best": str(best), "proposed_minus_this": sign}


def test_judge_counts_best_matches_and_pairs() -> None:
    preds = []
    for k in range(8):
        name = f"i{k}"
        preds += [_prediction(name, "static", False, "+"), _prediction(name, "llm_only_v2", False, "tie"),
                  _prediction(name, "formula", True, "-"), _prediction(name, "proposed", False, "")]
    measured = {}
    for k in range(8):
        name = f"i{k}"
        good = k < 6  # 6 instances behave as predicted, 2 don't
        measured |= {(name, "static"): 0.3, (name, "llm_only_v2"): 0.6 if good else 0.9,
                     (name, "formula"): 0.95 if good else 0.5, (name, "proposed"): 0.6}
    verdict = E.judge(measured, preds)
    assert verdict["best_matches"] == 6
    # good instances: 3/3 pairs; bad ones: static + (0.3 > 0.05), tie fails (-0.3), formula - fails (+0.1)
    assert verdict["pairs_confirmed"] == pytest.approx((6 * 3 + 2 * 1) / 24, abs=1e-4)
    assert verdict["pass"] is (verdict["pairs_confirmed"] >= 0.75)


def test_scores_average_runs_then_repeats() -> None:
    rows = [{"instance": "a", "controller": "proposed", "repeat": 1, "intent_met": 1.0},
            {"instance": "a", "controller": "proposed", "repeat": 1, "intent_met": 0.0},
            {"instance": "a", "controller": "proposed", "repeat": 2, "intent_met": 1.0}]
    assert E.scores(rows)[("a", "proposed")] == pytest.approx((0.5 + 1.0) / 2)


def test_final_and_score_refuse_a_spent_set_or_an_uncommitted_tree(tmp_path, monkeypatch) -> None:
    with pytest.raises(SystemExit, match="--final"):
        E.score(False)
    register = tmp_path / "spent.json"
    register.write_text(json.dumps({"sets": {E.SPENT_NAME: {"date": "2026-11-12", "commit": "x", "result_file": "r"}}}))
    with pytest.raises(SystemExit, match="already scored"):
        E.score(True, register)
    from eval import m09d

    monkeypatch.setattr(m09d, "commit", lambda: "abc+uncommitted")
    with pytest.raises(SystemExit, match="commit the code"):
        E.score(True, tmp_path / "empty.json")


# Real `docker logs --timestamps` lines of the gNB on 2026-11-11, when the simulator ran at half speed; the second one
# has an E2 agent message interleaved before the MAC line, as the gNB really printed it.
SLOW_LOG = """2026-11-11T21:32:41.481509455Z 3495.541877 [NR_MAC] I Frame.Slot 640.0
2026-11-11T21:32:44.155251645Z [E2 AGENT][E2SM-KPM] Conditio3498.215611 [NR_MAC] I Frame.Slot 768.0
2026-11-11T21:32:46.862996985Z 3500.923219 [NR_MAC] I Frame.Slot 896.0
2026-11-11T21:32:47.000000000Z 3501.0 [NR_MAC] I UE 7eb1: dlsch_rounds 657551/0/0/0, MCS (0) 28 (Qm 6)
2026-11-11T21:32:49.562000000Z 3503.6 [NR_MAC] I Frame.Slot 0.0
"""


def test_frame_marks_reads_the_gnb_log_including_interleaved_lines() -> None:
    marks = E.frame_marks(SLOW_LOG)
    assert [f for _, f in marks] == [640, 768, 896, 0]
    assert marks[1][0] - marks[0][0] == pytest.approx(2.674, abs=1e-3)


def test_rfsim_speed_handles_the_frame_wrap_and_measures_the_slow_day() -> None:
    # 3 steps of 128 frames (the last across the 1024 wrap) = 3.84 simulated s in 8.08 wall s
    assert E.rfsim_speed(E.frame_marks(SLOW_LOG)) == pytest.approx(0.475, abs=0.002)


def test_rfsim_speed_is_one_at_real_time_and_none_without_two_marks() -> None:
    real_time = [(100.0 + 1.28 * k, (128 * k) % 1024) for k in range(12)]
    assert E.rfsim_speed(real_time) == pytest.approx(1.0)
    assert E.rfsim_speed(real_time[:1]) is None
    assert E.rfsim_speed([]) is None


def test_a_slow_or_unmeasured_simulator_makes_the_run_invalid() -> None:
    assert E.speed_problems(1.0) == []
    assert E.speed_problems(0.83) == []  # the slowest healthy run under traffic, 2026-11-13
    assert E.speed_problems(E.MIN_RFSIM_SPEED_RUN) == []
    assert E.speed_problems(0.74) and E.speed_problems(0.47) and E.speed_problems(None)


def test_play_refuses_to_start_on_a_slow_simulator(monkeypatch) -> None:
    monkeypatch.setattr(E, "measure_rfsim_speed", lambda since, until=None: 0.47)
    with pytest.raises(SystemExit, match="0.47x real time"):
        E.require_real_time()
    monkeypatch.setattr(E, "measure_rfsim_speed", lambda since, until=None: None)
    with pytest.raises(SystemExit):
        E.require_real_time()


def test_the_speed_is_read_from_a_well_formed_docker_logs_command(monkeypatch) -> None:
    # 2026-11-13: --until was once inserted between --since and its value, so every run was "not measurable"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return type("Done", (), {"stdout": SLOW_LOG, "stderr": ""})()

    monkeypatch.setattr(E.subprocess, "run", fake_run)
    assert E.measure_rfsim_speed(100.0, 220.0) == pytest.approx(0.475, abs=0.002)
    assert E.measure_rfsim_speed(100.0) == pytest.approx(0.475, abs=0.002)
    assert calls == [["docker", "logs", "--timestamps", "--since", "100.000", "--until", "220.000", E.GNB],
                     ["docker", "logs", "--timestamps", "--since", "100.000", E.GNB]]
