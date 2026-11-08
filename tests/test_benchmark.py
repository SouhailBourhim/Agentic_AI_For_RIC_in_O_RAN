"""M09b: the generator keeps its frozen families, the references behave as defined, the test seeds stay closed."""

import numpy as np
import pytest

from eval import benchmark as B
from eval import generator as G
from eval.oracle import choose
from eval.scenarios import ScenarioIntent, facts
from rl.baselines import FormulaPolicy
from rl.oracle import split_grid
from sim.cell import CellConfig, apply_min_share
from sim.env import build_observation
from verifier.objective import MIN_SLICE_SHARE, SLICES, RLObjective, SliceValues

ALL = G.dev_instances()


def test_same_seed_same_instance_and_the_test_seeds_stay_closed() -> None:
    assert G.generate("f3", 7) == G.generate("f3", 7)
    assert G.generate("f3", 7) != G.generate("f3", 8)
    with pytest.raises(SystemExit):
        G.generate("f1", 0, split="test")
    assert {i.seed for i in ALL}.isdisjoint({G.seed_of(f, k, "test") for f in G.FAMILIES for k in range(G.PER_FAMILY)})


def test_every_intent_fits_the_run_and_states_whole_numbers() -> None:
    for inst in ALL:
        assert len(inst.intents) >= 1
        for intent in inst.intents:
            assert intent.window[1] <= G.SECONDS and intent.expected["duration_s"] >= 60
            assert all(float(t).is_integer() for t in intent.targets.values())
            assert ("Mb/s" in intent.text) and ("minutes" in intent.text)


def test_the_families_keep_their_frozen_ranges() -> None:
    c = G.C
    for inst in ALL:
        targets = [t for i in inst.intents for t in i.targets.values()]
        if inst.family == "f1" and inst.variant == "a":
            assert max(m for _, m in inst.steps["urllc"]) >= 0.40 * c - 0.5 and inst.intents[0].max_latency_ms in G.BOUNDS
        if inst.family == "f1" and inst.variant == "b":
            t1, t2 = (list(i.targets.values())[0] for i in inst.intents)
            assert 27 <= t2 <= 31 and t2 - t1 <= 0.3 * c  # within the verifier's step, above a third of the cell
        if inst.family == "f2":
            assert list(inst.intents[0].targets) in (["embb"], ["be"]) and inst.intents[0].max_latency_ms is None
            assert 0.95 * c - 2 <= sum(inst.steps[s][0][1] for s in SLICES) <= 1.10 * c + 2
        if inst.family == "f3":
            levels = [m for _, m in inst.steps["urllc"]]
            assert len(levels) >= 12 and max(levels) >= 0.40 * c - 0.5
        if inst.family == "f4" and inst.variant == "b":
            assert 35 <= targets[0] <= 45
        if inst.family == "f5":
            s = list(inst.intents[0].targets)[0]
            assert targets[0] <= inst.steps[s][0][1] - 1 + 1e-9
        if not (inst.family == "f4" and inst.variant == "b"):
            assert max(targets) <= 31
    assert sum(i.family == "f1" and i.variant == "a" for i in ALL) == 20
    assert sum(i.family == "f4" and i.variant == "b" for i in ALL) == 10


def test_the_formula_gives_each_target_its_need_and_urllc_its_load_under_a_bound() -> None:
    cfg = CellConfig(capacity_mbps=79.0)
    loads = {"embb": 40.0, "urllc": 20.0, "be": 25.0}
    objective = RLObjective(weights=SliceValues(embb=0.4, urllc=0.3, be=0.3), min_thr_mbps=SliceValues(be=15.0), max_latency_ms=10.0)
    obs = build_observation(loads, loads, {s: 1.0 for s in SLICES}, {s: 1 / 3 for s in SLICES}, {s: 1.0 for s in SLICES}, objective, cfg)
    shares = FormulaPolicy(0.6).shares(obs)
    assert shares.sum() == pytest.approx(1.0)
    assert shares[SLICES.index("urllc")] >= 20.0 / 79.0 / (1 - 0.6 / 10.0) - 1e-6
    assert shares[SLICES.index("be")] >= 15.0 / 79.0 - 1e-6


def test_the_oracle_picks_a_split_that_meets_a_feasible_goal() -> None:
    cfg = CellConfig(capacity_mbps=79.0, delay_model="flat", max_backlog_mb=400.0, base_latency_ms={s: 0.6 for s in SLICES})
    intent = ScenarioIntent(0.0, "x", facts(300.0, urllc=25.0, max_latency_ms=5.0), 0.0)
    loads = np.tile([40.0, 30.0, 25.0], (60, 1))
    split = choose(apply_min_share(split_grid(), MIN_SLICE_SHARE), [intent], loads, np.zeros(3), np.full(3, 0.6), cfg, np.full(3, 1 / 3))
    assert split[1] * 79.0 >= 30.0  # URLLC must carry its whole load to stay under the bound


def test_the_oracle_drains_a_full_urllc_queue_instead_of_serving_embb() -> None:
    """The dev benchmark's failure (f3_03): URLLC's queue at the buffer limit when its bounded intent starts.
    No split meets the bound within a second; looking 60 s ahead, draining the queue does, later."""
    cfg = CellConfig(capacity_mbps=79.0, delay_model="flat", max_backlog_mb=407.7, base_latency_ms={s: 0.6 for s in SLICES})
    intent = ScenarioIntent(80.0, "x", facts(240.0, urllc=11.0, max_latency_ms=5.0), 80.0)
    loads = np.tile([41.0, 33.0, 20.0], (60, 1))
    full = np.array([0.0, 407.7, 0.0])
    split = choose(apply_min_share(split_grid(), MIN_SLICE_SHARE), [intent], loads, full, np.full(3, 0.6), cfg,
                   np.array([0.475, 0.3, 0.225]), t=80.0)
    assert split[1] * 79.0 > 33.0 + 407.7 / 60  # above its load by enough to empty the buffer within the horizon


def test_the_formula_gives_urllc_the_slack_while_over_the_bound() -> None:
    cfg = CellConfig(capacity_mbps=79.0)
    loads = {"embb": 40.0, "urllc": 30.0, "be": 20.0}
    objective = RLObjective(weights=SliceValues(embb=0.4, urllc=0.3, be=0.3), min_thr_mbps=SliceValues(urllc=11.0), max_latency_ms=5.0)
    shares = {s: 1 / 3 for s in SLICES}

    def urllc_share(latency_ms: float) -> float:
        lat = {"embb": 0.6, "urllc": latency_ms, "be": 0.6}
        obs = build_observation(loads, loads, lat, shares, {s: 1.0 for s in SLICES}, objective, cfg)
        return float(FormulaPolicy(0.6).shares(obs)[SLICES.index("urllc")])

    under = 30.0 / 74.0 / (1 - 0.6 / 5.0)  # under the bound: its load and margin, sized against 74 Mb/s
    assert urllc_share(3.0) == pytest.approx(under)
    assert urllc_share(800.0) == pytest.approx(1.0 - 2 * MIN_SLICE_SHARE)  # over it: all the slack (no other targets here)


def test_joint_feasibility_needs_all_intents_at_once() -> None:
    t = np.arange(1.0, 361.0)
    kpis = {s: {"t": t, "thr": np.full(360, 20.0), "load": np.full(360, 20.0), "lat": np.full(360, 1.0)} for s in SLICES}
    a = ScenarioIntent(0.0, "a", facts(300.0, embb=20.0), 0.0)
    b = ScenarioIntent(60.0, "b", facts(240.0, be=25.0), 60.0)  # BE's load is 20: min(25, 20) = 20 is met
    assert B.jointly_feasible(kpis, (a, b)) is True
    kpis["be"]["thr"][100:] = 10.0
    assert B.jointly_feasible(kpis, (a, b)) is False


def test_the_bootstrap_interval_contains_the_median() -> None:
    values = np.array([0.1, 0.4, 0.5, 0.6, 0.9] * 4)
    low, high = B.bootstrap_ci(values)
    assert low <= np.median(values) <= high


class ScriptedLLM:
    """Answers by what is asked: a parse (schema) with the rule parser's facts, a plan with a fixed split.
    Records every plan prompt, so the test can see what LLM only was told."""

    model = "scripted"

    def __init__(self) -> None:
        self.plans: list[str] = []

    def chat(self, messages, tools=None, schema=None):
        from agent.llm import LLMReply, ToolCall
        from agent.parser import parse_facts
        if schema is not None:
            text = messages[-1]["content"].split("\n")[-1]
            return LLMReply(parse_facts(text).model_dump_json(), latency_s=0.5)
        self.plans.append(messages[-1]["content"])
        return LLMReply("", [ToolCall("set_slice_share", {"embb": 0.5, "urllc": 0.25, "be": 0.25})], latency_s=0.5)


@pytest.mark.parametrize("controller, joins", [("llm_only", False), ("llm_only_v2", True)])
def test_the_fairer_llm_only_serves_a_second_request_while_it_runs(controller, joins, tmp_path, monkeypatch) -> None:
    import agent.graph
    from eval.harness import RunSpec, run_once

    monkeypatch.setattr(agent.graph, "parse_messages", lambda text: [{"role": "user", "content": text}])  # the text alone
    inst = G.generate("f1", 20)  # eMBB 17 Mb/s, then raised to 30
    llm = ScriptedLLM()
    run_once(RunSpec("test", inst.name, controller, inst.seed), inst.scenario(), tmp_path / controller, None, llm,
             cfg=B.profile_cell(inst), traffic=inst.traffic(0.0))
    second = inst.intents[1]
    told_30 = [p for p in llm.plans if '"embb": 30.0' in p]
    assert bool(told_30) == joins
    if joins:
        assert second.text in told_30[0]


def test_the_formula_serves_a_bounded_urllc_first_when_the_needs_exceed_the_cell() -> None:
    cfg = CellConfig(capacity_mbps=79.0)
    loads = {"embb": 37.0, "urllc": 33.0, "be": 19.5}
    objective = RLObjective(weights=SliceValues(embb=0.4, urllc=0.3, be=0.3),
                            min_thr_mbps=SliceValues(embb=23.7, urllc=12.0, be=23.7), max_latency_ms=10.0)
    obs = build_observation(loads, loads, {s: 0.6 for s in SLICES}, {s: 1 / 3 for s in SLICES}, {s: 1.0 for s in SLICES}, objective, cfg)
    shares = FormulaPolicy(0.6).shares(obs)
    assert shares.sum() == pytest.approx(1.0)
    assert shares[SLICES.index("urllc")] == pytest.approx(33.0 / 74.0 / (1 - 0.6 / 10.0))  # its whole need, not scaled
