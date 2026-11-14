"""M10c': the m10c recipe (RL on the m10 cell), its goal-test splits, the oracle with a channel, and the median seed.

What these tests hold: the recipe trains what the plan says (m10 cell, 34 inputs, alpha 0.25, its own validation
split); earlier recipes keep every option they had; the goal test gives each scenario its own channel and the same
one to every objective told there; and the oracle predicts with the channel the env then applies.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from rl import goal_test, m09, sweep
from rl.oracle import GreedyOracle
from sim.channel import sample_channel
from sim.env import OBS_SIZE_WITH_APPLIED, OBS_SIZE_WITH_EFFICIENCY, SlicingEnv
from verifier.objective import DEFAULT_OBJECTIVE

# ---------------------------------------------------------------- the recipe


def test_the_m10c_recipe_is_v9_noswitch_on_the_m10_cell() -> None:
    r = m09.RECIPES["m10c"]
    assert {k: v for k, v in r.items() if k not in ("profile", "efficiency", "alpha", "validation")} == \
        {k: v for k, v in m09.RECIPES["v9_noswitch"].items() if k != "profile"}
    assert (r["profile"], r["efficiency"], r["alpha"], r["validation"]) == ("m10", True, 0.25, "m10c_validation")
    assert m09.strict_checkpoint("m10c") and m09.validation_split("m10c") == "m10c_validation"
    assert "m10c" not in m09.V9_RECIPES  # M09h's tables and rules stay about its own six arms


def test_earlier_recipes_keep_their_options() -> None:
    for recipe in m09.V8_RECIPES + m09.V9_RECIPES:
        assert m09.option(recipe, "efficiency") is False and m09.option(recipe, "alpha") == m09.FILTER_ALPHA
    assert m09.validation_split("v9_noswitch") == "m10_validation"
    assert m09.validation_split("v8_nooverserve") == "m09_validation"
    assert not m09.strict_checkpoint("v8") and m09.strict_checkpoint("v9")
    assert "channel" not in m09.env_options("m09", "on")  # no channel option at all on the calibrated cell


def test_the_m10c_training_env_has_a_channel_and_34_inputs() -> None:
    r = m09.RECIPES["m10c"]
    env = SlicingEnv(episode_steps=m09.EPISODE_STEPS, action_filter_alpha=r["alpha"], min_share=0.05,
                     reward_version=r["reward"], **m09.training_options("m10", "on", True, 0.0, True))
    assert env.observation_space.shape == (OBS_SIZE_WITH_EFFICIENCY,) and env.action_filter_alpha == 0.25
    env.reset(seed=3)
    first = env.channel_trace.factors.copy()
    env.reset(seed=3)
    assert np.array_equal(first, env.channel_trace.factors)  # a seed gives its channel
    env.reset()
    assert not np.array_equal(first, env.channel_trace.factors)  # the next episode gets another
    obs, *_ = env.step(np.array([0.4, 0.3, 0.3], dtype=np.float32))
    assert obs.shape == (OBS_SIZE_WITH_EFFICIENCY,) and np.isfinite(obs).all()


# ---------------------------------------------------------------- the goal test


def test_the_m10c_splits_have_fresh_seeds_and_a_channel_per_scenario() -> None:
    bases = sorted(goal_test.M09_SEEDS.values())
    assert goal_test.M09_SEEDS["m10c_validation"] == 13000 and goal_test.M09_SEEDS["m10c_test"] == 14000
    assert all(b - a >= 1000 for a, b in zip(bases, bases[1:]))  # every split's 300 seeds stay apart
    cases = goal_test.case_set("m10c_validation")
    assert set(cases.channels) == set(cases.scenarios)
    traces = [c.factors for c in cases.channels.values()]
    assert all(not np.array_equal(traces[0], t) for t in traces[1:])
    assert goal_test.setting_of("m10c_test").alpha == 0.25
    assert goal_test.setting_of("m10c_test").env["efficiency_features"] is True
    assert "m10c_test" in goal_test.FINAL_SPLITS
    assert goal_test.case_set("m10_validation").channels == {}  # M09h's splits unchanged


def test_every_objective_told_in_a_scenario_meets_the_same_channel() -> None:
    """The paired advantage compares the same cell: the rollout's channel is the scenario's, not seed 0's."""
    cases = goal_test.case_set("m10c_validation")
    scenario = next(iter(cases.scenarios))
    seen = []

    def act(env, _obs):
        seen.append(env.channel_trace.factors)
        return np.array([1 / 3] * 3)

    for objective in list(cases.objectives.values())[:2]:
        goal_test.rollout(act, cases.scenarios[scenario], objective, setting=goal_test.M10C_SETTING,
                          channel=cases.channels[scenario])
    assert np.array_equal(seen[0], cases.channels[scenario].factors) and np.array_equal(seen[0], seen[-1])


def test_the_greedy_oracle_predicts_with_the_channel(monkeypatch) -> None:
    import rl.oracle as oracle_module

    env = SlicingEnv(episode_steps=20, history="on", applied_split=True, efficiency_features=True)
    trace = sample_channel(np.random.default_rng(5), 20)
    env.reset(seed=0, options={"objective": DEFAULT_OBJECTIVE, "channel": trace})
    passed = []
    real = oracle_module.step_slices_batch

    def spy(*args):
        passed.append(args[-1])
        return real(*args)

    monkeypatch.setattr(oracle_module, "step_slices_batch", spy)
    oracle = GreedyOracle(DEFAULT_OBJECTIVE)
    for _ in range(3):
        env.step(oracle.act(env))
    assert all(np.array_equal(p, trace.at(float(t))) for t, p in enumerate(passed))


# ---------------------------------------------------------------- the median seed


def test_the_median_seed_of_m10c(tmp_path: Path) -> None:
    """Ranked by the kept checkpoint's validation reward; a seed with no checkpoint ranks last."""
    rewards = {70: 0.10, 71: 0.30, 72: None, 73: 0.20, 74: 0.25}
    for seed, reward in rewards.items():
        stem = tmp_path / f"td3_m10c_seed{seed}"
        stem.with_suffix(".zip").write_text("")
        history = [{"t2_pass": reward is not None, "t1a_pass": True, "t1a_required": True,
                    "t1b_captured": 0.6, "reward": reward or 0.0}]
        stem.with_name(f"{stem.name}_validation.json").write_text(json.dumps(history))
        if reward is not None:
            stem.with_name(f"{stem.name}_best.zip").write_text("")
    chosen = sweep.median_seeds(tmp_path, m09.M10_RECIPES)["m10c"]
    assert chosen == tmp_path / "td3_m10c_seed73_best.zip"  # order 72 (none), 70, 73, 74, 71: the third


def test_older_observations_are_a_prefix_of_the_m10c_one() -> None:
    assert OBS_SIZE_WITH_APPLIED + 6 == OBS_SIZE_WITH_EFFICIENCY
    with pytest.raises(ValueError):
        m09.cell("m10x")


# ---------------------------------------------------------------- the test set and the scoring (eval.m10c)


def test_the_m10c_test_set_is_fresh_and_has_no_f4() -> None:
    from eval import generator as G

    with pytest.raises(SystemExit):
        G.generate("f1", 0, split="m10c_test")
    test = G.m10c_test_instances()
    assert len(test) == 160 and {i.family for i in test} == {"f1", "f2", "f3", "f5"}
    used = {i.seed for i in G.dev_instances()} | {i.seed for i in G.test_instances()}
    assert all(i.seed >= 300000 for i in test) and not used & {i.seed for i in test}


def test_the_h1_readings_in_their_order() -> None:
    from eval.m10c import reading

    assert reading(0.01, 0.04) == "RL superior"  # superior comes first, even inside the equivalence margin
    assert reading(-0.04, -0.01) == "formula better"
    assert reading(-0.04, 0.03) == "equivalent"
    assert reading(-0.08, 0.03) == "inconclusive" and reading(-0.03, 0.07) == "inconclusive"


def test_the_summary_on_a_synthetic_table() -> None:
    from eval import m10c

    rows = []
    for k, family in enumerate(["f1", "f2", "f3", "f5"] * 3):
        for e in m10c.EXECUTORS:
            met = {"m10_rl": 0.7, "formula_est": 0.6, "formula_told": 0.65, "oracle": 0.9}.get(e, 0.5)
            rows.append({"family": family, "variant": "", "instance": f"i{k}", "intent": 0, "executor": e,
                         "accepted": True, "intent_met": met})
    out = m10c.summary(rows)
    assert out["h1"]["mean"] == pytest.approx(0.1) and out["h1"]["reading"] == "RL superior"
    assert out["foresight_closed"] == pytest.approx((0.7 - 0.65) / (0.9 - 0.65))
    assert set(out["h1_by_family"]) == {"f1", "f2", "f3", "f5"} and out["intents"] == 12


def _seed_files(folder: Path, rewards: dict[int, float | None]) -> None:
    for seed, reward in rewards.items():
        stem = folder / f"td3_m10c_seed{seed}"
        stem.with_suffix(".zip").write_text("")
        history = [{"t2_pass": reward is not None, "t1a_pass": True, "t1a_required": True, "t1b_captured": 0.6,
                    "reward": reward or 0.0}]
        stem.with_name(f"{stem.name}_validation.json").write_text(json.dumps(history))
        if reward is not None:
            stem.with_name(f"{stem.name}_best.zip").write_text("")


def test_the_choice_is_the_median_seed_with_its_fallback(tmp_path: Path, monkeypatch) -> None:
    from eval import m10c

    monkeypatch.setattr(m10c, "POLICY_DIR", tmp_path)
    monkeypatch.setattr(m10c, "CHOICE", tmp_path / "choice.json")
    _seed_files(tmp_path, {70: 0.1, 71: 0.3, 72: None, 73: 0.2, 74: 0.25})
    choice = m10c.choose()
    assert choice["file"].endswith("td3_m10c_seed73_best.zip") and not choice["fallback_to_final_model"]
    for f in tmp_path.glob("td3_m10c_seed7[0134]_best.zip"):
        f.unlink()  # now only seed 72's... none: every seed without a checkpoint
    _seed_files(tmp_path, {70: None, 71: None, 73: None, 74: None})
    choice = m10c.choose()
    assert choice["fallback_to_final_model"] and choice["file"].endswith(".zip") and "_best" not in choice["file"]


def test_the_learned_executor_plays_at_its_alpha(tmp_path: Path, monkeypatch, tiny_policy_file: Path) -> None:
    from eval import m10c
    from eval.generator import dev_instances

    monkeypatch.setattr(m10c, "RESULTS", tmp_path)
    inst = dev_instances(["f5"], 1)[0]
    work = m10c.play((inst, "m10_rl", "t", str(tiny_policy_file)))
    run = json.loads((work / "run.json").read_text())
    assert run["executor"] == "m10_rl" and run["alpha"] == 0.25 and run["delay_ticks"] == 2


def test_the_test_is_scored_once_and_only_with_final() -> None:
    from eval import m10c

    with pytest.raises(SystemExit, match="final"):
        m10c.score_test(final=False)
