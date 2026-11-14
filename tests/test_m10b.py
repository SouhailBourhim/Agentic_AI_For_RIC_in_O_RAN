"""M10b: the non-learned executors (formula_est, mpc_est, formula_told), the LLM-free executor mode, and the checks.

The executors must read only what they say they read (formula_est and mpc_est: the observation), rev 4 must be
untouched by the refactor that gave it a sizing hook, and the checks' arithmetic must be what DECISIONS says.
"""

from pathlib import Path

import numpy as np
import pytest

from backends.sim import SimBackend
from eval import m10b
from eval.generator import dev_instances
from eval.harness import EXECUTORS, RunSpec, apply_scripted, run_once
from eval.oracle import FormulaToldPolicy, MpcEstPolicy, objective_intents
from rl.baselines import FORMULA_EST_MARGIN, SIZING_FRACTION, FormulaEstPolicy, FormulaPolicy, estimated_rates
from sim.cell import CellConfig
from sim.env import OBS_SIZE_WITH_APPLIED, OBS_SIZE_WITH_EFFICIENCY
from sim.traffic import TrafficPattern
from store import a1_policy, kpi
from verifier.objective import DEFAULT_OBJECTIVE, SLICES

# FormulaPolicy (rev 4) on _observations(), computed at cb12c44, before rev 4 got its sizing hook.
REV4_GOLDEN = np.array([
    [0.284134556112, 0.620396761172, 0.095468682715], [0.126743915750, 0.626908120800, 0.246347963449],
    [0.049604939157, 0.621919487475, 0.328475573368], [0.125502981131, 0.684281724155, 0.190215294714],
    [0.384404189632, 0.316148544388, 0.299447265980], [0.000000000000, 0.882845544533, 0.117154455467],
    [0.281995906561, 0.651920242908, 0.066083850531], [0.336564398295, 0.663435601705, 0.000000000000],
    [0.021748146017, 0.680411241464, 0.297840612519], [0.280194699362, 0.719805300638, 0.000000000000],
    [0.156373136764, 0.321670003327, 0.521956859909], [0.347350856265, 0.478382201775, 0.174266941960],
    [0.417578434363, 0.219932956883, 0.362488608754], [0.117425607266, 0.334226977003, 0.548347415731],
    [0.399081353530, 0.273153820151, 0.327764826319], [0.050000000000, 0.900000000000, 0.050000000000]])


def _observations(n: int = 16) -> list[np.ndarray]:
    """Random 34-value observations reaching every branch of rev 4 (bound or not, needs over the cell or not)."""
    rng = np.random.default_rng(20261114)
    out = []
    for _ in range(n):
        o = np.zeros(OBS_SIZE_WITH_EFFICIENCY)
        for i in range(3):
            o[5 * i:5 * i + 5] = [rng.uniform(0, 0.6), rng.uniform(0, 0.5), rng.uniform(0, 0.3), rng.uniform(0.05, 0.9),
                                  rng.uniform(0, 1)]
        o[15:18] = rng.dirichlet(np.ones(3))
        o[18:21] = np.where(rng.random(3) < 0.5, rng.uniform(0, 0.4, 3), 0.0)
        o[21] = rng.uniform(0.04, 0.2) if rng.random() < 0.6 else 0.0
        o[22:] = rng.uniform(0, 1.5, OBS_SIZE_WITH_EFFICIENCY - 22)
        out.append(o.astype(np.float32))
    return out


def _with_rates(obs: np.ndarray, latest, average) -> np.ndarray:
    obs = obs.copy()
    obs[OBS_SIZE_WITH_APPLIED:OBS_SIZE_WITH_APPLIED + 3] = latest
    obs[OBS_SIZE_WITH_APPLIED + 3:] = average
    return obs


# ---------------------------------------------------------------- the executors


def test_rev4_is_unchanged_by_its_sizing_hook() -> None:
    policy = FormulaPolicy(0.592)
    assert np.array([policy.shares(o) for o in _observations()]) == pytest.approx(REV4_GOLDEN, abs=1e-11)


def test_formula_est_with_rates_at_one_and_rev4s_margin_is_rev4() -> None:
    est, rev4 = FormulaEstPolicy(0.592, estimate="ewma", margin=1.0 - SIZING_FRACTION), FormulaPolicy(0.592)
    for obs in _observations():
        obs = _with_rates(obs, [1.0] * 3, [1.0] * 3)
        assert est.shares(obs) == pytest.approx(rev4.shares(obs), abs=1e-12)


def test_a_slice_measured_at_half_the_rate_needs_twice_the_share() -> None:
    """Needs over the cell, no bound: rev 4's rules split in proportion to need = min(target, load) / rate."""
    obs = _observations()[0].copy()
    obs[[0, 5, 10]] = 0.5  # every load above its target
    obs[18:22] = [0.4, 0.4, 0.4, 0.0]  # three targets of 0.4 of the cell, no bound
    policy = FormulaEstPolicy(0.592, estimate="latest", margin=0.0)
    assert policy.shares(_with_rates(obs, [1.0] * 3, [1.0] * 3)) == pytest.approx([1 / 3] * 3)
    assert policy.shares(_with_rates(obs, [0.5, 1.0, 1.0], [1.0] * 3)) == pytest.approx([0.5, 0.25, 0.25])


def test_the_estimate_options() -> None:
    obs = _with_rates(_observations()[0], [0.6, 1.2, 1.0], [0.9, 1.0, 1.0])
    assert estimated_rates(obs, "latest", 0.0) == pytest.approx([0.6, 1.2, 1.0])
    assert estimated_rates(obs, "ewma", 0.1) == pytest.approx([0.81, 0.9, 0.9])
    assert estimated_rates(obs, "min", 0.0) == pytest.approx([0.6, 1.0, 1.0])
    spread = abs(0.6 - 0.9) / 0.9
    assert estimated_rates(obs, "ewma", 0.0, volatility=0.5)[0] == pytest.approx(0.9 * (1 - 0.5 * spread))
    with pytest.raises(ValueError):
        estimated_rates(obs[:OBS_SIZE_WITH_APPLIED])  # a 28-value observation has no efficiency
    with pytest.raises(ValueError):
        FormulaEstPolicy(0.592, estimate="median")


def test_the_objective_becomes_one_pseudo_intent_per_target() -> None:
    obs = np.zeros(OBS_SIZE_WITH_EFFICIENCY, dtype=np.float32)
    obs[18:22] = [20 / 79, 15 / 79, 0.0, 0.10]  # eMBB 20, URLLC 15 under 10 ms
    intents = objective_intents(obs, 79.0, 60.0)
    assert [(i.targets, i.max_latency_ms) for i in intents] == [({"embb": pytest.approx(20.0)}, None),
                                                               ({"urllc": pytest.approx(15.0)}, pytest.approx(10.0))]
    obs[18:21] = 0.0
    assert [(i.targets, i.max_latency_ms) for i in objective_intents(obs, 79.0, 60.0)] == [({}, pytest.approx(10.0))]
    obs[21] = 0.0
    assert objective_intents(obs, 79.0, 60.0) == []


def test_mpc_est_reads_only_the_observation_and_meets_a_feasible_target() -> None:
    """Built with no backend. Steady loads, every rate known to be 0.8: URLLC's 20 Mb/s needs 20 / (79 x 0.8)."""
    model = CellConfig(capacity_mbps=79.0, base_latency_ms={s: 0.6 for s in SLICES}, max_backlog_mb=400.0, delay_model="flat")
    policy = MpcEstPolicy(model)
    obs = np.zeros(OBS_SIZE_WITH_EFFICIENCY, dtype=np.float32)
    obs[[0, 5, 10]] = [40 / 79, 25 / 79, 30 / 79]  # loads
    obs[[2, 7, 12]] = 0.006  # 0.6 ms: empty queues
    obs[[3, 8, 13]] = 1 / 3
    obs[19] = 20 / 79  # URLLC target 20 Mb/s
    obs = _with_rates(obs, [0.8] * 3, [0.8] * 3)
    shares = policy.shares(obs)
    assert shares[1] * 79.0 * 0.8 * (1 - FORMULA_EST_MARGIN) >= 20.0 - 1e-6  # mpc_est sizes with formula_est's margin
    assert shares.sum() == pytest.approx(1.0)


def test_formula_told_is_formula_est_fed_the_truth() -> None:
    cfg = CellConfig(capacity_mbps=76.0, reported_capacity_mbps=79.0)
    traffic = {s: TrafficPattern("steady", 20.0, 20.0, noise_std=0.0) for s in SLICES}
    backend = SimBackend(cfg, traffic, seed=0)
    backend.sleep(1.0)
    obs = _observations()[3]
    told = FormulaToldPolicy(backend, cfg).shares(obs)
    truth = obs.copy()
    for i, s in enumerate(SLICES):
        truth[5 * i] = 20.0 / 79.0
        truth[5 * i + 2] = backend._last[s].latency_ms / 100.0
    truth = _with_rates(truth, [76.0 / 79.0] * 3, truth[OBS_SIZE_WITH_APPLIED + 3:])
    assert told == pytest.approx(FormulaEstPolicy(cfg.told_urllc_base_latency_ms, "latest", 0.0).shares(truth))


# ---------------------------------------------------------------- the executor mode


def test_existing_run_ids_are_unchanged_and_an_executor_names_itself() -> None:
    assert RunSpec("m09b_dev", "f1_00", "static", 100000).run_id == "m09b_dev/f1_00/static/seed100000"
    assert RunSpec("lever", "s1", "scripted", 5000, importance=8.0).run_id == "lever/s1/scripted_i8/seed5000"
    assert RunSpec("check_r1", "f1_01", "scripted", 100001, executor="mpc_est").run_id == \
        "check_r1/f1_01/scripted_mpc_est/seed100001"


def test_an_executor_needs_the_scripted_controller(tmp_path: Path) -> None:
    inst = dev_instances(["f5"], 1)[0]
    for spec in (RunSpec("x", inst.name, "static", inst.seed, executor="rev4"),
                 RunSpec("x", inst.name, "scripted", inst.seed, executor="nonsense")):
        with pytest.raises(ValueError):
            run_once(spec, inst.scenario(), tmp_path, None)


def test_a_scripted_intent_with_no_importance_runs_the_default_weights(ric_db) -> None:
    backend = SimBackend(CellConfig(capacity_mbps=79.0))
    kpi.write_backend_info(ric_db, backend.info(), "test")
    intent = dev_instances(["f5"], 1)[0].intents[0]
    apply_scripted(ric_db, intent, None, 0.0)
    schedule = a1_policy.current_schedule(ric_db)
    assert schedule and schedule[0].objective.weights == DEFAULT_OBJECTIVE.weights
    stated = {s: v for s, v in intent.targets.items()}
    assert {s: getattr(schedule[0].objective.min_thr_mbps, s) for s in stated} == stated


def test_every_executor_plays_without_an_llm(tmp_path: Path, monkeypatch) -> None:
    """One short instance per executor through m10b.play: no LLM anywhere, the verifier logged once per intent."""
    monkeypatch.setattr(m10b, "RESULTS", tmp_path)
    inst = dev_instances(["f5"], 1)[0]
    for executor in EXECUTORS:
        work = m10b.play((inst, executor, "test"))
        assert (work / "run.json").exists()
    rows = m10b.score([inst], EXECUTORS, "test")
    assert len(rows) == len(EXECUTORS) * len(inst.intents) and all(r["accepted"] for r in rows)


# ---------------------------------------------------------------- the checks


def _row(instance: str, intent: int, executor: str, met: float, accepted: bool = True) -> dict:
    return {"family": "f1", "variant": "a", "instance": instance, "intent": intent, "executor": executor,
            "accepted": accepted, "intent_met": met}


def _table(values: dict[str, list[float]], rejected: bool = True) -> list[dict]:
    rows = [_row(f"i{k}", 0, e, v) for e, vs in values.items() for k, v in enumerate(vs)]
    if rejected:  # a refused intent the oracle "meets": it must count for nothing
        rows += [_row("i0", 1, e, 1.0 if e == "oracle" else 0.0, accepted=False) for e in values]
    return rows


def test_the_checks_on_a_synthetic_table() -> None:
    values = {"static": [0.2, 0.2, 0.2, 0.2], "v7": [0.3, 0.9, 0.3, 0.3], "rev4": [0.5, 0.5, 0.5, 0.5],
              "formula_est": [0.6, 0.6, 0.6, 0.6], "mpc_est": [0.605, 0.6, 0.6, 0.6],
              "formula_told": [0.8, 0.8, 0.8, 0.8], "oracle": [0.95, 0.95, 0.8, 1.0]}
    out = m10b.checks(_table(values))
    assert out["intents"] == 4
    assert out["oracle_feasible_share"] == pytest.approx(0.75) and not out["difficulty_pass"]
    assert out["envelope"] == pytest.approx((0.95 + 0.95 + 0.8 + 1.0) / 4)
    assert out["best_candidate_mean"] == pytest.approx(0.60125)
    assert out["headroom"] == pytest.approx(0.925 - 0.60125) and out["headroom_pass"]
    assert out["comparator"] == "formula_est"  # mpc_est is ahead by 0.00125, inside the 0.01 tie: the simpler wins


def test_the_comparator_is_the_best_candidate_beyond_the_tie() -> None:
    values = {e: [0.5] * 3 for e in EXECUTORS}
    values["mpc_est"] = [0.7] * 3
    values["oracle"] = [0.95] * 3
    out = m10b.checks(_table(values, rejected=False))
    assert out["comparator"] == "mpc_est" and out["difficulty_pass"]
    assert out["headroom"] == pytest.approx(0.25)


def test_the_paired_interval_is_a_minus_b() -> None:
    values = {e: [0.5] * 6 for e in EXECUTORS}
    values["oracle"] = [0.9] * 6
    mean, low, high = m10b.paired_interval(_table(values), "oracle", "rev4")
    assert mean == pytest.approx(0.4) and low == pytest.approx(0.4) and high == pytest.approx(0.4)


def test_tuning_and_checking_use_disjoint_halves() -> None:
    even, odd = {i.name for i in m10b.half(0)}, {i.name for i in m10b.half(1)}
    assert not even & odd and len(even) == len(odd) == 100
    assert even | odd == {i.name for i in dev_instances()}


def test_the_check_runs_once_and_after_its_rules(tmp_path: Path, monkeypatch) -> None:
    def refuse(**_):
        raise SystemExit("commit first")
    monkeypatch.setattr(m10b, "guard", refuse)
    with pytest.raises(SystemExit, match="commit"):
        m10b.check(1)
    monkeypatch.setattr(m10b, "guard", lambda **_: "abc1234")
    monkeypatch.setattr(m10b, "RESULTS", tmp_path)
    (tmp_path / "check_r1").mkdir()
    (tmp_path / "check_r1" / "result.json").write_text("{}")
    with pytest.raises(SystemExit, match="once"):
        m10b.check(1)


def test_the_tuning_budget(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m10b, "REVISIONS_CSV", tmp_path / "revisions.csv")
    with pytest.raises(SystemExit):
        m10b.tune(m10b.MAX_REVISIONS + 1, "too many")
    m10b.write_rows(m10b.REVISIONS_CSV, [{"revision": 1, "note": "x"}])
    with pytest.raises(SystemExit, match="already"):
        m10b.tune(1, "again")
