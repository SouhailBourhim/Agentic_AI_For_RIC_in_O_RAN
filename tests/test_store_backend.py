"""M04a: the store tables and the SimBackend (same cell as training, same observation)."""

import numpy as np
import pytest

from backends.sim import SimBackend, scenario_traffic
from rl.runner import observation_from_records
from sim.cell import CellConfig
from sim.env import SlicingEnv
from store import a1_policy, intents, kpi
from store.kpi import KpiRecord
from verifier.objective import DEFAULT_OBJECTIVE, SLICES, RLObjective, SliceValues

# ------------------------------------------------------------------ SimBackend


def test_sim_backend_matches_the_training_env() -> None:
    """Same traffic, seed and splits: SimBackend and SlicingEnv give identical KPIs and observations.

    This is what makes the runner's cell the one the policy was trained on.
    """
    cfg = CellConfig()
    traffic = scenario_traffic("urllc_spike", cfg)
    env = SlicingEnv(cfg)
    env_obs, _ = env.reset(seed=3, options={"traffic": traffic, "objective": DEFAULT_OBJECTIVE})
    backend = SimBackend(cfg, traffic, seed=3)
    splits = np.random.default_rng(0).dirichlet(np.ones(3), size=30)
    for split in splits:
        assert observation_from_records(backend.read_kpis(), DEFAULT_OBJECTIVE, cfg) == pytest.approx(env_obs)
        env_obs, _, _, _, info = env.step(split)
        backend.apply_prb_split(list(split))
        backend.sleep(1.0)
        records = {r.slice: r for r in backend.read_kpis()}
        for s in SLICES:
            assert records[s].dl_thr_mbps == pytest.approx(info["throughput_mbps"][s])
            assert records[s].latency_ms == pytest.approx(info["latency_ms"][s])
            assert records[s].offered_load_mbps == pytest.approx(info["load_mbps"][s])
    assert backend.now() == env.time_s == 30.0


@pytest.mark.parametrize("shares", [[0.5, 0.5], [0.6, 0.6, -0.2], [0.5, 0.3, 0.3], [float("nan"), 0.5, 0.5]])
def test_sim_backend_refuses_malformed_split(shares: list[float]) -> None:
    with pytest.raises(ValueError):
        SimBackend().apply_prb_split(shares)


def test_sim_backend_paces_against_wall_clock() -> None:
    """speed=20: 10 ticks of 1 s take about 0.5 s of real time."""
    import time

    backend = SimBackend(speed=20.0)
    start = time.monotonic()
    backend.sleep(10.0)
    assert 0.4 < time.monotonic() - start < 1.0
    assert backend.now() == 10.0


# ------------------------------------------------------------------ KPI table


def test_get_kpis_mean_p95_last_over_window(ric_db) -> None:
    for t in range(1, 21):  # throughput t Mb/s at t seconds
        kpi.write_records(ric_db, [KpiRecord(float(t), "sim", "c", "urllc", float(t), 0.3, 0.5, 1)], None)
    result = kpi.get_kpis(ric_db, "urllc", window_s=10.0)  # ts 11..20
    thr = result["slices"]["urllc"]["dl_thr_mbps"]
    assert (thr["mean"], thr["p95"], thr["last"]) == (15.5, 20.0, 20.0)
    assert "latency_ms" not in result["slices"]["urllc"]  # NULL over the window: left out
    assert kpi.get_kpis(ric_db, "embb")["slices"] == {}


def test_get_kpis_before_first_tick(ric_db) -> None:
    assert kpi.get_kpis(ric_db) == {"ts": None, "window_s": 10.0, "slices": {}}


# ------------------------------------------------------------------ a1_policy timetable


def _segment(valid_from: float, expires_at: float | None, ids: tuple[str, ...]) -> a1_policy.Segment:
    objective = RLObjective(weights=SliceValues(embb=0.2, urllc=0.6, be=0.2), valid_from=valid_from,
                            valid_until=expires_at)
    return a1_policy.Segment(objective, ids, {"accepted": True}, "console", valid_from, expires_at)


def test_in_force_follows_the_newest_timetable(ric_db) -> None:
    a1_policy.write_schedule(ric_db, [_segment(10, 50, ("i0001",)), _segment(50, 80, ("i0001", "i0002"))],
                             "apply i0002", 10.0)
    assert a1_policy.in_force(ric_db, 5.0) is None  # before the first segment: default objective
    assert a1_policy.in_force(ric_db, 10.0).intent_ids == ("i0001",)
    assert a1_policy.in_force(ric_db, 50.0).intent_ids == ("i0001", "i0002")  # end excluded, start included
    assert a1_policy.in_force(ric_db, 80.0) is None
    a1_policy.write_schedule(ric_db, [], "cancel i0001", 60.0)  # an empty timetable replaces it
    assert a1_policy.in_force(ric_db, 60.0) is None
    assert a1_policy.current_schedule(ric_db) == []


def test_schedule_write_is_atomic(ric_db) -> None:
    """A failure in the middle leaves the previous timetable in force, not half a new one."""
    a1_policy.write_schedule(ric_db, [_segment(0, 100, ("i0001",))], "apply i0001", 0.0)
    broken = a1_policy.Segment(None, ("i0002",), {}, "console", 0.0, 100.0)  # type: ignore[arg-type]
    with pytest.raises(AttributeError):
        a1_policy.write_schedule(ric_db, [_segment(0, 100, ("i0002",)), broken], "apply i0002", 0.0)
    assert a1_policy.in_force(ric_db, 1.0).intent_ids == ("i0001",)


# ------------------------------------------------------------------ intents table


def test_intent_records_keep_their_history(ric_db) -> None:
    first = intents.create(ric_db, "a", now_wall=100.0, issued_backend=5.0)
    second = intents.create(ric_db, "b", now_wall=101.0, issued_backend=6.0)
    assert (first.intent_id, second.intent_id) == ("i0001", "i0002")
    rec = intents.set_state(ric_db, "i0001", "rejected", 102.0, reason="parser: not understood")
    assert rec.state_times == {"received": 100.0, "rejected": 102.0}
    with pytest.raises(ValueError):
        intents.set_state(ric_db, "i0001", "done", 103.0)
    assert [r.intent_id for r in intents.in_states(ric_db, ("received",))] == ["i0002"]
