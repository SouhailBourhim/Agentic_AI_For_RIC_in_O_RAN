"""M11a: the runner's safe state. Stale KPIs, a dead actuator or an exception put the equal split on the cell at
once; three clean ticks in a row give the policy back. On the simulator it never happens.

The faults are injected by backends.faults.FaultyBackend, the simulated cell with chosen seconds of trouble.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest

from agent.service import app
from backends.faults import FaultyBackend
from backends.sim import SimBackend, scenario_traffic
from monitoring.exporter import RicCollector
from rl.policy import SlicePolicy
from rl.runner import RESUME_AFTER_TICKS, Runner
from sim.cell import CellConfig
from store import a1_policy, db, kpi
from store.events import read_events
from verifier.objective import RLObjective, SliceValues

EQUAL = pytest.approx([1 / 3, 1 / 3, 1 / 3])


class ApplyFailsBackend(FaultyBackend):
    """A pipe to the xApp that breaks at chosen seconds. Here, not in backends/faults.py: a call to apply_prb_split
    outside rl/runner.py breaks invariant 4's check, which skips tests/."""

    def __init__(self, apply_error=()) -> None:
        super().__init__()
        self.apply_error = set(apply_error)

    def apply_prb_split(self, shares: list[float]) -> None:
        if self.second() in self.apply_error:
            raise OSError("broken pipe to the xApp")
        super().apply_prb_split(shares)


class FixedPolicy:
    """Always the same uneven split, so a safe tick (equal split) is easy to tell apart. Raises at chosen calls."""

    def __init__(self, raise_at=()) -> None:
        self.calls = 0
        self.raise_at = set(raise_at)

    def shares(self, _obs: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.calls in self.raise_at:
            raise RuntimeError("boom")
        return np.array([0.6, 0.3, 0.1])


def runner(backend, policy=None, on_error: str = "safe") -> Runner:
    return Runner(backend, policy or FixedPolicy(), db.connect(timeout_s=0.0), on_error=on_error)


def safe_events() -> list[tuple[str, str | None]]:
    return [(e["event"], e["data"]["reason"]) for e in read_events() if e["event"].startswith("safe_mode")]


def test_stale_kpis_give_the_equal_split_until_three_clean_ticks(ric_db) -> None:
    r = runner(FaultyBackend(stale={3, 4}))
    results = [r.tick() for _ in range(9)]
    assert [x.mode for x in results] == ["normal"] * 3 + ["safe"] * 4 + ["normal"] * 2
    assert RESUME_AFTER_TICKS == 3  # ticks 5 and 6 are clean but still safe; tick 7 is the third: the policy is back
    assert all(x.shares == EQUAL for x in results[3:7])
    assert results[7].shares == pytest.approx([0.6, 0.3, 0.1])
    assert safe_events() == [("safe_mode_enter", "kpi_stale"), ("safe_mode_exit", "kpi_stale")]
    exit_event = next(e for e in read_events() if e["event"] == "safe_mode_exit")
    assert exit_event["ts_backend"] == 7.0 and exit_event["data"]["safe_s"] == 4.0


def test_the_runner_state_row_follows_the_mode(ric_db) -> None:
    r = runner(FaultyBackend(stale={2}))
    r.tick()
    assert kpi.read_runner_state(ric_db) == {"mode": "normal", "reason": None, "since_backend": 0.0}
    r.tick()
    r.tick()
    assert kpi.read_runner_state(ric_db) == {"mode": "safe", "reason": "kpi_stale", "since_backend": 2.0}
    r.run(RESUME_AFTER_TICKS)
    assert kpi.read_runner_state(ric_db)["mode"] == "normal"


def test_a_dead_actuator_is_reported_before_stale_kpis(ric_db) -> None:
    r = runner(FaultyBackend(stale={1}, down={1, 2}))
    r.run(3)
    assert safe_events() == [("safe_mode_enter", "actuator_down")]


def test_a_new_reason_while_safe_is_logged_and_shown(ric_db) -> None:
    r = runner(FaultyBackend(stale={1}, down={2}))
    r.run(3)
    assert safe_events() == [("safe_mode_enter", "kpi_stale"), ("safe_mode_enter", "actuator_down")]
    state = kpi.read_runner_state(ric_db)
    assert state["reason"] == "actuator_down" and state["since_backend"] == 1.0  # safe since the first problem


def test_a_policy_exception_does_not_stop_the_loop(ric_db) -> None:
    r = runner(FaultyBackend(), FixedPolicy(raise_at={2}))
    results = [r.tick() for _ in range(3)]
    assert results[1].mode == "safe" and results[1].shares == EQUAL
    enter = next(e for e in read_events() if e["event"] == "safe_mode_enter")
    assert enter["data"]["reason"] == "error" and enter["data"]["error"] == "RuntimeError: boom"


def test_evaluation_runs_still_raise(ric_db) -> None:
    r = runner(FaultyBackend(), FixedPolicy(raise_at={1}), on_error="raise")
    with pytest.raises(RuntimeError, match="boom"):
        r.tick()
    with pytest.raises(ValueError):
        runner(FaultyBackend(), on_error="ignore")


def test_a_backend_read_exception_does_not_stop_the_loop(ric_db) -> None:
    r = runner(FaultyBackend(read_error={2}))
    results = [r.tick() for _ in range(4)]
    # Tick 1 ends at second 2: its KPI read fails and that tick's record is skipped. Tick 2's own read, before
    # deciding, fails too: that one puts the safe split on the cell.
    assert [x.mode for x in results] == ["normal", "normal", "safe", "safe"]
    assert safe_events() == [("safe_mode_enter", "error")]
    assert sorted({row[0] for row in ric_db.execute("SELECT ts FROM kpi")}) == [1.0, 3.0, 4.0]


def test_a_failed_apply_makes_the_next_tick_an_error(ric_db) -> None:
    r = runner(ApplyFailsBackend(apply_error={1}))
    results = [r.tick() for _ in range(3)]  # the apply at second 1 raises: the tick still returns
    assert [x.mode for x in results] == ["normal", "normal", "safe"]
    enter = next(e for e in read_events() if e["event"] == "safe_mode_enter")
    assert enter["ts_backend"] == 2.0 and enter["data"]["error"] == "OSError: broken pipe to the xApp"


def test_the_policy_filter_restarts_from_the_safe_split(ric_db, tiny_policy_file) -> None:
    """The first resumed split is one filter step away from the equal split, not the last pre-outage output."""
    policy = SlicePolicy.load(tiny_policy_file, alpha=0.1)
    r = runner(FaultyBackend(stale={2}), policy)
    results = [r.tick() for _ in range(2 + 1 + RESUME_AFTER_TICKS)]
    resumed = results[-1]
    assert resumed.mode == "normal" and results[-2].mode == "safe"
    assert max(abs(s - 1 / 3) for s in resumed.shares) <= 0.1 * (1 - 1 / 3) + 1e-6


def test_the_simulator_never_enters_the_safe_state(ric_db, tiny_policy_file) -> None:
    """Health is always good on SimBackend: every earlier simulator run plays the path it played before M11a."""
    cfg = CellConfig()
    backend = SimBackend(cfg, scenario_traffic("demo", cfg), seed=0, speed=0.0)
    r = Runner(backend, SlicePolicy.load(tiny_policy_file), db.connect(timeout_s=0.0))
    assert all(r.tick().mode == "normal" for _ in range(600))
    assert safe_events() == []


def test_status_and_prometheus_show_the_safe_state(ric_db) -> None:
    r = runner(FaultyBackend(down={1, 2, 3}))
    r.run(1)
    kpi.write_backend_info(ric_db, r.backend.info(), "fixed test policy")
    assert TestClient(app).get("/status").json()["runner"]["mode"] == "normal"
    r.run(1)
    assert TestClient(app).get("/status").json()["runner"] == {"mode": "safe", "reason": "actuator_down",
                                                              "since_backend": 1.0}
    registry = CollectorRegistry()
    registry.register(RicCollector(db.db_path()))
    assert 'ric_runner_safe_mode{reason="actuator_down"} 1.0' in generate_latest(registry).decode()


def test_a_corrupt_policy_row_keeps_the_previous_objective(ric_db) -> None:
    """M11c: a row whose objective fails the schema used to raise outside the tick and end the runner. Now the
    segment read before stays in force, as when the database can't be read, and the problem is logged once."""
    protect = RLObjective(weights=SliceValues(embb=0.2, urllc=0.6, be=0.2), min_thr_mbps=SliceValues(urllc=10.0),
                          valid_from=0.0, valid_until=100.0)
    a1_policy.write_schedule(ric_db, [a1_policy.Segment(protect, ("i0001",), {"accepted": True}, "console", 0.0, 100.0)],
                             "test", 0.0)
    r = runner(FaultyBackend())
    assert r.tick().policy_id == 1
    ric_db.execute("UPDATE a1_policy SET objective = ? WHERE policy_id = 1", ('{"weights": "not a table"}',))
    results = [r.tick() for _ in range(3)]
    assert all(x.policy_id == 1 and x.mode == "normal" and not x.db_ok for x in results)
    assert r.row is not None and r.row.objective == protect
    invalid = [e for e in read_events() if e["event"] == "a1_policy_invalid"]
    assert len(invalid) == 1 and invalid[0]["data"]["kept_policy_id"] == 1
    assert invalid[0]["data"]["error"].startswith("ValidationError")
