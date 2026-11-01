"""Shared fixtures for the M04a plumbing tests: a temporary ric.db and event log, a tiny policy,
and a simulated cell with its runner.

The tiny policy is a TD3 model trained for 200 steps: it is not good, but it has the real
observation and action shapes, so the plumbing runs exactly as with the champion. CI has no
MLflow store, so tests never depend on the registry.
"""

import sqlite3
from pathlib import Path

import pytest

from backends.sim import SimBackend, scenario_traffic
from rl.policy import SlicePolicy
from rl.runner import Runner
from rl.train import make_model
from sim.cell import CellConfig
from sim.env import SlicingEnv
from store import db, kpi


@pytest.fixture(scope="session")
def tiny_policy_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("policy") / "td3_tiny_seed0.zip"  # "td3" prefix: SlicePolicy.load picks TD3
    model = make_model(SlicingEnv(episode_steps=50), seed=0, algo="td3")
    model.learning_starts = 100
    model.learn(total_timesteps=200)
    model.save(path)
    return path


@pytest.fixture
def ric_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> sqlite3.Connection:
    """A fresh var/ric.db and events file for one test, used by every module through RIC_DB."""
    monkeypatch.setenv("RIC_DB", str(tmp_path / "ric.db"))
    monkeypatch.setenv("RIC_EVENTS", str(tmp_path / "events.jsonl"))
    conn = db.connect()
    yield conn
    conn.close()


def start_cell(conn: sqlite3.Connection, policy_file: Path, scenario: str = "demo", seed: int = 0) -> Runner:
    """A simulated cell at speed 0 with its runner, as `python -m rl.runner` would set them up."""
    cfg = CellConfig()
    backend = SimBackend(cfg, scenario_traffic(scenario, cfg), seed=seed, speed=0.0)
    kpi.write_backend_info(conn, backend.info(), "tiny test policy")
    return Runner(backend, SlicePolicy.load(policy_file), db.connect(timeout_s=0.0), cfg.dt_s)


def run_agent(conn: sqlite3.Connection, text: str, mode: str = "demo", now_wall: float = 0.0, llm=None,
              runner: Runner | None = None, **settings):
    """Submit an intent and run the agent on it, in-process. Defaults reproduce M04a's behaviour:
    rule-based agent, no observation, no revision, stop at awaiting_confirmation in demo mode.
    With a runner, backend time passes by running it (wait_and_observe)."""
    from agent.tools import AgentContext, Settings
    from agent.worker import submit_and_run

    options = {"mode": mode, "observe_s": 0.0, "max_revisions": 0, "wait_for_confirmation": False, **settings}
    ctx = AgentContext(conn=conn, settings=Settings(**options), llm=llm, now_wall=lambda: now_wall,
                       wait_backend=(lambda s: runner.run(int(s))) if runner is not None else None)
    return submit_and_run(ctx, text)
