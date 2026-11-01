"""M02c: tracking never changes training, backfill is idempotent, the registry gate decides."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import CallbackList

from mlops import backfill, registry, tracking
from rl import train as train_module
from rl.train import TrackingCallback, make_model
from sim.env import SlicingEnv
from verifier.objective import MIN_SLICE_SHARE


@pytest.fixture(autouse=True)
def temporary_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own MLflow database and artifact folder."""
    import mlflow

    monkeypatch.setenv("RIC_MLFLOW_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("RIC_MLFLOW_ARTIFACTS", str(tmp_path / "artifacts"))
    yield tmp_path
    while mlflow.active_run() is not None:
        mlflow.end_run()


def _tiny_td3(path: Path, seed: int = 0) -> Path:
    model = make_model(SlicingEnv(episode_steps=50), seed=seed, algo="td3")
    model.learning_starts = 100
    model.learn(total_timesteps=200)
    model.save(path)
    return path


def test_training_run_records_params_git_metrics_and_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow

    monkeypatch.setattr(train_module, "TRACK_EVERY", 100)
    monkeypatch.chdir(tmp_path)
    train_module.train(seed=3, timesteps=400, threads=1, algo="td3", episode_steps=50, group="m02")
    run = tracking.find_run("m02-round1", model_name="td3_seed3")
    assert run is not None and run.info.status == "FINISHED"
    assert run.data.params["seed"] == "3" and "git_sha" in run.data.tags
    history = mlflow.MlflowClient().get_metric_history(run.info.run_id, "train/reward_per_step_last100")
    assert [m.step for m in history] == [100, 200, 300, 400]
    files = [f.path for f in mlflow.MlflowClient().list_artifacts(run.info.run_id, "files")]
    assert "files/td3_seed3.zip" in files and "files/monitor.csv" in files


def test_tracking_does_not_change_training() -> None:
    """Same seed, with and without tracking: identical actor weights."""
    torch.set_num_threads(1)

    def weights(tracked: bool) -> list[np.ndarray]:
        model = make_model(SlicingEnv(episode_steps=50), seed=7, algo="td3")
        model.learning_starts = 100
        with tracking.training_run("td3_same_seed7", "m02", {"seed": 7}, enabled=tracked):
            model.learn(total_timesteps=300, callback=CallbackList([TrackingCallback(every=50)]))
        return [p.detach().numpy().copy() for p in model.actor.parameters()]

    for a, b in zip(weights(True), weights(False)):
        assert np.array_equal(a, b)


def test_tracking_failure_does_not_stop_training(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIC_MLFLOW_URI", "sqlite:////proc/no_such_dir/mlflow.db")  # cannot be created
    model = make_model(SlicingEnv(episode_steps=50), seed=0, algo="td3")
    model.learning_starts = 100
    with pytest.warns(UserWarning, match="tracking failed"):
        with tracking.training_run("td3_fail_seed0", "m02", {"seed": 0}) as run:
            model.learn(total_timesteps=200, callback=CallbackList([TrackingCallback(every=50)]))
    assert run is None and model.num_timesteps == 200


def test_backfill_imports_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models/m02").mkdir(parents=True)
    (tmp_path / "models/m02/td3_seed0.json").write_text(json.dumps({"algo": "td3", "seed": 0}))
    run_dir = tmp_path / "runs/m02/td3_seed0"
    run_dir.mkdir(parents=True)
    rows = "\n".join(f"{10.0 + i},{200},{i}" for i in range(30))
    (run_dir / "monitor.csv").write_text('#{"t_start": 1700000000.0, "env_id": "x"}\nr,l,t\n' + rows + "\n")
    backfill.main()
    backfill.main()  # second time: nothing new
    import mlflow

    exp = mlflow.get_experiment_by_name("m02-round1")
    runs = mlflow.search_runs([exp.experiment_id], output_format="list")
    assert len(runs) == 1 and runs[0].data.tags["backfilled"] == "True"
    history = mlflow.MlflowClient().get_metric_history(runs[0].info.run_id, "train/reward_per_step_10ep")
    assert len(history) == 3 and history[0].step == 2000


def test_registered_model_enforces_the_observation_signature(tmp_path: Path) -> None:
    import mlflow

    version = registry.register(_tiny_td3(tmp_path / "td3_tiny_seed0.zip"))
    model = mlflow.pyfunc.load_model(f"models:/{registry.MODEL_NAME}/{version}")
    shares = model.predict(np.zeros((2, 22), dtype=np.float32))
    assert shares.shape == (2, 3) and np.allclose(shares.sum(axis=1), 1.0)
    with pytest.raises(Exception):
        model.predict(np.zeros((1, 21), dtype=np.float32))


def test_gate_needs_a_held_out_pass_and_rollback_restores(tmp_path: Path) -> None:
    floor = str(MIN_SLICE_SHARE)
    v1 = registry.register(_tiny_td3(tmp_path / "td3_old_seed0.zip"), {"promotion_basis": "bootstrap"})
    registry.client().set_registered_model_alias(registry.MODEL_NAME, registry.CHAMPION, v1)
    path = _tiny_td3(tmp_path / "td3_new_seed1_best.zip", seed=1)
    v2 = registry.register(path, {"validation_t1b": "0.7", "validation_reward": "0.26", "min_share": floor})

    assert registry.promote() is False  # no held-out goal test recorded yet
    assert str(registry.alias_version(registry.CHAMPION).version) == v1

    tracking.log_evaluation("goal-test", "td3_new_seed1_best", {"t1b_captured": 0.8, "pass": False},
                            {"split": "test", "final": "true", "recipe_all_pass": "False", "min_share": floor})
    assert registry.promote() is False  # recorded, but FAIL

    tracking.log_evaluation("goal-test", "td3_new_seed1_best", {"t1b_captured": 0.8, "pass": True},
                            {"split": "test", "final": "true", "recipe_all_pass": "True", "min_share": floor})
    assert registry.promote() is True
    assert str(registry.alias_version(registry.CHAMPION).version) == v2

    assert registry.rollback() is True
    assert str(registry.alias_version(registry.CHAMPION).version) == v1


def test_gate_refuses_a_version_trained_without_the_floor(tmp_path: Path) -> None:
    """A version that passed everything but was trained (and tested) without the per-slice floor
    the runner now enforces cannot be promoted (M04a); it can still be a rollback target."""
    old = _tiny_td3(tmp_path / "td3_nofloor_seed2_best.zip", seed=2)
    (tmp_path / "td3_nofloor_seed2.json").write_text(json.dumps({"min_share": None}))
    v = registry.register(old, {"validation_t1b": "0.99", "validation_reward": "0.3"})
    assert registry.versions()[-1].tags["min_share"] == "None"
    tracking.log_evaluation("goal-test", "td3_nofloor_seed2_best", {"t1b_captured": 0.9, "pass": True},
                            {"split": "test", "final": "true", "recipe_all_pass": "True", "min_share": "None"})
    chosen, reasons = registry.gate(registry.versions())
    assert chosen is None and "not trained with the deployed per-slice floor" in reasons[0]
    assert v == str(registry.versions()[-1].version)  # still registered: a rollback target


def test_champion_is_none_without_a_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RIC_MLFLOW_URI", f"sqlite:///{tmp_path}/absent/mlflow.db")
    assert registry.champion() is None
    assert not (tmp_path / "absent").exists()
