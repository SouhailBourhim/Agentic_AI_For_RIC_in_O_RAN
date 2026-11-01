"""Import the training runs made before MLflow existed (M02 rounds 1-2, M02b dev round).

Run with:  uv run python -m mlops.backfill

Each run is rebuilt from the files training already wrote: the settings JSON next to the model,
monitor.csv (one line per episode), and, for M02b, the validation history. The M02 evaluation
(runs/m02/eval.json, reward v1) is imported as one evaluation run per policy.
Running it twice changes nothing: every imported run carries a `source_path` tag and is skipped
if it already exists.

What is inferred, and said so: the git commit of an old run is the last commit before its model
file was written (tag `git_sha_inferred`), not a recorded fact (`git_sha` in live runs).
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mlops import tracking

TRAINING_GROUPS = ("m02", "m02b/dev")
EVAL_PATH = Path("runs/m02/eval.json")
EPISODES_PER_POINT = 10


def inferred_commit(path: Path) -> str:
    """Last commit made before the file's modification time."""
    when = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    out = subprocess.run(["git", "rev-list", "-1", f"--before={when}", "HEAD"],
                         capture_output=True, text=True, check=False).stdout.strip()
    return out or "unknown"


def curve_from_monitor(path: Path) -> tuple[list[int], list[float], float]:
    """(env step, mean reward per step over EPISODES_PER_POINT episodes) and the start time."""
    header = json.loads(path.read_text().splitlines()[0].lstrip("#"))
    data = np.loadtxt(path, delimiter=",", skiprows=2, usecols=(0, 1), ndmin=2)
    rewards, lengths = data[:, 0], data[:, 1]
    ends = np.cumsum(lengths).astype(int)
    steps, values = [], []
    for i in range(0, len(rewards) - EPISODES_PER_POINT + 1, EPISODES_PER_POINT):
        chunk = slice(i, i + EPISODES_PER_POINT)
        steps.append(int(ends[chunk][-1]))
        values.append(float(rewards[chunk].sum() / lengths[chunk].sum()))
    return steps, values, float(header["t_start"])


def import_training_run(client, settings_path: Path, group: str) -> str | None:
    """Create one finished MLflow run from a model's files; returns its id (None if skipped)."""
    from mlflow.entities import Metric

    name = settings_path.stem
    source = str(settings_path)
    experiment = tracking.experiment_for(group, name)
    if tracking.find_run(experiment, source_path=source) is not None:
        return None
    settings = json.loads(settings_path.read_text())
    monitor = Path("runs") / group / name / "monitor.csv"
    steps, values, t_start = curve_from_monitor(monitor) if monitor.exists() else ([], [], time.time())
    experiment_id = tracking.use_experiment(experiment)
    tags = {**tracking.run_tags(name, settings), "source_path": source, "backfilled": "True",
            "git_sha_inferred": inferred_commit(settings_path), "mlflow.runName": name}
    run = client.create_run(experiment_id, start_time=int(t_start * 1000), tags=tags)
    run_id = run.info.run_id
    for key, value in settings.items():
        client.log_param(run_id, key, str(value))
    now = int(time.time() * 1000)
    metrics = [Metric("train/reward_per_step_10ep", v, now, s) for s, v in zip(steps, values)]
    history_path = settings_path.with_name(f"{name}_validation.json")
    if history_path.exists():
        for m in json.loads(history_path.read_text()):
            for key in ("t1a_share_positive", "t1b_captured", "reward"):
                metrics.append(Metric(f"validation/{key}", float(m[key]), now, int(m["step"])))
    for i in range(0, len(metrics), 1000):  # MLflow accepts at most 1000 metrics per batch
        client.log_batch(run_id, metrics=metrics[i:i + 1000])
    for path in (settings_path, settings_path.with_suffix(".zip"), monitor, history_path,
                 settings_path.with_name(f"{name}_best.zip")):
        if path.exists():
            client.log_artifact(run_id, str(path), artifact_path="files")
    client.set_terminated(run_id, "FINISHED", end_time=int(settings_path.stat().st_mtime * 1000))
    return run_id


def import_m02_evaluation() -> int:
    """One run per policy in experiment m02-eval, from runs/m02/eval.json (reward v1)."""
    if not EVAL_PATH.exists() or tracking.find_run("m02-eval", source_path=str(EVAL_PATH)) is not None:
        return 0
    results = json.loads(EVAL_PATH.read_text())
    gaps = results["goal_conditioning"]
    criteria = results["criteria"]
    count = 0
    for name, metrics in results["random"].items():
        if name in ("equal split", "proportional to load"):
            continue
        values = {f"{k}_mean": float(np.nanmean(v)) for k, v in metrics.items()}
        if name in gaps:
            values["urllc_share_gap"] = gaps[name]["protect URLLC"]["share_urllc"] - gaps[name]["maximise eMBB"]["share_urllc"]
        model = name.removesuffix(" raw")
        variant = "raw" if name.endswith(" raw") else "filtered"
        algo = model.split("_seed")[0]
        tags = {"split": "m02-eval", "variant": variant, "reward_version": "v1", "source_path": str(EVAL_PATH),
                "backfilled": "True", "gap_criterion_group_pass": str(criteria["goal_conditioning_share_gap"]["pass"][algo]),
                "judged_variant": criteria.get("variant", {}).get(algo, "filtered")}
        tracking.log_evaluation("m02-eval", model, values, tags, record_git=False)
        count += 1
    return count


def main() -> None:
    from mlflow import MlflowClient

    tracking.setup()
    client = MlflowClient()
    for group in TRAINING_GROUPS:
        settings_files = sorted(p for p in (Path("models") / group).glob("*_seed*.json")
                                if not p.stem.endswith("_validation"))
        imported = [import_training_run(client, p, group) for p in settings_files]
        print(f"{group}: {sum(r is not None for r in imported)} imported, "
              f"{sum(r is None for r in imported)} already there")
    print(f"m02-eval: {import_m02_evaluation()} evaluation runs imported")


if __name__ == "__main__":
    main()
