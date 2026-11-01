"""Experiment tracking with local MLflow.

Everything is stored on this machine: runs and metrics in a SQLite file, files (models, CSVs,
JSON results) in an artifact folder. Look at it with:

    uv run mlflow ui --backend-store-uri sqlite:///var/mlflow.db      (then localhost:5000)

The rule that matters: tracking must never change or stop what it tracks. Every MLflow call
made during training goes through `safe()`: an error prints a warning and training goes on.
Tracking only reads values that training computes anyway; it draws no random numbers.

Where things are stored is read from RIC_MLFLOW_URI (default sqlite:///var/mlflow.db), with
artifacts in RIC_MLFLOW_ARTIFACTS (default var/mlartifacts), so tests can use a temporary place.
"""

import os
import subprocess
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_URI = "sqlite:///var/mlflow.db"
DEFAULT_ARTIFACTS = "var/mlartifacts"


def tracking_uri() -> str:
    return os.environ.get("RIC_MLFLOW_URI", DEFAULT_URI)


def artifact_root() -> Path:
    return Path(os.environ.get("RIC_MLFLOW_ARTIFACTS", DEFAULT_ARTIFACTS)).resolve()


def setup() -> None:
    """Point MLflow at our store (creating var/ if needed)."""
    import mlflow

    uri = tracking_uri()
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)


def use_experiment(name: str) -> str:
    """Select (or create) an experiment whose files go under the artifact root; returns its id."""
    import mlflow

    setup()
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        location = (artifact_root() / name).as_uri()
        mlflow.create_experiment(name, artifact_location=location)
    return mlflow.set_experiment(name).experiment_id


def safe(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call an MLflow function; on any error warn and return None instead of raising."""
    try:
        return fn(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 (tracking must never stop training)
        warnings.warn(f"MLflow tracking failed, training continues: {error!r}", stacklevel=2)
        return None


def git_state() -> dict[str, str]:
    """The exact code a run used: commit and whether there were uncommitted changes."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()

    return {"git_sha": git("rev-parse", "HEAD") or "unknown",
            "git_dirty": str(bool(git("status", "--porcelain", "--untracked-files=no")))}


def experiment_for(group: str, name: str) -> str:
    """Experiment name from the training group and model name (one experiment per round)."""
    if group == "m02":  # the M02 rounds shared one folder; the model name tells them apart
        return "m02-round2" if "v2" in name else "m02-round1"
    # Every later round has its own group: m02b/dev -> m02b-dev, m04a_floor/confirm -> m04a_floor-confirm.
    # (Before this fix, groups other than m02b fell back to m02-round1: the five M04a floor runs,
    # seeds 20-24, are filed there; their params and tags identify them.)
    return group.replace("/", "-")


def run_tags(name: str, settings: dict[str, Any]) -> dict[str, str]:
    """Tags shared by live and backfilled training runs."""
    recipe = name.split("_seed")[0]
    return {"model_name": name, "recipe": recipe, "seed": str(settings.get("seed")),
            "algo": str(settings.get("algo"))}


@contextmanager
def training_run(name: str, group: str, settings: dict[str, Any], enabled: bool = True) -> Iterator[Any]:
    """Open an MLflow run around one training; yields the active run, or None if disabled/failed."""
    if not enabled:
        yield None
        return
    import mlflow

    run = None
    if safe(use_experiment, experiment_for(group, name)) is not None:
        run = safe(mlflow.start_run, run_name=name, tags={**run_tags(name, settings), **git_state()})
    if run is not None:
        safe(mlflow.log_params, {k: str(v) for k, v in settings.items()})
    try:
        yield run
        if run is not None:
            safe(mlflow.end_run, "FINISHED")
    except BaseException:
        if run is not None:
            safe(mlflow.end_run, "FAILED")
        raise


def log_metrics(metrics: dict[str, float], step: int) -> None:
    import mlflow

    if mlflow.active_run() is not None:
        safe(mlflow.log_metrics, metrics, step=step)


def log_artifacts(paths: list[Path], subdir: str | None = None) -> None:
    import mlflow

    if mlflow.active_run() is None:
        return
    for path in paths:
        if path.exists():
            safe(mlflow.log_artifact, str(path), artifact_path=subdir)


def find_run(experiment: str, **tags: str) -> Any:
    """The most recent run of an experiment with these tag values, or None."""
    import mlflow

    setup()
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return None
    query = " and ".join(f"tags.{k} = '{v}'" for k, v in tags.items())
    runs = mlflow.search_runs([exp.experiment_id], filter_string=query, output_format="list",
                              order_by=["attributes.start_time DESC"], max_results=1)
    return runs[0] if runs else None


TRAINING_EXPERIMENTS = ("m04a_floor-confirm", "m02b-confirm", "m02b-dev", "m02-round2", "m02-round1")


def training_run_of(model_name: str) -> Any:
    """The training run that produced a model, searched across the training experiments."""
    for experiment in TRAINING_EXPERIMENTS:
        run = find_run(experiment, model_name=model_name)
        if run is not None:
            return run
    return None


def log_evaluation(experiment: str, model_name: str, metrics: dict[str, Any], tags: dict[str, str],
                   result_file: Path | None = None, record_git: bool = True) -> None:
    """One evaluation run for one model: numbers as metrics, verdicts as tags, linked to training.

    record_git=False for evaluations imported from old files, whose code version is unknown.
    """
    import mlflow

    if safe(use_experiment, experiment) is None:
        return
    numbers = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float) and not isinstance(v, bool)}
    flags = {k: str(v) for k, v in metrics.items() if isinstance(v, bool)}
    training = safe(training_run_of, model_name)
    link = {"training_run_id": training.info.run_id} if training is not None else {}
    run = safe(mlflow.start_run, run_name=f"{model_name} {tags.get('split', '')}".strip(),
               tags={"model_name": model_name, **tags, **flags, **link, **(git_state() if record_git else {})})
    if run is None:
        return
    safe(mlflow.log_metrics, numbers)
    if result_file is not None and result_file.exists():
        safe(mlflow.log_artifact, str(result_file))
    safe(mlflow.end_run, "FINISHED")
