"""Model registry for the RL policy, and the gate that decides which version is deployed.

Run with:
  uv run python -m mlops.registry bootstrap               register today's deployed policy as version 1
  uv run python -m mlops.registry register "<glob>"       register trained models as new versions
  uv run python -m mlops.registry promote --auto          move `champion` if the gate allows it
  uv run python -m mlops.registry rollback                move `champion` back to the previous version
  uv run python -m mlops.registry status                  list versions, aliases and their evidence

Vocabulary. The registered model `slice-policy` has numbered *versions* (one per trained model
we consider). An *alias* is a movable name on one version: `champion` is the deployed policy,
the one the RL runner (M04a) will load as models:/slice-policy@champion. Deploying a new policy,
or rolling back, is moving that alias: nothing is copied or retrained.

The gate is plain deterministic Python, like the verifier: no judgement call. `promote --auto`
moves `champion` to a version only if
  0. it was trained with the actuator rules deployed today: the per-slice PRB floor
     (tag min_share = MIN_SLICE_SHARE, read from the model's settings at registration; M04a).
     Older versions stay registered and can still be rolled back to, but not promoted;
  1. the held-out goal test (rl/goal_test.py --split test --final) recorded PASS for it,
  2. that test recorded that its recipe passed on every confirmation seed, and
  3. it is the version the deployment rule picks among those: highest validation T1b, then
     validation reward (rule fixed in M02b before any result).
Otherwise it refuses and prints why. The evidence is read from the MLflow evaluation runs, not
typed in.

Each version stores the SB3 model file as an MLflow "pyfunc" model with a *signature*: input
22 float32 values (the observation), output 3 PRB shares. The signature documents the
observation contract every backend must respect. Its predict() is the raw actor (stateless);
the deployed smoothing filter keeps state, so the runner builds SlicePolicy from the file.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec

from mlops import tracking
from verifier.objective import MIN_SLICE_SHARE

MODEL_NAME = "slice-policy"
CHAMPION = "champion"
CANDIDATE = "candidate"
REGISTRY_EXPERIMENT = "registry"
GOAL_TEST_EXPERIMENT = "goal-test"
BOOTSTRAP_MODEL = Path("models/m02/td3_seed2.zip")
BOOTSTRAP_BASIS = "M02 fallback rule (option A, rl/select.py); goal-conditioning criterion FAIL"


class SlicePolicyModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper: observations (n, 22) -> raw PRB shares (n, 3)."""

    def load_context(self, context: Any) -> None:
        from rl.policy import SlicePolicy

        self.policy = SlicePolicy.load(context.artifacts["sb3_zip"], alpha=1.0)  # no filter: stateless

    def predict(self, context: Any, model_input: np.ndarray, params: dict | None = None) -> np.ndarray:
        from sim.cell import shares_from_action

        obs = np.asarray(model_input, dtype=np.float32).reshape(-1, 22)
        return np.array([shares_from_action(self.policy.raw_action(o)) for o in obs])


def signature() -> ModelSignature:
    # Unnamed tensors: predict() then receives a plain (n, 22) array, not a dict of named inputs.
    return ModelSignature(inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, 22))]),
                          outputs=Schema([TensorSpec(np.dtype(np.float64), (-1, 3))]))


def client() -> MlflowClient:
    tracking.setup()
    return MlflowClient()


def best_validation(model_path: Path) -> dict[str, float]:
    """Validation numbers of the kept checkpoint (from <name>_validation.json), if any."""
    stem = model_path.stem.removesuffix("_best")
    history_path = model_path.with_name(f"{stem}_validation.json")
    if not history_path.exists():
        return {}
    eligible = [m for m in json.loads(history_path.read_text()) if m["t2_pass"]]
    if not eligible:
        return {}
    best = max(eligible, key=lambda m: (m["t1b_captured"], m["reward"]))  # the ValidationCheckpoint rule
    return {"validation_t1b": best["t1b_captured"], "validation_t1a": best["t1a_share_positive"],
            "validation_reward": best["reward"], "validation_step": best["step"]}


def trained_min_share(model_path: Path) -> str:
    """The per-slice floor the model was trained with, from its settings file ('None' = no floor)."""
    settings = model_path.with_name(f"{model_path.stem.removesuffix('_best')}.json")
    if not settings.exists():
        return "None"
    return str(json.loads(settings.read_text()).get("min_share"))


def register(model_path: Path, extra_tags: dict[str, str] | None = None) -> str:
    """Log a model file as a pyfunc model and register it; returns the new version number."""
    model_name = model_path.stem
    tags = {"model_name": model_name, "source_file": str(model_path), "min_share": trained_min_share(model_path),
            **{k: str(v) for k, v in best_validation(model_path).items()}, **(extra_tags or {})}
    training = tracking.training_run_of(model_name.removesuffix("_best"))
    if training is not None:
        tags["training_run_id"] = training.info.run_id
    tracking.use_experiment(REGISTRY_EXPERIMENT)
    with mlflow.start_run(run_name=f"register {model_name}", tags={"model_name": model_name, **tracking.git_state()}):
        info = mlflow.pyfunc.log_model(
            name="policy", python_model=SlicePolicyModel(), artifacts={"sb3_zip": str(model_path)},
            signature=signature(), input_example=np.zeros((1, 22), dtype=np.float32),
            registered_model_name=MODEL_NAME,
        )
    version = str(info.registered_model_version)
    c = client()
    for key, value in tags.items():
        c.set_model_version_tag(MODEL_NAME, version, key, value)
    return version


def versions() -> list[Any]:
    c = client()
    try:
        return sorted(c.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version))
    except Exception:  # noqa: BLE001 (no registered model yet)
        return []


def alias_version(alias: str) -> Any:
    try:
        return client().get_model_version_by_alias(MODEL_NAME, alias)
    except Exception:  # noqa: BLE001 (alias not set)
        return None


def bootstrap() -> None:
    """Register the policy deployed today (td3_seed2) as version 1 and champion, with its basis."""
    if versions():
        print("registry already has versions; nothing to do")
        return
    version = register(BOOTSTRAP_MODEL, {"promotion_basis": BOOTSTRAP_BASIS})
    client().set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    print(f"registered {BOOTSTRAP_MODEL} as version {version}, alias {CHAMPION}")


def final_goal_test(model_name: str) -> Any:
    """The held-out goal test record of a model (run tagged final=true), or None."""
    return tracking.find_run(GOAL_TEST_EXPERIMENT, model_name=model_name, final="true")


def gate(candidates: list[Any]) -> tuple[Any | None, list[str]]:
    """Apply the three checks; returns (the version to promote or None, the reasons)."""
    reasons, passing = [], []
    for v in candidates:
        name = v.tags.get("model_name", "?")
        record = final_goal_test(name)
        if v.tags.get("min_share") != str(MIN_SLICE_SHARE):
            reasons.append(f"v{v.version} {name}: not trained with the deployed per-slice floor "
                           f"(min_share {v.tags.get('min_share', 'None')}, deployed {MIN_SLICE_SHARE})")
        elif record is None:
            reasons.append(f"v{v.version} {name}: no held-out goal test recorded")
        elif record.data.tags.get("min_share") != str(MIN_SLICE_SHARE):
            reasons.append(f"v{v.version} {name}: its held-out goal test did not run with the per-slice floor")
        elif record.data.tags.get("pass") != "True":
            reasons.append(f"v{v.version} {name}: held-out goal test FAIL")
        elif record.data.tags.get("recipe_all_pass") != "True":
            reasons.append(f"v{v.version} {name}: its recipe did not pass on every confirmation seed")
        elif "validation_t1b" not in v.tags:
            reasons.append(f"v{v.version} {name}: no validation numbers")
        else:
            passing.append(v)
    if not passing:
        return None, reasons
    chosen = max(passing, key=lambda v: (float(v.tags["validation_t1b"]), float(v.tags["validation_reward"])))
    reasons.append(f"v{chosen.version} {chosen.tags['model_name']}: passes, and the deployment rule picks it "
                   f"(validation T1b {float(chosen.tags['validation_t1b']):.3f})")
    return chosen, reasons


def promote() -> bool:
    """Move champion to the version the gate picks; returns True if it moved."""
    current = alias_version(CHAMPION)
    chosen, reasons = gate([v for v in versions() if current is None or str(v.version) != str(current.version)])
    for reason in reasons:
        print("  " + reason)
    if chosen is None:
        print("gate: REFUSED, champion unchanged")
        return False
    c = client()
    if current is not None:
        c.set_model_version_tag(MODEL_NAME, chosen.version, "previous_champion", str(current.version))
    c.set_model_version_tag(MODEL_NAME, chosen.version, "promotion_basis",
                            f"gate: held-out goal test PASS, trained with min_share {MIN_SLICE_SHARE}")
    c.set_registered_model_alias(MODEL_NAME, CHAMPION, chosen.version)
    print(f"gate: champion -> version {chosen.version} ({chosen.tags['model_name']})")
    return True


def rollback() -> bool:
    current = alias_version(CHAMPION)
    previous = current.tags.get("previous_champion") if current is not None else None
    if previous is None:
        print("no previous champion recorded; nothing to roll back to")
        return False
    client().set_registered_model_alias(MODEL_NAME, CHAMPION, previous)
    print(f"champion rolled back: version {current.version} -> {previous}")
    return True


def champion() -> tuple[str, Path] | None:
    """(model name, local path of its SB3 file) of the deployed version, or None.

    Returns None without creating anything when no local MLflow store exists (fresh clone, CI).
    """
    uri = tracking.tracking_uri()
    if uri.startswith("sqlite:///") and not Path(uri.removeprefix("sqlite:///")).exists():
        return None
    version = alias_version(CHAMPION)
    if version is None:
        return None
    local = Path(mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}@{CHAMPION}"))
    zips = sorted((local / "artifacts").glob("*.zip"))
    return (version.tags.get("model_name", "?"), zips[0]) if zips else None


def status() -> None:
    aliases = {a: alias_version(a) for a in (CHAMPION, CANDIDATE)}
    for v in versions():
        names = [a for a, av in aliases.items() if av is not None and str(av.version) == str(v.version)]
        record = final_goal_test(v.tags.get("model_name", ""))
        verdict = "-" if record is None else ("PASS" if record.data.tags.get("pass") == "True" else "FAIL")
        print(f"v{v.version:<3} {v.tags.get('model_name', '?'):<28} held-out test {verdict:<5} "
              f"val T1b {v.tags.get('validation_t1b', '-')[:5]:<6} {' '.join(names)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("bootstrap")
    reg = sub.add_parser("register")
    reg.add_argument("pattern", help='glob of model files, e.g. "models/m02b/confirm/*_best.zip"')
    pro = sub.add_parser("promote")
    pro.add_argument("--auto", action="store_true", required=True, help="apply the gate and the deployment rule")
    sub.add_parser("rollback")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "bootstrap":
        bootstrap()
    elif args.command == "register":
        for path in sorted(Path().glob(args.pattern)):
            print(f"{path} -> version {register(path)}")
    elif args.command == "promote":
        promote()
    elif args.command == "rollback":
        rollback()
    else:
        status()


if __name__ == "__main__":
    main()
