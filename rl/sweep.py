"""Launch the M02b training rounds: several rl.train runs in parallel, one log file each.

Run with:
  uv run python -m rl.sweep dev                       15 runs: 5 recipes x 3 seeds (models/m02b/dev/)
  uv run python -m rl.sweep confirm --recipe full     5 fresh seeds of one recipe (models/m02b/confirm/)
  uv run python -m rl.sweep m09c-dev                  M09c: 4 recipes x 3 seeds (models/m09c/dev/)
  uv run python -m rl.sweep m09c-confirm --recipe v8  M09c: 5 fresh seeds of the chosen recipe (models/m09c/confirm/)
  uv run python -m rl.sweep m09h-dev                  M09h: 6 v9 arms x 5 seeds, 15 at a time (models/m09h/dev/)
  uv run python -m rl.sweep m09h-median               M09h: each arm's median seed, the one the dev benchmark plays
  uv run python -m rl.sweep m09h-confirm --recipe v9  M09h: 5 fresh seeds of the chosen arm (models/m09h/confirm/)
  add --dry-run to print the commands without running them.

The development round is an ablation: the full recipe, and four versions each missing one
change, so the note can say which change matters. The recipe for the confirmation round is
then chosen on the validation cases only (uv run python -m rl.goal_test --split validation).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from rl.m09 import RECIPES as M09_RECIPES
from rl.m09 import V8_RECIPES, V9_RECIPES

TIMESTEPS = 400_000
THREADS = 2  # 15 runs x 2 threads = 30 of the 32 hardware threads
VAL_EVERY = 25_000
DEV_SEEDS = (0, 1, 2)
CONFIRM_SEEDS = (10, 11, 12, 13, 14)  # never used before
M09C_DEV_SEEDS = (30, 31, 32)  # M09c (DECISIONS): never used before
M09C_CONFIRM_SEEDS = (40, 41, 42, 43, 44)
M09H_DEV_SEEDS = (50, 51, 52, 53, 54)  # M09h (DECISIONS): never used before
M09H_CONFIRM_SEEDS = (60, 61, 62, 63, 64)
MAX_PARALLEL = 15  # 15 runs x 2 threads = 30 of the 32 hardware threads; the rest wait for a free slot
# What every M09c recipe keeps from v7's recipe (M04a floor round); the rest is rl/m09.py's setting and recipe.
M09C_COMMON = {"relabel": 0.8, "surge-max": 5.0, "load-scale": 1.2, "min-share": 0.05}

# Every M02b change on (B1 reward v2, B2 relabeling, B3 coverage, B4 filter in the loop).
FULL = {"relabel": 0.8, "filter-in-loop": True, "surge-max": 5.0, "load-scale": 1.2,
        "episode-steps": 50, "reward": "v2"}
# Each ablation is the full recipe with one change undone (back to its M02 value).
RECIPES = {
    "full": FULL,
    "norelabel": {**FULL, "relabel": 0.0},
    "noreward": {**FULL, "reward": "v1"},
    "nofilter": {**FULL, "filter-in-loop": False},
    "nocoverage": {**FULL, "surge-max": 3.0, "load-scale": 1.0, "episode-steps": 200},
}


def command(recipe: str, seed: int, group: str) -> list[str]:
    """The rl.train command line of one run; the model is named td3_<recipe>_seed<N>."""
    args = [sys.executable, "-m", "rl.train", "--algo", "td3", "--tag", f"_{recipe}", "--seed", str(seed),
            "--timesteps", str(TIMESTEPS), "--threads", str(THREADS), "--val-every", str(VAL_EVERY),
            "--group", group]
    if recipe in M09_RECIPES:
        return args + ["--m09", recipe] + [x for flag, value in M09C_COMMON.items() for x in (f"--{flag}", str(value))]
    for flag, value in RECIPES[recipe].items():
        if value is True:
            args.append(f"--{flag}")
        elif value is not False:
            args += [f"--{flag}", str(value)]
    return args


def launch(runs: list[tuple[str, int]], group: str, dry_run: bool, max_parallel: int = MAX_PARALLEL) -> None:
    """Start the runs, at most max_parallel at a time, and wait for all of them."""
    log_dir = Path("runs") / group / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for recipe, seed in runs:
        print(" ".join(command(recipe, seed, group)[1:]))
    if dry_run:
        return
    waiting, procs = list(runs), []

    def start_next() -> None:
        while waiting and sum(p.poll() is None for _, p in procs) < max_parallel:
            recipe, seed = waiting.pop(0)
            log = open(log_dir / f"td3_{recipe}_seed{seed}.log", "w")  # noqa: SIM115 (kept open while running)
            procs.append((f"td3_{recipe}_seed{seed}",
                          subprocess.Popen(command(recipe, seed, group), stdout=log, stderr=subprocess.STDOUT)))

    start_next()
    print(f"\n{len(runs)} runs, {max_parallel} at a time; logs in {log_dir}/; "
          f"TensorBoard: uv run tensorboard --logdir runs/{group}/tb")
    start = time.time()
    while waiting or any(p.poll() is None for _, p in procs):
        time.sleep(60)
        start_next()
        running = sum(p.poll() is None for _, p in procs)
        print(f"  {(time.time() - start) / 60:5.1f} min: {running} running, {len(procs) - running} finished, "
              f"{len(waiting)} waiting")
    failed = [name for name, p in procs if p.returncode != 0]
    print("all runs finished" + (f"; FAILED: {failed} (see their logs)" if failed else " successfully"))


def checkpoint_reward(model: Path) -> float | None:
    """Validation reward of the checkpoint a run kept (ValidationCheckpoint's rule), or None if it kept none."""
    history = model.with_name(f"{model.stem}_validation.json")
    if not model.with_name(f"{model.stem}_best.zip").exists() or not history.exists():
        return None
    eligible = [m for m in json.loads(history.read_text())
                if m["t2_pass"] and (m["t1a_pass"] or not m.get("t1a_required"))]
    return max(eligible, key=lambda m: (m["t1b_captured"], m["reward"]))["reward"] if eligible else None


def median_seeds(folder: Path = Path("models/m09h/dev")) -> dict[str, Path | None]:
    """Per v9 arm, the checkpoint of its median seed by validation reward (DECISIONS: an arm is judged by its median
    seed, not its best). A seed without a checkpoint ranks last; if the median one has none, the arm has no policy."""
    out: dict[str, Path | None] = {}
    for recipe in V9_RECIPES:
        runs = sorted(folder.glob(f"td3_{recipe}_seed*.zip"))
        runs = [r for r in runs if not r.stem.endswith("_best")]
        if not runs:
            continue
        ranked = sorted(runs, key=lambda r: -1e9 if checkpoint_reward(r) is None else checkpoint_reward(r))
        median = ranked[(len(ranked) - 1) // 2]  # 5 seeds: the 3rd
        out[recipe] = median.with_name(f"{median.stem}_best.zip") if checkpoint_reward(median) is not None else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("round", choices=["dev", "confirm", "m09c-dev", "m09c-confirm", "m09h-dev", "m09h-median",
                                          "m09h-confirm"])
    parser.add_argument("--recipe", choices=sorted(RECIPES) + sorted(M09_RECIPES), help="the recipe to confirm")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.round == "m09c-dev":
        launch([(r, s) for r in V8_RECIPES for s in M09C_DEV_SEEDS], "m09c/dev", args.dry_run)
    elif args.round == "m09c-confirm":
        if args.recipe not in V8_RECIPES:
            parser.error("m09c-confirm needs --recipe, one of the v8 recipes (chosen by the rule in DECISIONS)")
        launch([(args.recipe, s) for s in M09C_CONFIRM_SEEDS], "m09c/confirm", args.dry_run)
    elif args.round == "m09h-dev":
        launch([(r, s) for r in V9_RECIPES for s in M09H_DEV_SEEDS], "m09h/dev", args.dry_run)
    elif args.round == "m09h-median":
        for recipe, path in median_seeds().items():
            print(f"{recipe:14s} " + (f"uv run python -m eval.benchmark --dev --controllers proposed --policy {path} "
                                      f"--study m09h_{recipe}" if path else "median seed has no checkpoint: arm out"))
    elif args.round == "m09h-confirm":
        if args.recipe not in V9_RECIPES:
            parser.error("m09h-confirm needs --recipe, one of the v9 arms (chosen by the rule in DECISIONS)")
        launch([(args.recipe, s) for s in M09H_CONFIRM_SEEDS], "m09h/confirm", args.dry_run)
    elif args.round == "dev":
        launch([(r, s) for r in RECIPES for s in DEV_SEEDS], "m02b/dev", args.dry_run)
    else:
        if not args.recipe:
            parser.error("confirm needs --recipe (the one chosen by rl.goal_test --split validation)")
        launch([(args.recipe, s) for s in CONFIRM_SEEDS], "m02b/confirm", args.dry_run)


if __name__ == "__main__":
    main()
