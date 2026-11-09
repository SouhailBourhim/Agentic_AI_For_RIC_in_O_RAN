"""Launch the M02b training rounds: several rl.train runs in parallel, one log file each.

Run with:
  uv run python -m rl.sweep dev                       15 runs: 5 recipes x 3 seeds (models/m02b/dev/)
  uv run python -m rl.sweep confirm --recipe full     5 fresh seeds of one recipe (models/m02b/confirm/)
  uv run python -m rl.sweep m09c-dev                  M09c: 4 recipes x 3 seeds (models/m09c/dev/)
  uv run python -m rl.sweep m09c-confirm --recipe v8  M09c: 5 fresh seeds of the chosen recipe (models/m09c/confirm/)
  add --dry-run to print the commands without running them.

The development round is an ablation: the full recipe, and four versions each missing one
change, so the note can say which change matters. The recipe for the confirmation round is
then chosen on the validation cases only (uv run python -m rl.goal_test --split validation).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from rl.m09 import RECIPES as M09C_RECIPES

TIMESTEPS = 400_000
THREADS = 2  # 15 runs x 2 threads = 30 of the 32 hardware threads
VAL_EVERY = 25_000
DEV_SEEDS = (0, 1, 2)
CONFIRM_SEEDS = (10, 11, 12, 13, 14)  # never used before
M09C_DEV_SEEDS = (30, 31, 32)  # M09c (DECISIONS): never used before
M09C_CONFIRM_SEEDS = (40, 41, 42, 43, 44)
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
    if recipe in M09C_RECIPES:
        return args + ["--m09", recipe] + [x for flag, value in M09C_COMMON.items() for x in (f"--{flag}", str(value))]
    for flag, value in RECIPES[recipe].items():
        if value is True:
            args.append(f"--{flag}")
        elif value is not False:
            args += [f"--{flag}", str(value)]
    return args


def launch(runs: list[tuple[str, int]], group: str, dry_run: bool) -> None:
    log_dir = Path("runs") / group / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for recipe, seed in runs:
        cmd = command(recipe, seed, group)
        print(" ".join(cmd[1:]))
        if not dry_run:
            log = open(log_dir / f"td3_{recipe}_seed{seed}.log", "w")  # noqa: SIM115 (kept open while running)
            procs.append((f"td3_{recipe}_seed{seed}", subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
    if dry_run:
        return
    print(f"\n{len(procs)} runs started; logs in {log_dir}/; TensorBoard: uv run tensorboard --logdir runs/{group}/tb")
    start = time.time()
    while any(p.poll() is None for _, p in procs):
        time.sleep(60)
        running = sum(p.poll() is None for _, p in procs)
        print(f"  {(time.time() - start) / 60:5.1f} min: {running} running, {len(procs) - running} finished")
    failed = [name for name, p in procs if p.returncode != 0]
    print("all runs finished" + (f"; FAILED: {failed} (see their logs)" if failed else " successfully"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("round", choices=["dev", "confirm", "m09c-dev", "m09c-confirm"])
    parser.add_argument("--recipe", choices=sorted(RECIPES) + sorted(M09C_RECIPES), help="the recipe to confirm")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.round == "m09c-dev":
        launch([(r, s) for r in M09C_RECIPES for s in M09C_DEV_SEEDS], "m09c/dev", args.dry_run)
    elif args.round == "m09c-confirm":
        if args.recipe not in M09C_RECIPES:
            parser.error("m09c-confirm needs --recipe, one of rl.m09.RECIPES (chosen by the rule in DECISIONS)")
        launch([(args.recipe, s) for s in M09C_CONFIRM_SEEDS], "m09c/confirm", args.dry_run)
    elif args.round == "dev":
        launch([(r, s) for r in RECIPES for s in DEV_SEEDS], "m02b/dev", args.dry_run)
    else:
        if not args.recipe:
            parser.error("confirm needs --recipe (the one chosen by rl.goal_test --split validation)")
        launch([(args.recipe, s) for s in CONFIRM_SEEDS], "m02b/confirm", args.dry_run)


if __name__ == "__main__":
    main()
