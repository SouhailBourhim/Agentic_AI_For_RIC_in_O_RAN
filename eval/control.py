"""The control comparison on the simulator: runs the matrix and writes one table (M08a).

  uv run python -m eval.control --dev                                  seed 4999, for bugs only
  uv run python -m eval.control --controllers static rl_only           seconds
  uv run python -m eval.control --controllers llm_only proposed        needs Ollama; long
  uv run python -m eval.control --study bridge                         M08c's reference
  uv run python -m eval.control --study lever                          importance 6 or 8, no LLM
  uv run python -m eval.control --study fixed                          after the fixes, fresh seeds; needs Ollama

A finished run is never played twice, so a stopped command can simply be started again. Raw runs go
to eval/results/m08a/ (git-ignored); the table of all finished runs to eval/results/m08a/runs.csv and
docs/learn/data/M08a-runs.csv. No result is written by hand anywhere.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

from agent.llm import OllamaClient
from agent.prompts import PARSE_VERSION
from eval.harness import CONTROLLERS, USES_AGENT, RunSpec, run_once
from eval.metrics import measure_run
from eval.scenarios import DEV_SEED, SCENARIOS, SEEDS
from rl.runner import SERVED_SMOOTHING_ALPHA, load_policy

RESULTS = Path("eval/results/m08a")
TABLE = Path("docs/learn/data/M08a-runs.csv")
TESTBED_DELAY_TICKS = 2  # the late view that matches the testbed (M07b)
BRIDGE_PROPOSED_SEEDS = range(5000, 5005)
LEVER_IMPORTANCES = (6.0, 8.0)
FIXED_SEEDS = range(7000, 7010)  # fresh: the fixes were chosen on seeds 6000-6009, not reported as results


def headline(controllers: list[str], seeds) -> list[RunSpec]:
    return [RunSpec("headline", s, c, seed) for s in SCENARIOS for c in controllers for seed in seeds]


def bridge() -> list[RunSpec]:
    """Simulator settings against testbed settings: RL only on all four combinations (the first is in
    the headline study), LLM + RL on the testbed-like one."""
    settings = [(0.25, TESTBED_DELAY_TICKS), (SERVED_SMOOTHING_ALPHA, 0), (SERVED_SMOOTHING_ALPHA, TESTBED_DELAY_TICKS)]
    specs = [RunSpec("bridge", s, "rl_only", seed, alpha, delay) for s in SCENARIOS for alpha, delay in settings
             for seed in SEEDS]
    return specs + [RunSpec("bridge", s, "proposed", seed, SERVED_SMOOTHING_ALPHA, TESTBED_DELAY_TICKS)
                    for s in SCENARIOS for seed in BRIDGE_PROPOSED_SEEDS]


def fixed() -> list[RunSpec]:
    """After the fixes of 2026-11-07, on fresh seeds: the two controllers they change, as served and at
    the testbed's delay. Static and LLM only do not use the policy, its objective or its filter."""
    specs = [RunSpec("fixed", s, c, seed, SERVED_SMOOTHING_ALPHA) for s in SCENARIOS for c in ("rl_only", "proposed")
             for seed in FIXED_SEEDS]
    return specs + [RunSpec("fixed_late", s, "proposed", seed, SERVED_SMOOTHING_ALPHA, TESTBED_DELAY_TICKS)
                    for s in SCENARIOS for seed in FIXED_SEEDS]


def lever() -> list[RunSpec]:
    return [RunSpec("lever", "s1", "scripted", seed, importance=i) for i in LEVER_IMPORTANCES for seed in SEEDS]


def write_table() -> int:
    """One row per finished run and intent, from every run.json under RESULTS (dev runs excluded)."""
    rows = []
    for path in sorted(RESULTS.glob("*/*/*/seed*/run.json")):
        spec = json.loads(path.read_text())
        if spec["study"] in ("dev", "smoke"):
            continue
        measured = measure_run(path.parent, SCENARIOS[spec["scenario"]])
        base = {k: spec[k] for k in ("study", "scenario", "controller", "seed", "alpha", "delay_ticks", "importance")}
        rows += [{**base, **intent, **measured["run"], "wall_s": spec["wall_s"]} for intent in measured["intents"]]
    if not rows:
        return 0
    for out in (RESULTS / "runs.csv", TABLE):
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", choices=["headline", "bridge", "lever", "fixed"], default="headline")
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS))
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--dev", action="store_true", help=f"seed {DEV_SEED} only; not in the table")
    args = parser.parse_args()

    if args.dev:
        specs = [RunSpec("dev", s, c, DEV_SEED) for s in SCENARIOS for c in args.controllers]
    else:
        specs = {"headline": lambda: headline(args.controllers, SEEDS), "bridge": bridge, "lever": lever, "fixed": fixed}[args.study]()
    specs = [s for s in specs if s.scenario in args.scenarios]
    to_play = [s for s in specs if not (RESULTS / s.run_id / "run.json").exists()]
    print(f"{len(specs)} runs, {len(specs) - len(to_play)} already finished")

    _, label = load_policy(None)  # refreshes the cache from the registry's champion
    policy_file = Path("var/policy_cache") / json.loads(Path("var/policy_cache/served.json").read_text())["file"]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    llm = None
    if any(s.controller in USES_AGENT for s in to_play):
        llm = OllamaClient()
        llm.chat([{"role": "user", "content": "ok"}])  # loads the model: a cold load is charged to no run
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "manifest.json").write_text(json.dumps(
        {"commit": commit, "policy": label, "llm": llm.model if llm else None, "parse_prompt": PARSE_VERSION}, indent=1))

    for i, spec in enumerate(to_play, 1):
        work = run_once(spec, SCENARIOS[spec.scenario], RESULTS / spec.run_id, policy_file,
                        llm if spec.controller in USES_AGENT else None)
        wall = json.loads((work / "run.json").read_text())["wall_s"]
        print(f"[{i}/{len(to_play)}] {spec.run_id}: {wall:.0f} s", flush=True)
    print(f"{write_table()} rows in {TABLE}; policy {label}, commit {commit}")


if __name__ == "__main__":
    main()
