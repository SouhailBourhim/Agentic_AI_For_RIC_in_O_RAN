"""How much does an LLM controller's outcome change when the same instance is played again? (M09b)

  uv run python -m eval.repeat_runs --play      3 more plays of the 20 tuning instances, 4 LLM controllers (~40 min)
  uv run python -m eval.repeat_runs --table     the spread per instance, from the dev benchmark's play + these 3

M09a found that qwen3:8b through Ollama doesn't always give the same decision for the same prompt, and that
tiny irrelevant input differences flip it. Rebuilding one prompt and sending it again (eval.llm_repeat) measures
that for one decision; this measures what it does to the result: the same instance, seed and code, played
4 times (the dev benchmark's play counts as the first). The only thing that differs between plays is the model's
answers and the real latency charged to the cell (eval.harness ChargedLLM).
Instances: eval.tune_llm_only.TUNE_SET (4 per family). Runs: eval/results/m09b/repeats (git-ignored).
Table: docs/learn/data/M09b-repeats.csv, one row per instance, controller and intent.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from agent.llm import OllamaClient
from eval.benchmark import FEASIBLE, STUDY, profile_cell, work_of
from eval.generator import generate
from eval.harness import RunSpec, run_once
from eval.metrics import measure_run
from eval.sim_profiles import KPI_DELAY_TICKS, noise_std
from eval.tune_llm_only import TUNE_SET
from eval.validate_sim import CLOCK_OFFSET_S, served_policy
from rl.runner import SERVED_SMOOTHING_ALPHA

RESULTS = Path("eval/results/m09b/repeats")
TABLE = Path("docs/learn/data/M09b-repeats.csv")
CONTROLLERS = ("llm_only", "llm_only_v2", "proposed", "formula")
EXTRA_PLAYS = 3


def repeat_work(inst, controller: str, play: int) -> Path:
    return RESULTS / RunSpec(f"{STUDY}_rep{play}", inst.name, controller, inst.seed).run_id


def play_all() -> None:
    llm, policy = OllamaClient(), served_policy()
    jobs = [(generate(f, i), c, k) for f, i in TUNE_SET for c in CONTROLLERS for k in range(1, EXTRA_PLAYS + 1)]
    for n, (inst, controller, k) in enumerate(jobs, 1):
        work = repeat_work(inst, controller, k)
        if not (work / "run.json").exists():  # resumes if stopped
            spec = RunSpec(f"{STUDY}_rep{k}", inst.name, controller, inst.seed, alpha=SERVED_SMOOTHING_ALPHA,
                           delay_ticks=KPI_DELAY_TICKS)
            run_once(spec, inst.scenario(), work, policy, llm, cfg=profile_cell(inst), traffic=inst.traffic(noise_std()),
                     clock_offset_s=CLOCK_OFFSET_S)
        print(f"[{n}/{len(jobs)}] {inst.name} {controller} play {k + 1}", flush=True)


def table() -> list[dict]:
    rows = []
    for f, i in TUNE_SET:
        inst = generate(f, i)
        for controller in CONTROLLERS:
            works = [work_of(inst, controller)] + [repeat_work(inst, controller, k) for k in range(1, EXTRA_PLAYS + 1)]
            works = [w for w in works if (w / "run.json").exists()]
            if len(works) < 2:
                continue
            plays = np.array([[r["intent_met"] for r in measure_run(w, inst.scenario())["intents"]] for w in works])
            for k in range(plays.shape[1]):
                met = plays[:, k]
                rows.append({"family": f, "instance": inst.name, "controller": controller, "intent": k + 1, "plays": len(met),
                             "mean": round(float(met.mean()), 3), "min": round(float(met.min()), 3), "max": round(float(met.max()), 3),
                             "range": round(float(met.max() - met.min()), 3),
                             "flips": bool((met >= FEASIBLE).any() and (met < FEASIBLE).any())})
    if not rows:
        raise SystemExit("no repeated plays yet: run --play first")
    with TABLE.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--play", action="store_true")
    group.add_argument("--table", action="store_true")
    args = parser.parse_args()
    if args.play:
        play_all()
        return
    rows = table()
    for controller in CONTROLLERS:
        mine = [r for r in rows if r["controller"] == controller]
        print(f"{controller:12s} intents {len(mine):3d}  median range {np.median([r['range'] for r in mine]):.2f}"
              f"  range > 0.2: {sum(r['range'] > 0.2 for r in mine):2d}  crosses the 0.9 line: {sum(r['flips'] for r in mine):2d}")


if __name__ == "__main__":
    main()
