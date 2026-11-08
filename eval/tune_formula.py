"""Revisions of the LLM + formula baseline (M09b), on the same fixed subset of dev instances as LLM only's prompt.

  uv run python -m eval.tune_formula <revision> "<what changed>" [--peak-window 30] [--repeats 3]

Plays LLM + formula, with rl.baselines.FormulaPolicy as it is in the working tree, on eval.tune_llm_only.TUNE_SET
(20 dev instances), and appends the revision and its scores to docs/learn/data/M09b-formula-revisions.csv.
--peak-window sets rl.baselines.PEAK_WINDOW_S in this process only (0 = size URLLC to its current load), so two
revisions that differ only by it are scored from the same code. The same budget as LLM only's prompt (at most 5
revisions, 3 plays each, DECISIONS 2026-11-08): the best mean intent met is frozen before M09d.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from agent.llm import OllamaClient
from eval.benchmark import STUDY, profile_cell
from eval.generator import generate
from eval.harness import RunSpec, run_once
from eval.metrics import measure_run
from eval.sim_profiles import KPI_DELAY_TICKS, noise_std
from eval.tune_llm_only import TUNE_SET
from eval.validate_sim import CLOCK_OFFSET_S, served_policy
from rl import baselines
from rl.runner import SERVED_SMOOTHING_ALPHA

RESULTS = Path("eval/results/m09b/tune_formula")
LOG = Path("docs/learn/data/M09b-formula-revisions.csv")
MAX_REVISIONS = 5


def score(revision: int, repeats: int) -> dict[str, float]:
    llm, policy = OllamaClient(), served_policy()
    per_family: dict[str, list[float]] = {}
    for (family, index), repeat in ((pair, k) for pair in TUNE_SET for k in range(repeats)):
        inst = generate(family, index)
        study = f"{STUDY}_formula_r{revision}_rep{repeat}"  # same instance, the model's own spread
        spec = RunSpec(study, inst.name, "formula", inst.seed, alpha=SERVED_SMOOTHING_ALPHA, delay_ticks=KPI_DELAY_TICKS)
        work = run_once(spec, inst.scenario(), RESULTS / spec.run_id, policy, llm, cfg=profile_cell(inst),
                        traffic=inst.traffic(noise_std()), clock_offset_s=CLOCK_OFFSET_S)
        met = [r["intent_met"] for r in measure_run(work, inst.scenario())["intents"]]
        per_family.setdefault(family, []).extend(met)
        print(f"  {inst.name}: {[round(m, 2) for m in met]}", flush=True)
    return {"mean_met": float(np.mean([m for v in per_family.values() for m in v])),
            **{f"{f}_mean": float(np.mean(v)) for f, v in per_family.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("revision", type=int)
    parser.add_argument("change", help="what this revision changed, for the log")
    parser.add_argument("--peak-window", type=int, default=baselines.PEAK_WINDOW_S)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.revision <= MAX_REVISIONS:
        raise SystemExit(f"revision 1 to {MAX_REVISIONS} (revision 0 is the frozen formula, scored by the dev benchmark)")
    baselines.PEAK_WINDOW_S = args.peak_window
    result = score(args.revision, args.repeats)
    row = {"revision": args.revision, "repeats": args.repeats, "peak_window_s": args.peak_window,
           **{k: round(v, 3) for k, v in result.items()}, "change": args.change}
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
    print({k: v for k, v in row.items() if k != "change"})


if __name__ == "__main__":
    main()
