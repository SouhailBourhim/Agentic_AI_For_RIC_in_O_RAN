"""Revisions of LLM only's prompt for the fairer baseline (M09b), on a fixed subset of dev instances only.

  uv run python -m eval.tune_llm_only <revision> <prompt file> [--repeats 3]

Plays LLM only with the prompt in <prompt file> on TUNE_SET (20 dev instances, 4 per family, both variants of F1
and F4), and appends the revision, its full text and its scores to docs/learn/data/M09b-llm-only-revisions.csv.
At most 5 revisions (DECISIONS); the best mean intent met becomes LLM_ONLY_PLAN["v2"] in agent/prompts.py, frozen
before the benchmark. The same budget-and-log rule a fair comparison needs: what was tried is on record.
"""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from agent import prompts
from agent.llm import OllamaClient
from eval.benchmark import STUDY, profile_cell
from eval.generator import generate
from eval.harness import RunSpec, run_once
from eval.metrics import measure_run
from eval.sim_profiles import KPI_DELAY_TICKS, noise_std
from eval.validate_sim import CLOCK_OFFSET_S, served_policy
from rl.runner import SERVED_SMOOTHING_ALPHA

RESULTS = Path("eval/results/m09b/tune")
LOG = Path("docs/learn/data/M09b-llm-only-revisions.csv")
TUNE_SET = [("f1", 0), ("f1", 1), ("f1", 20), ("f1", 21), ("f2", 0), ("f2", 1), ("f2", 2), ("f2", 3),
            ("f3", 0), ("f3", 1), ("f3", 2), ("f3", 3), ("f4", 0), ("f4", 1), ("f4", 30), ("f4", 31),
            ("f5", 0), ("f5", 1), ("f5", 2), ("f5", 3)]
MAX_REVISIONS = 5


def score(revision: int, prompt: str, repeats: int = 1) -> dict[str, float]:
    prompts.LLM_ONLY_PLAN["v2"] = prompt  # this process only: the file in agent/prompts.py changes only for the winner
    llm, policy = OllamaClient(), served_policy()
    per_family: dict[str, list[float]] = {}
    for (family, index), repeat in ((pair, k) for pair in TUNE_SET for k in range(repeats)):
        inst = generate(family, index)
        study = f"{STUDY}_tune_r{revision}" + (f"_rep{repeat}" if repeats > 1 else "")  # same instance, the model's own spread
        spec = RunSpec(study, inst.name, "llm_only_v2", inst.seed, alpha=SERVED_SMOOTHING_ALPHA, delay_ticks=KPI_DELAY_TICKS)
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
    parser.add_argument("prompt", type=Path)
    parser.add_argument("--repeats", type=int, default=1, help="plays of each instance (3 from revision 3 on, DECISIONS)")
    args = parser.parse_args()
    if not 0 <= args.revision <= MAX_REVISIONS:
        raise SystemExit(f"revision 0 (v1, the reference) to {MAX_REVISIONS}")
    text = args.prompt.read_text().strip()
    result = score(args.revision, text, args.repeats)
    row = {"revision": args.revision, "repeats": args.repeats, "sha": hashlib.sha256(text.encode()).hexdigest()[:10],
           **{k: round(v, 3) for k, v in result.items()}, "prompt": text}
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
    print({k: v for k, v in row.items() if k != "prompt"})


if __name__ == "__main__":
    main()
