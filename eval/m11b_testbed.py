"""M11b on the testbed: S3 paired on the same cell, the same day (rules in DECISIONS 2026-11-21).

  uv run python -m eval.m11b_testbed --play     3 repeats x {static, mon, nomon}, order rotated; resumes if stopped
  uv run python -m eval.m11b_testbed --score    which repeats count, then the verdict

mon is LLM + RL with the served monitor, nomon the same with --no-monitor, static the equal split: the cell's reference
for that repeat. A repeat counts when its three runs are valid and each carried >= 0.68 Mb/s per PRB over the run;
verdict: the median over the repeats that count of mon - nomon on S3 intent 1's met (> 0 helps, < -0.05 hurts).
The first pass is kept as repeatN_lowcap (reported, not scored). Raw runs: eval/results/m11b_testbed/ (git-ignored).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from eval.metrics import load_run, measure_run
from eval.testbed_control import TestbedRun, check_image, play, require_testbed_up, scenario_as_played
from store.events import read_events

RESULTS = Path("eval/results/m11b_testbed")
ARMS = (("static", "static", ""), ("mon", "proposed", ""), ("nomon", "proposed", "--no-monitor"))
REPEATS = (1, 2, 3)
MIN_PER_PRB = 0.68  # Mb/s per PRB over the run; M08d's cell carried 0.72-0.75
N_PRB = 106
HELPS, HURTS = 0.0, -0.05


def run_of(study: str, controller: str, repeat: int) -> TestbedRun:
    return TestbedRun(study, "s3", controller, repeat)


def play_all() -> None:
    check_image()
    require_testbed_up()
    for repeat in REPEATS:
        k = repeat - 1
        for study, controller, extra in ARMS[k:] + ARMS[:k]:  # rotated, so slow drift favours no arm
            run = run_of(study, controller, repeat)
            if (RESULTS / run.run_id / "run.json").exists():
                continue
            work = play(run, 1.0, extra_args=extra, results=RESULTS)
            spec = json.loads((work / "run.json").read_text())
            print(run.run_id, "valid" if spec["valid"] else spec["problems"], flush=True)


def per_prb(kpis: dict) -> float:
    """Median Mb/s per PRB of the slices while they were saturated (offered > 1.1 x served)."""
    values = [kpis[s]["thr"][m] / (kpis[s]["share"][m] * N_PRB) for s in kpis
              for m in [(kpis[s]["load"] > kpis[s]["thr"] * 1.1) & (kpis[s]["share"] > 0.05)]]
    values = np.concatenate(values)
    return float(np.nanmedian(values)) if values.size else float("nan")


def measure(work: Path) -> dict:
    spec = json.loads((work / "run.json").read_text())
    events = read_events(work / "events.jsonl")
    intents = measure_run(work, scenario_as_played("s3", events))["intents"]
    return {"valid": spec["valid"], "per_prb": per_prb(load_run(work)[0]), "met": intents[0]["intent_met"],
            "fallbacks": [(round(e["ts_backend"]), e["data"]["slice"]) for e in events if e["event"] == "executor_fallback"]}


def score() -> dict:
    out, diffs = {}, []
    for repeat in REPEATS:
        runs = {study: measure(RESULTS / run_of(study, controller, repeat).run_id) for study, controller, _ in ARMS}
        counts = all(r["valid"] and r["per_prb"] >= MIN_PER_PRB for r in runs.values())
        out[f"repeat{repeat}"] = {"counts": counts, **runs}
        if counts:
            diffs.append(runs["mon"]["met"] - runs["nomon"]["met"])
    median = float(np.median(diffs)) if diffs else None
    verdict = ("inconclusive: no repeat counts" if median is None else
               "helps" if median > HELPS else "hurts" if median < HURTS else "no measurable difference")
    out["verdict"] = {"mon_minus_nomon": diffs, "median": median, "reading": verdict}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--play", action="store_true")
    group.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.play:
        play_all()
    else:
        print(json.dumps(score(), indent=1, default=float))


if __name__ == "__main__":
    main()
