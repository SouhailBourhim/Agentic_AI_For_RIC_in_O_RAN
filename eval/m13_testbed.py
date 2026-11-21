"""M13 on the testbed: S3 paired, with and without the verifier's offer (rules in DECISIONS 2026-11-21).

  uv run python -m eval.m13_testbed --play     3 repeats x {nooffer, offer}, order rotated; resumes if stopped
  uv run python -m eval.m13_testbed --score    which repeats count, then the verdict
  uv run python -m eval.m13_testbed --replay   once, after --score: replays the repeats that didn't count (the first
                                               try is kept as repeatN_lowcap, reported, not scored)

S3's second intent ("23 Mb/s to URLLC under 5 ms") doesn't fit beside the first (eMBB and BE 23 Mb/s each): the
verifier rejects it with an offer. Both arms are LLM + RL with the served monitor; `offer` shows the offer to the
model (RIC_OFFERS=1), `nooffer` is the agent as before M13. A repeat counts when both runs are valid and each carried
>= 0.68 Mb/s per PRB. Verdict, over the repeats that count: the median of offer - nooffer on intent 2's delivered
(eval.m13.delivered; > 0 helps), and of intent 1's met (>= -0.05: no harm). Raw runs: eval/results/m13_testbed/.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from eval.m11b_testbed import MIN_PER_PRB, per_prb
from eval.m13 import decisions, delivered, met
from eval.metrics import intent_id_of, load_run
from eval.testbed_control import TestbedRun, check_image, play, require_testbed_up, scenario_as_played
from store.events import read_events

RESULTS = Path("eval/results/m13_testbed")
ARMS = (("nooffer", "0"), ("offer", "1"))
REPEATS = (1, 2, 3)
HELPS, NO_HARM = 0.0, -0.05


def run_of(arm: str, repeat: int) -> TestbedRun:
    return TestbedRun(arm, "s3", "proposed", repeat)


def play_all() -> None:
    check_image()
    require_testbed_up()
    for repeat in REPEATS:
        order = ARMS if repeat % 2 else ARMS[::-1]  # rotated, so slow drift favours no arm
        for arm, offers in order:
            run = run_of(arm, repeat)
            if (RESULTS / run.run_id / "run.json").exists():
                continue
            work = play(run, 1.0, results=RESULTS, agent_env={"RIC_OFFERS": offers})
            spec = json.loads((work / "run.json").read_text())
            print(run.run_id, "valid" if spec["valid"] else spec["problems"], flush=True)


def measure(work: Path) -> dict:
    spec = json.loads((work / "run.json").read_text())
    events = read_events(work / "events.jsonl")
    kpis = load_run(work)[0]
    first, second = scenario_as_played("s3", events).intents
    verdict, offer, accepted = decisions(events, intent_id_of(events, second))
    return {"valid": spec["valid"], "per_prb": per_prb(kpis), "delivered_2": delivered(kpis, second),
            "met_2": met(kpis, second), "met_1": met(kpis, first),
            "first_check": (verdict or {}).get("check"), "offer": offer, "applied_2": accepted}


def repeats() -> dict:
    out = {}
    for repeat in REPEATS:
        runs = {arm: measure(RESULTS / run_of(arm, repeat).run_id) for arm, _ in ARMS}
        out[f"repeat{repeat}"] = {"counts": all(r["valid"] and r["per_prb"] >= MIN_PER_PRB for r in runs.values()),
                                  **runs}
    return out


def score() -> dict:
    out = repeats()
    counting = [out[f"repeat{r}"] for r in REPEATS if out[f"repeat{r}"]["counts"]]
    gain = [r["offer"]["delivered_2"] - r["nooffer"]["delivered_2"] for r in counting]
    harm = [r["offer"]["met_1"] - r["nooffer"]["met_1"] for r in counting]
    if not counting:
        reading = "inconclusive: no repeat counts"
    else:
        helps, harmless = np.median(gain) > HELPS, np.median(harm) >= NO_HARM
        reading = f"{'helps' if helps else 'does not help'}; {'no harm' if harmless else 'harms intent 1'}"
    out["verdict"] = {"offer_minus_nooffer_delivered_2": gain, "offer_minus_nooffer_met_1": harm,
                      "median_gain": float(np.median(gain)) if gain else None,
                      "median_harm": float(np.median(harm)) if harm else None, "reading": reading}
    return out


def replay() -> None:
    """Keep each repeat that didn't count as repeatN_lowcap, then play it again (once: a second failure is left out)."""
    for repeat, result in ((r, repeats()[f"repeat{r}"]) for r in REPEATS):
        if result["counts"]:
            continue
        for arm, _ in ARMS:
            work = RESULTS / run_of(arm, repeat).run_id
            kept = work.with_name(f"{work.name}_lowcap")
            if kept.exists():
                raise SystemExit(f"{kept} exists: repeat {repeat} was replayed already, it is left out")
            work.rename(kept)
    play_all()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--play", action="store_true")
    group.add_argument("--score", action="store_true")
    group.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.play:
        play_all()
    elif args.replay:
        replay()
    else:
        print(json.dumps(score(), indent=1, default=float))


if __name__ == "__main__":
    main()
