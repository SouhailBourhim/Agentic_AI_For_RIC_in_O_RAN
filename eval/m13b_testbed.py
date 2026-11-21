"""M13b on the testbed: S3b paired with and without the offer, and one S3 smoke run (rules in DECISIONS 2026-11-21).

  uv run python -m eval.m13b_testbed --play     S3b, 3 repeats x {nooffer, offer}, order rotated; resumes if stopped
  uv run python -m eval.m13b_testbed --smoke    S3, one run with the offer and the load-aware verifier
  uv run python -m eval.m13b_testbed --score    which repeats count, the S3b verdict and the smoke check
  uv run python -m eval.m13b_testbed --replay   once, after --score: replays the S3b repeats that didn't count

Every run has the load-aware verifier (RIC_BOUND_FROM_LOAD=1). S3b is S3's two intents with a 15 Mb/s URLLC surge
(eval.testbed_scenarios.EXTRA_SCENARIOS): the 23 Mb/s request still doesn't fit, and the offer (16 Mb/s under 5 ms)
is above the traffic, so it can be held. In S3 (31.5 Mb/s of traffic) the load-aware verifier should make no offer.
A repeat counts as in M13 (both runs valid, >= 0.68 Mb/s per PRB). S3b verdict over the counting repeats: median of
offer - nooffer on intent 2's delivered (> 0 helps) and on intent 1's met (>= -0.05 no harm). The smoke run passes
when intent 2's first answer is a conflict with no offer and intent 1 is met >= 0.907 (M13's lowest nooffer, 0.957,
- 0.05). Raw runs: eval/results/m13b_testbed/.
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

RESULTS = Path("eval/results/m13b_testbed")
ARMS = (("nooffer", "0"), ("offer", "1"))
REPEATS = (1, 2, 3)
HELPS, NO_HARM = 0.0, -0.05
SMOKE_MET = 0.957 - 0.05  # M13's lowest nooffer intent 1 met on S3 (docs/learn/data/M13-testbed.json), - 0.05


def run_of(arm: str, repeat: int, scenario: str = "s3b") -> TestbedRun:
    return TestbedRun(arm, scenario, "proposed", repeat)


def play_one(run: TestbedRun, offers: str) -> None:
    if (RESULTS / run.run_id / "run.json").exists():
        return
    work = play(run, 1.0, results=RESULTS, agent_env={"RIC_OFFERS": offers, "RIC_BOUND_FROM_LOAD": "1"})
    spec = json.loads((work / "run.json").read_text())
    print(run.run_id, "valid" if spec["valid"] else spec["problems"], flush=True)


def play_all() -> None:
    check_image()
    require_testbed_up()
    for repeat in REPEATS:
        order = ARMS if repeat % 2 else ARMS[::-1]  # rotated, so slow drift favours no arm
        for arm, offers in order:
            play_one(run_of(arm, repeat), offers)


def smoke() -> None:
    check_image()
    require_testbed_up()
    play_one(run_of("smoke", 1, "s3"), "1")


def measure(work: Path, scenario: str) -> dict:
    spec = json.loads((work / "run.json").read_text())
    events = read_events(work / "events.jsonl")
    kpis = load_run(work)[0]
    first, second = scenario_as_played(scenario, events).intents
    verdict, offer, accepted = decisions(events, intent_id_of(events, second))
    return {"valid": spec["valid"], "per_prb": per_prb(kpis), "delivered_2": delivered(kpis, second),
            "met_2": met(kpis, second), "met_1": met(kpis, first), "first_check": (verdict or {}).get("check"),
            "first_reason": (verdict or {}).get("reason"), "offer": offer, "applied_2": accepted,
            "fallbacks": sum(e["event"] == "executor_fallback" for e in events)}


def repeats() -> dict:
    out = {}
    for repeat in REPEATS:
        runs = {arm: measure(RESULTS / run_of(arm, repeat).run_id, "s3b") for arm, _ in ARMS}
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
    smoke_work = RESULTS / run_of("smoke", 1, "s3").run_id
    if (smoke_work / "run.json").exists():
        s = measure(smoke_work, "s3")
        s["passes"] = bool(s["valid"] and s["first_check"] == "conflict" and s["offer"] is None
                           and s["met_1"] >= SMOKE_MET)
        out["smoke_s3"] = s
    return out


def replay() -> None:
    """Keep each S3b repeat that didn't count as repeatN_lowcap, then play it again (once)."""
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
    for flag in ("--play", "--smoke", "--score", "--replay"):
        group.add_argument(flag, action="store_true")
    args = parser.parse_args()
    if args.play:
        play_all()
    elif args.smoke:
        smoke()
    elif args.replay:
        replay()
    else:
        print(json.dumps(score(), indent=1, default=float))


if __name__ == "__main__":
    main()
