"""M13b: the verifier sizes a bounded URLLC target from its traffic, on the simulator (m09 cell, served v7 + monitor).

  uv run python -m eval.m13b --regress         M13's dev set (F4b + F6) load-aware: every decision as M13's offer arm
  uv run python -m eval.m13b --dev             the 40 F7 dev instances x {reject, blind_offer, aware_offer}, scripted
  uv run python -m eval.m13b --score           rules H and G on dev (DECISIONS 2026-11-21) -> docs/learn/data/M13b-dev*
  uv run python -m eval.m13b --play            the m13b_test split, once (guard: committed code and DECISIONS)
  uv run python -m eval.m13b --score --final   H and G on the test split, once; marks m13b/test spent

The set (eval.generator.m13b_instances): F7, S3's shape. eMBB and best effort guaranteed, then a URLLC request under
a bound that the load-blind verifier rejects as a conflict, with URLLC traffic at 0.5-1.5 x its target. Arms, all
scripted (the operator takes any offer at once, eval.harness.apply_scripted), all with the served executor:
  reject       no offer, load-blind: as before M13
  blind_offer  M13 as shipped: the offer, sized from targets alone
  aware_offer  M13b: the offer from the load-aware verifier (RIC_BOUND_FROM_LOAD=1), when there is one
Raw runs: eval/results/m13b/ (git-ignored).
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval import benchmark, m13
from eval.generator import Instance, m13_instances, m13b_instances
from eval.harness import RunSpec, run_once
from eval.m09d import guard
from eval.metrics import intent_id_of, load_run
from eval.sim_profiles import KPI_DELAY_TICKS, noise_std
from eval.spent import mark_spent
from eval.validate_sim import CLOCK_OFFSET_S, served_policy
from rl.monitor import SERVED_TRIGGER
from rl.runner import SERVED_SMOOTHING_ALPHA

ROOT = Path("eval/results/m13b")
DATA = Path("docs/learn/data")
SPENT_NAME = "m13b/test"
ARMS = ("reject", "blind_offer", "aware_offer")
SWITCHES = {"reject": (False, False), "blind_offer": (True, False), "aware_offer": (True, True)}  # offers, from load
WORKERS = 8
# The frozen rules (DECISIONS 2026-11-21, M13b), on F7.
NO_HARM = 0.02  # H: mean in-force met, aware_offer >= reject - 0.02
GAIN = 0.05  # G: over the instances where the load-aware verifier offers, mean delivered aware_offer - reject >= +0.05


def spec_of(inst: Instance, label: str, study: str) -> RunSpec:
    offers, from_load = SWITCHES[label]
    return RunSpec(study, inst.name, "scripted", inst.seed, alpha=SERVED_SMOOTHING_ALPHA, delay_ticks=KPI_DELAY_TICKS,
                   executor="v7", monitor=SERVED_TRIGGER.name, offers=offers, bound_from_load=from_load)


def work_of(inst: Instance, label: str, study: str) -> Path:
    return ROOT / spec_of(inst, label, study).run_id


def instances(study: str) -> list[Instance]:
    return m13_instances("dev") if study == "regress" else m13b_instances("m13b_test" if study == "test" else "dev")


def jobs(study: str) -> list[tuple[Instance, str, str]]:
    labels = ("aware_offer",) if study == "regress" else ARMS
    return [(inst, label, study) for inst in instances(study) for label in labels]


def play(job: tuple[Instance, str, str]) -> Path:
    """One arm on one instance. A finished run is not played again."""
    inst, label, study = job
    work = work_of(inst, label, study)
    if (work / "run.json").exists():
        return work
    return run_once(spec_of(inst, label, study), inst.scenario(), work, served_policy(), None,
                    cfg=benchmark.profile_cell(inst, "m09"), traffic=inst.traffic(noise_std()),
                    clock_offset_s=CLOCK_OFFSET_S)


def play_all(todo: list) -> None:
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for n, work in enumerate(pool.map(play, todo, chunksize=2), 1):
            if n % 20 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] {work}", flush=True)


# ---------------------------------------------------------------- the numbers


def traffic_of(inst: Instance) -> float:
    """URLLC's traffic during the request (F7's second step)."""
    return float(inst.steps["urllc"][-1][1])


def row_of(inst: Instance, label: str, study: str) -> dict:
    """F7: the request is the second intent (rejected by the load-blind verifier), the guarantee in force the first."""
    kpis, events = load_run(work_of(inst, label, study))
    guarantee, request = inst.scenario().intents
    intent_id = intent_id_of(events, request)
    first, offer, accepted = m13.decisions(events, intent_id)
    applied = any(e["event"] == "applied" and e.get("intent_id") == intent_id for e in events)
    offered = None if offer is None else {s: v for s, v in offer["min_thr_mbps"].items() if v}
    return {"instance": inst.name, "label": label, "traffic": traffic_of(inst), "target": request.targets["urllc"],
            "bound": request.max_latency_ms, "first_check": (first or {}).get("check", ""),
            "first_reason": (first or {}).get("reason", ""), "offer": json.dumps(offered),
            "applied": applied and accepted is not None,
            "delivered": m13.delivered(kpis, request), "met": m13.met(kpis, request),
            "kept": m13.met(kpis, m13.with_targets(request, accepted)) if applied and accepted else "",
            "request_over_bound": m13.over_bound(kpis, request),
            "in_force_met": m13.met(kpis, guarantee),
            "fallbacks": sum(e["event"] == "executor_fallback" for e in events)}


def mean(rows: list[dict], key: str, label: str, instances: set[str] | None = None) -> float:
    values = [float(r[key]) for r in rows if r["label"] == label and r.get(key, "") != ""
              and (instances is None or r["instance"] in instances)]
    return float(np.mean(values)) if values else float("nan")


def aware_offered(rows: list[dict]) -> set[str]:
    """The instances where the load-aware verifier made an offer: G's subset (fixed by the verifier, not by results)."""
    return {r["instance"] for r in rows if r["label"] == "aware_offer" and r["offer"] != "null"}


def check_pairing(rows: list[dict]) -> None:
    """Every arm's first answer is a conflict, and the two load-blind arms got the same offer."""
    by: dict[str, dict] = {}
    for r in rows:
        by.setdefault(r["instance"], {})[r["label"]] = r
    for name, arms in by.items():
        if any(a["first_check"] != "conflict" for a in arms.values()):
            raise SystemExit(f"{name}: a first answer is not a conflict: {[a['first_check'] for a in arms.values()]}")
        if arms["reject"]["offer"] != arms["blind_offer"]["offer"]:
            raise SystemExit(f"{name}: the load-blind arms got different offers: not a paired comparison")


def summary(rows: list[dict]) -> dict:
    offered = aware_offered(rows)
    out = {"instances": len({r["instance"] for r in rows}), "aware_offered": len(offered)}
    for label in ARMS:
        mine = [r for r in rows if r["label"] == label]
        out[label] = {"with_offer": sum(r["offer"] != "null" for r in mine), "applied": sum(r["applied"] for r in mine),
                      "delivered": mean(rows, "delivered", label), "met": mean(rows, "met", label),
                      "kept": mean(rows, "kept", label), "in_force_met": mean(rows, "in_force_met", label),
                      "fallbacks": int(sum(r["fallbacks"] for r in mine)),
                      "aware_offered": {"delivered": mean(rows, "delivered", label, offered),
                                        "in_force_met": mean(rows, "in_force_met", label, offered)},
                      "no_aware_offer": {"delivered": mean(rows, "delivered", label, {r["instance"] for r in rows} - offered),
                                         "in_force_met": mean(rows, "in_force_met", label,
                                                              {r["instance"] for r in rows} - offered)}}
    return out


def verdict(s: dict) -> dict:
    harm = s["aware_offer"]["in_force_met"] - s["reject"]["in_force_met"]
    gain = s["aware_offer"]["aware_offered"]["delivered"] - s["reject"]["aware_offered"]["delivered"]
    gain_ok = bool(s["aware_offered"] > 0 and gain >= GAIN)
    return {"H_in_force_change": harm, "H": bool(harm >= -NO_HARM), "G_gain_where_offered": gain, "G": gain_ok,
            "both": bool(harm >= -NO_HARM and gain_ok),
            "reported_blind_offer_in_force_change": s["blind_offer"]["in_force_met"] - s["reject"]["in_force_met"]}


def score(study: str) -> dict:
    rows = [row_of(inst, label, study) for inst, label, _ in jobs(study)]
    check_pairing(rows)
    s = summary(rows)
    out = {"study": study, "summary": s, "verdict": verdict(s)}
    m13.write_csv(DATA / f"M13b-{study}-runs.csv", rows)
    (DATA / f"M13b-{study}.json").write_text(json.dumps(out, indent=1))
    return out


def regress() -> dict:
    """M13's dev set, load-aware with the offer taken, against M13's own offer arm (load-blind, already played and
    committed in docs/learn/data/M13-dev-runs.csv): every first decision, offer and delivered share must be equal."""
    play_all(jobs("regress"))
    before = {r["instance"]: r for r in csv.DictReader((DATA / "M13-dev-runs.csv").open()) if r["label"] == "offer"}
    differ = []
    for inst in instances("regress"):
        kpis, events = load_run(work_of(inst, "aware_offer", "regress"))
        intent = inst.scenario().intents[m13.rejected_index(inst)]
        first, offer, _ = m13.decisions(events, intent_id_of(events, intent))
        offered = None if offer is None else {s: v for s, v in offer["min_thr_mbps"].items() if v}
        was = before[inst.name]
        same = ((first or {}).get("check", "") == was["first_check"] and json.dumps(offered) == was["offer"]
                and abs(m13.delivered(kpis, intent) - float(was["delivered"])) < 1e-9)
        if not same:
            differ.append({"instance": inst.name, "explained": traffic_above_bounded_target(inst, kpis, intent.at_s),
                           "before": [was["first_check"], was["offer"]], "now": [(first or {}).get("check"), offered]})
    # The rule: identical, except where a bounded URLLC guarantee's recorded traffic topped its target in the 30 s
    # before the request (F6's loads are drawn up to 1.0 x the target, plus the calibrated 0.09 % noise): there the
    # new sizing applies by design.
    out = {"instances": len(before), "differ": differ, "holds": all(d["explained"] for d in differ)}
    (DATA / "M13b-regress.json").write_text(json.dumps(out, indent=1))
    return out


def traffic_above_bounded_target(inst: Instance, kpis: dict, at_s: float) -> bool:
    """Did a bounded URLLC guarantee in force (F6's first intent) carry more than its target in the 30 s before t?"""
    bounded = [i for i in inst.intents if i.max_latency_ms is not None and i.targets.get("urllc") and i.at_s < at_s]
    if not bounded:
        return False
    t, load = kpis["urllc"]["t"], kpis["urllc"]["load"]
    recent = load[(t > at_s - 30.0) & (t <= at_s + 5.0)]  # + the few seconds the agent takes to propose
    return bool(recent.size and recent.max() > bounded[0].targets["urllc"])


def missing(study: str) -> list[str]:
    return [str(work_of(i, label, study)) for i, label, _ in jobs(study)
            if not (work_of(i, label, study) / "run.json").exists()]


def play_test() -> None:
    revision = guard(name=SPENT_NAME)
    (ROOT / "test").mkdir(parents=True, exist_ok=True)
    (ROOT / "test" / "played.json").write_text(json.dumps({"commit": revision}))
    play_all(jobs("test"))


def score_test(final: bool) -> dict:
    if not final:
        raise SystemExit("the test split is scored once: add --final")
    revision = guard(name=SPENT_NAME)
    if gone := missing("test"):
        raise SystemExit(f"{len(gone)} runs missing, e.g. {gone[0]}: run --play first")
    out = score("test")
    out["played"] = json.loads((ROOT / "test" / "played.json").read_text())
    out["commit"] = revision
    result = DATA / "M13b-test.json"
    result.write_text(json.dumps(out, indent=1))
    mark_spent(SPENT_NAME, result, note=f"H and G: {json.dumps(out['verdict'])}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("--regress", "--dev", "--play", "--score"):
        group.add_argument(flag, action="store_true")
    parser.add_argument("--final", action="store_true", help="with --score: the test split, once")
    args = parser.parse_args()
    if args.regress:
        print(json.dumps(regress(), indent=1))
    elif args.dev:
        play_all(jobs("dev"))
    elif args.play:
        play_test()
    else:
        print(json.dumps(score_test(final=True) if args.final else score("dev"), indent=1, default=float))


if __name__ == "__main__":
    main()
