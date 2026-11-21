"""M13: the verifier's offer after a rejection, on the simulator (m09 cell, the served v7 policy and monitor).

  uv run python -m eval.m13 --probe           one dev instance per scripted arm, timed
  uv run python -m eval.m13 --dev             the 50 dev instances x {reject, offer}, scripted (no LLM)
  uv run python -m eval.m13 --score           rules G and H on dev (DECISIONS 2026-11-21) -> docs/learn/data/M13-dev*
  uv run python -m eval.m13 --play            the m13_test split, once (guard: committed code and DECISIONS)
  uv run python -m eval.m13 --score --final   G and H on the test split, once; marks m13/test spent
  uv run python -m eval.m13 --llm-probe       one dev instance per LLM arm, timed (needs Ollama)
  uv run python -m eval.m13 --llm-dev         LLM + RL on the 50 dev instances x {llm_nooffer, llm_offer}
  uv run python -m eval.m13 --llm-score       the LLM part, reported -> docs/learn/data/M13-llm*

The set (eval.generator.m13_instances): F4b, one target above the policy's training range (10), and F6, a URLLC
guarantee then an eMBB + best effort request that doesn't fit beside it (40). Each has one intent the verifier
rejects whatever the controller: the first-rejected intent. Arms:
  reject       scripted, as before M13: the rejected intent is dropped (the cell keeps only its implied shares)
  offer        scripted, the operator takes the verifier's offer at once (eval.harness.apply_scripted)
  llm_nooffer  LLM + RL (the proposed controller), the agent of today
  llm_offer    the same, with the offer in the model's view after the rejection (Settings.offers)
Every arm runs the served executor: v7 at alpha 0.10 with the served runtime monitor.
Raw runs: eval/results/m13/ (git-ignored).
"""

import argparse
import csv
import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from agent.llm import OllamaClient
from eval import benchmark
from eval.generator import Instance, m13_instances
from eval.harness import RunSpec, run_once
from eval.m09d import guard
from eval.metrics import in_window, intent_id_of, load_run, met_mask
from eval.scenarios import ScenarioIntent
from eval.sim_profiles import KPI_DELAY_TICKS, noise_std
from eval.spent import mark_spent
from eval.validate_sim import CLOCK_OFFSET_S, served_policy
from rl.monitor import SERVED_TRIGGER
from rl.runner import SERVED_SMOOTHING_ALPHA
from verifier.objective import SLICES

ROOT = Path("eval/results/m13")
DATA = Path("docs/learn/data")
SPENT_NAME = "m13/test"
SCRIPTED = ("reject", "offer")
LLM_ARMS = ("llm_nooffer", "llm_offer")
OFFERS = {"reject": False, "offer": True, "llm_nooffer": False, "llm_offer": True}
WORKERS = 8
# The frozen rules (DECISIONS 2026-11-21), on the scripted arms.
GAIN = 0.05  # G: mean delivered(offer) - delivered(reject) over the first-rejected intents >= +0.05
NO_HARM = 0.02  # H: mean met of F6's guarantee in force, offer >= reject - 0.02 (M11b's margin)


def rejected_index(inst: Instance) -> int:
    """The intent the verifier rejects whatever the controller: F4b's only one, F6's second."""
    return 1 if inst.family == "f6" else 0


def spec_of(inst: Instance, label: str, study: str) -> RunSpec:
    llm = label in LLM_ARMS
    return RunSpec(study, inst.name, "proposed" if llm else "scripted", inst.seed, alpha=SERVED_SMOOTHING_ALPHA,
                   delay_ticks=KPI_DELAY_TICKS, executor=None if llm else "v7", monitor=SERVED_TRIGGER.name,
                   offers=OFFERS[label])


def work_of(inst: Instance, label: str, study: str) -> Path:
    return ROOT / spec_of(inst, label, study).run_id


def split_of(study: str) -> str:
    return "m13_test" if study == "test" else "dev"


def play(job: tuple[Instance, str, str], llm=None) -> Path:
    """One arm on one instance. A finished run is not played again."""
    inst, label, study = job
    work = work_of(inst, label, study)
    if (work / "run.json").exists():
        return work
    return run_once(spec_of(inst, label, study), inst.scenario(), work, served_policy(), llm,
                    cfg=benchmark.profile_cell(inst, "m09"), traffic=inst.traffic(noise_std()),
                    clock_offset_s=CLOCK_OFFSET_S)


def jobs(study: str, labels=SCRIPTED) -> list[tuple[Instance, str, str]]:
    return [(inst, label, study) for inst in m13_instances(split_of(study)) for label in labels]


def play_scripted(todo: list) -> None:
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for n, work in enumerate(pool.map(play, todo, chunksize=2), 1):
            if n % 20 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] {work}", flush=True)


def play_llm(todo: list) -> None:
    """One at a time: one model on the GPU."""
    llm = OllamaClient()
    for n, job in enumerate(todo, 1):
        started = time.monotonic()
        work = play(job, llm)
        print(f"[{n}/{len(todo)}] {work} {time.monotonic() - started:.0f} s", flush=True)


def probe(labels) -> None:
    inst = m13_instances()[10]  # an F6 instance: the two-intent case
    llm = OllamaClient() if labels == LLM_ARMS else None
    for label in labels:
        work = work_of(inst, label, "dev")
        (work / "run.json").unlink(missing_ok=True)
        started = time.monotonic()
        play((inst, label, "dev"), llm)
        seconds = time.monotonic() - started
        n = len(m13_instances())
        per = WORKERS if llm is None else 1
        print(f"{label}: {seconds:.1f} s per run; {n} runs ~ {seconds * n / per / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- the numbers


def delivered(kpis: dict, intent: ScenarioIntent) -> float:
    """M13's primary measure: per second of the window, min(1, thr / min(target, load)) averaged over the stated
    slices, and 0 in a second where the stated latency bound is broken; the mean over the window. A graded
    intent_met, against what the operator stated (not against an offer)."""
    t = kpis["urllc"]["t"]
    parts = []
    for s, target in intent.targets.items():
        need = np.minimum(target, kpis[s]["load"])
        parts.append(np.where(need > 0, np.clip(kpis[s]["thr"] / np.where(need > 0, need, 1.0), 0.0, 1.0), 1.0))
    score = np.mean(parts, axis=0)
    if intent.max_latency_ms is not None:
        score = score * (kpis["urllc"]["lat"] <= intent.max_latency_ms)
    return float(score[in_window(t, *intent.window)].mean())


def met(kpis: dict, intent: ScenarioIntent) -> float:
    t, ok = met_mask(kpis, intent)
    return float(ok[in_window(t, *intent.window)].mean())


def over_bound(kpis: dict, intent: ScenarioIntent) -> float | None:
    if intent.max_latency_ms is None:
        return None
    window = in_window(kpis["urllc"]["t"], *intent.window)
    return float((kpis["urllc"]["lat"][window] > intent.max_latency_ms).mean())


def with_targets(intent: ScenarioIntent, targets: dict[str, float]) -> ScenarioIntent:
    """The intent as applied: other targets, same bound and window (to score a promise against what was agreed)."""
    stated = {s: targets.get(s) or None for s in SLICES}
    return dataclasses.replace(intent, expected={**intent.expected, "min_thr_mbps": stated})


def decisions(events: list[dict], intent_id: str | None) -> tuple[dict | None, dict | None, dict | None]:
    """(the verifier's first decision, its offer, the targets of the first accepted proposal) for one intent."""
    first, accepted = None, None
    for e in events:
        if e.get("intent_id") != intent_id:
            continue
        data = e.get("data", {})
        if e["event"] == "agent_step" and data.get("tool") == "set_rl_objective" and data.get("verifier"):
            first = first or data["verifier"]
            if data["verifier"].get("accepted") and accepted is None:
                accepted = {s: v for s, v in (data.get("arguments") or {}).get("min_thr_mbps", {}).items() if v}
        if e["event"] == "offer_taken" and data["verifier"].get("accepted") and accepted is None:
            accepted = {s: v for s, v in data["offer"]["min_thr_mbps"].items() if v}
    return first, (first or {}).get("offer"), accepted


def row_of(inst: Instance, label: str, study: str) -> dict:
    kpis, events = load_run(work_of(inst, label, study))
    scenario = inst.scenario()
    k = rejected_index(inst)
    intent = scenario.intents[k]
    intent_id = intent_id_of(events, intent)
    first, offer, accepted = decisions(events, intent_id)
    applied = any(e["event"] == "applied" and e.get("intent_id") == intent_id for e in events)
    offered = None if offer is None else {s: v for s, v in offer["min_thr_mbps"].items() if v}
    row = {"family": inst.family, "instance": inst.name, "label": label,
           "first_rejected": first is not None and not first["accepted"],
           "first_check": (first or {}).get("check", ""), "offer_fraction": (offer or {}).get("fraction", ""),
           "offer": json.dumps(offered), "applied": applied and accepted is not None,
           "applied_is_offer": applied and accepted is not None and accepted == offered,
           "delivered": delivered(kpis, intent), "met": met(kpis, intent),
           "kept": met(kpis, with_targets(intent, accepted)) if applied and accepted else "",
           "fallbacks": sum(e["event"] == "executor_fallback" for e in events)}
    if inst.family == "f6":  # the guarantee already in force
        row["in_force_met"] = met(kpis, scenario.intents[0])
        row["in_force_over_bound"] = over_bound(kpis, scenario.intents[0])
    return row


def mean(rows: list[dict], label: str, key: str, family: str | None = None) -> float:
    values = [float(r[key]) for r in rows if r["label"] == label and (family is None or r["family"] == family)
              and r.get(key, "") != ""]
    return float(np.mean(values)) if values else float("nan")


def summary(rows: list[dict], labels) -> dict:
    out = {}
    for label in labels:
        mine = [r for r in rows if r["label"] == label]
        out[label] = {"runs": len(mine), "first_rejected": sum(r["first_rejected"] for r in mine),
                      "with_offer": sum(r["offer"] != "null" for r in mine),
                      "applied": sum(r["applied"] for r in mine),
                      "applied_is_offer": sum(r["applied_is_offer"] for r in mine),
                      "delivered": {f: mean(rows, label, "delivered", f) for f in ("f4", "f6", None)},
                      "met": {f: mean(rows, label, "met", f) for f in ("f4", "f6", None)},
                      "kept": mean(rows, label, "kept"),
                      "in_force_met": mean(rows, label, "in_force_met", "f6"),
                      "in_force_over_bound": mean(rows, label, "in_force_over_bound", "f6")}
        out[label]["delivered"]["all"] = out[label]["delivered"].pop(None)
        out[label]["met"]["all"] = out[label]["met"].pop(None)
    return out


def same_offers(rows: list[dict], labels) -> None:
    """Both arms of an instance must get the same first answer from the verifier: same cell, same request."""
    by = {}
    for r in rows:
        by.setdefault(r["instance"], {})[r["label"]] = (r["first_check"], r["offer"])
    differ = [i for i, v in by.items() if len({v[label] for label in labels}) > 1]
    if differ:
        raise SystemExit(f"the verifier answered differently across arms on {differ[:5]}: not a paired comparison")


def verdict(s: dict) -> dict:
    gain = s["offer"]["delivered"]["all"] - s["reject"]["delivered"]["all"]
    harm = s["offer"]["in_force_met"] - s["reject"]["in_force_met"]
    return {"G_gain": gain, "G": bool(gain >= GAIN), "H_in_force_change": harm, "H": bool(harm >= -NO_HARM),
            "both": bool(gain >= GAIN and harm >= -NO_HARM)}


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def score(study: str) -> dict:
    rows = [row_of(inst, label, study) for inst, label, _ in jobs(study)]
    same_offers(rows, SCRIPTED)
    s = summary(rows, SCRIPTED)
    out = {"study": study, "summary": s, "verdict": verdict(s)}
    write_csv(DATA / f"M13-{study}-runs.csv", rows)
    (DATA / f"M13-{study}.json").write_text(json.dumps(out, indent=1))
    return out


def missing(study: str, labels=SCRIPTED) -> list[str]:
    return [str(work_of(i, label, study)) for i, label, _ in jobs(study, labels)
            if not (work_of(i, label, study) / "run.json").exists()]


def play_test() -> None:
    revision = guard(name=SPENT_NAME)
    (ROOT / "test").mkdir(parents=True, exist_ok=True)
    (ROOT / "test" / "played.json").write_text(json.dumps({"commit": revision}))
    play_scripted(jobs("test"))


def score_test(final: bool) -> dict:
    if not final:
        raise SystemExit("the test split is scored once: add --final")
    revision = guard(name=SPENT_NAME)
    if gone := missing("test"):
        raise SystemExit(f"{len(gone)} runs missing, e.g. {gone[0]}: run --play first")
    out = score("test")
    out["played"] = json.loads((ROOT / "test" / "played.json").read_text())
    out["commit"] = revision
    result = DATA / "M13-test.json"
    result.write_text(json.dumps(out, indent=1))
    mark_spent(SPENT_NAME, result, note=f"G and H: {json.dumps(out['verdict'])}")
    return out


def score_llm() -> dict:
    if gone := missing("llm", LLM_ARMS):
        raise SystemExit(f"{len(gone)} runs missing, e.g. {gone[0]}: run --llm-dev first")
    rows = [row_of(inst, label, "llm") for inst, label, _ in jobs("llm", LLM_ARMS)]
    out = {"study": "llm", "summary": summary(rows, LLM_ARMS)}
    write_csv(DATA / "M13-llm-runs.csv", rows)
    (DATA / "M13-llm.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("--probe", "--dev", "--play", "--score", "--llm-probe", "--llm-dev", "--llm-score"):
        group.add_argument(flag, action="store_true")
    parser.add_argument("--final", action="store_true", help="with --score: the test split, once")
    args = parser.parse_args()
    if args.probe:
        probe(SCRIPTED)
    elif args.dev:
        play_scripted(jobs("dev"))
    elif args.play:
        play_test()
    elif args.score:
        print(json.dumps(score_test(final=True) if args.final else score("dev"), indent=1, default=float))
    elif args.llm_probe:
        probe(LLM_ARMS)
    elif args.llm_dev:
        play_llm(jobs("llm", LLM_ARMS))
    else:
        print(json.dumps(score_llm(), indent=1, default=float))


if __name__ == "__main__":
    main()
