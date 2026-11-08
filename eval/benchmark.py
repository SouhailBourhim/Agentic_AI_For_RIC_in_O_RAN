"""The broad benchmark (M09b): every controller on the generated instances, on the calibrated simulator.

  uv run python -m eval.benchmark --parse-check                 the real model parses every dev intent; mismatches drop the instance
  uv run python -m eval.benchmark --dev [--families f1 f3] [--controllers static proposed] [--limit 5]
  uv run python -m eval.benchmark --table                       the table and the per-family summary, from finished runs

Dev seeds only: the test seeds belong to M09d (eval.generator refuses them). A finished run is never played again.
--dev with no filter refuses uncommitted code, so the full matrix comes after its rules in the history.
Raw runs: eval/results/m09b/ (git-ignored). Tables: docs/learn/data/M09b-parse.csv, M09b-runs.csv, M09b-summary.csv.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from agent.graph import llm_parse
from agent.llm import OllamaClient
from eval.extraction import score_one
from eval.generator import FAMILIES, PER_FAMILY, Instance, dev_instances
from eval.harness import USES_AGENT, RunSpec, run_once
from eval.metrics import in_window, load_run, measure_run, met_mask, over_bound_after
from eval.sim_profiles import KPI_DELAY_TICKS, noise_std, testbed_cell
from eval.testbed_control import commit, guard_final
from eval.validate_sim import CLOCK_OFFSET_S, served_policy
from rl.runner import SERVED_SMOOTHING_ALPHA
from store.events import read_events
from verifier.objective import SLICES

RESULTS = Path("eval/results/m09b")
PARSE = Path("docs/learn/data/M09b-parse.csv")
RUNS = Path("docs/learn/data/M09b-runs.csv")
SUMMARY = Path("docs/learn/data/M09b-summary.csv")
CONTROLLERS = ("static", "rl_only", "llm_only", "llm_only_v2", "proposed", "formula", "oracle")
STUDY = "m09b_dev"
FEASIBLE = 0.9  # the oracle meets it in at least 90 % of the seconds (DECISIONS)
BOOTSTRAP = 2000


def profile_cell(instance: Instance):
    return testbed_cell("m09", instance.seed)


def work_of(instance: Instance, controller: str) -> Path:
    return RESULTS / RunSpec(STUDY, instance.name, controller, instance.seed).run_id


# ---------------------------------------------------------------- the parse check (H7)


def parse_check(instances: list[Instance]) -> list[dict]:
    llm = OllamaClient()
    rows = []
    for inst in instances:
        for k, intent in enumerate(inst.intents, 1):
            facts, latency, _ = llm_parse(llm, intent.text)
            verdict = score_one(intent.expected, facts)
            rows.append({"instance": inst.name, "intent": k, "text": intent.text, "correct": verdict["correct"],
                         "wrong_fields": " ".join(f for f, ok in verdict["fields"].items() if not ok), "latency_s": round(latency, 2)})
        print(f"{inst.name}: {[r['correct'] for r in rows if r['instance'] == inst.name]}", flush=True)
    PARSE.parent.mkdir(parents=True, exist_ok=True)
    with PARSE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def dropped() -> set[str]:
    """Instances with an intent the model didn't parse as stated: left out, and counted (H7)."""
    if not PARSE.exists():
        raise SystemExit("run --parse-check first: only intents with verified extraction are used (H7)")
    with PARSE.open() as f:
        return {r["instance"] for r in csv.DictReader(f) if r["correct"] != "True"}


# ---------------------------------------------------------------- the runs


def play(instance: Instance, controller: str, policy: Path, llm) -> Path:
    work = work_of(instance, controller)
    cell = profile_cell(instance)
    traffic = instance.traffic(noise_std())
    if (work / "run.json").exists():
        return work
    delay = 0 if controller == "oracle" else KPI_DELAY_TICKS  # the oracle reads the cell itself, no late view
    spec = RunSpec(STUDY, instance.name, controller, instance.seed, alpha=SERVED_SMOOTHING_ALPHA, delay_ticks=delay)
    return run_once(spec, instance.scenario(), work, policy, llm if controller in USES_AGENT else None,
                    cfg=cell, traffic=traffic, clock_offset_s=CLOCK_OFFSET_S)


def play_matrix(families, controllers, limit: int | None) -> None:
    skip = dropped()
    instances = [i for i in dev_instances(families, limit or PER_FAMILY) if i.name not in skip]
    print(f"{len(instances)} instances ({len(skip)} dropped by the parse check), {len(controllers)} controllers", flush=True)
    llm, policy = OllamaClient(), served_policy()
    RESULTS.mkdir(parents=True, exist_ok=True)
    total = len(instances) * len(controllers)
    for n, (inst, controller) in enumerate(((i, c) for i in instances for c in controllers), 1):
        work = play(inst, controller, policy, llm)
        wall = json.loads((work / "run.json").read_text()).get("wall_s", "")
        print(f"[{n}/{total}] {inst.name} {controller}: {wall} s", flush=True)


# ---------------------------------------------------------------- the table


def over_serving(kpis: dict, intent, cfg_told_capacity: float, base_ms: float) -> float | None:
    """Mean share above need, over the window, of the slices the intent names: what over-serving costs."""
    t = kpis["urllc"]["t"]
    window = in_window(t, *intent.window)
    if not window.any():
        return None
    extra = []
    for s, target in intent.targets.items():
        k = kpis[s]
        need = np.minimum(target, k["load"]) / cfg_told_capacity
        if s == "urllc" and intent.max_latency_ms is not None:
            need = k["load"] / cfg_told_capacity / max(1.0 - base_ms / intent.max_latency_ms, 1e-3)
        extra.append(float(np.mean(k["share"][window] - need[window])))
    return float(np.mean(extra))


def outcome_of(events: list[dict], text: str) -> str:
    """What the agent did with this intent: applied, rejected, or nothing (static, RL only, the oracle)."""
    ids = [e["intent_id"] for e in events if e["event"] == "received" and e["data"].get("text") == text]
    done = [e["data"].get("outcome", "") for e in events if e["event"] == "agent_done" and e["intent_id"] in ids]
    return done[-1] if done else ""


def jointly_feasible(oracle_kpis: dict, intents) -> bool | None:
    """All intents met at once in at least 90 % of the seconds they overlap (None if they never overlap)."""
    start, end = max(i.window[0] for i in intents), min(i.window[1] for i in intents)
    if end <= start:
        return None
    t = oracle_kpis["urllc"]["t"]
    together = np.ones(len(t), dtype=bool)
    for intent in intents:
        together &= met_mask(oracle_kpis, intent)[1]
    keep = in_window(t, start, end)
    return bool(together[keep].mean() >= FEASIBLE)


def write_table() -> list[dict]:
    rows = []
    for inst in dev_instances():
        oracle_work = work_of(inst, "oracle")
        if not (oracle_work / "run.json").exists():
            continue
        oracle_kpis = load_run(oracle_work)[0]
        oracle_met = [measure_run(oracle_work, inst.scenario())["intents"][k]["intent_met"] for k in range(len(inst.intents))]
        joint = jointly_feasible(oracle_kpis, inst.intents) if len(inst.intents) > 1 else None
        cell = profile_cell(inst)
        for controller in CONTROLLERS:
            work = work_of(inst, controller)
            if not (work / "run.json").exists():
                continue
            measured = measure_run(work, inst.scenario())
            kpis, events = load_run(work)
            for k, (row, intent) in enumerate(zip(measured["intents"], inst.intents, strict=True)):
                after = over_bound_after(kpis, intent)
                serve = over_serving(kpis, intent, cell.told_capacity_mbps, cell.told_urllc_base_latency_ms)
                rows.append({"family": inst.family, "variant": inst.variant, "instance": inst.name, "seed": inst.seed,
                             "controller": controller, **row, "oracle_met": oracle_met[k],
                             "feasible": oracle_met[k] >= FEASIBLE, "jointly_feasible": "" if joint is None else joint,
                             "of_oracle": row["intent_met"] / oracle_met[k] if oracle_met[k] > 0 else "",
                             "over_serving": "" if serve is None else serve, "over_bound_after_s": "" if after is None else after,
                             "outcome": outcome_of(events, intent.text), "true_capacity_mbps": round(cell.capacity_mbps, 2),
                             **measured["run"]})
    if rows:
        RUNS.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with RUNS.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def bootstrap_ci(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95 % interval of the median by resampling instances (the unit of the benchmark)."""
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(values, size=(BOOTSTRAP, len(values)), replace=True), axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def summarise(rows: list[dict]) -> list[dict]:
    """Per family and controller: median intent met with its interval; and the median paired difference
    with LLM + RL, instance by instance and intent by intent, with its interval."""
    out = []
    for family in FAMILIES:
        mine = [r for r in rows if r["family"] == family]
        ours = {(r["instance"], r["intent"]): r["intent_met"] for r in mine if r["controller"] == "proposed"}
        for controller in CONTROLLERS:
            theirs = {(r["instance"], r["intent"]): r["intent_met"] for r in mine if r["controller"] == controller}
            values = np.array(list(theirs.values()))
            if not len(values):
                continue
            paired = np.array([ours[k] - v for k, v in theirs.items() if k in ours])
            low, high = bootstrap_ci(values)
            d_low, d_high = bootstrap_ci(paired) if controller != "proposed" else (float("nan"), float("nan"))
            out.append({"family": family, "controller": controller, "n": len(values), "median_met": round(float(np.median(values)), 3),
                        "ci_low": round(low, 3), "ci_high": round(high, 3),
                        "proposed_minus_this": round(float(np.median(paired)), 3) if controller != "proposed" else "",
                        "diff_ci_low": round(d_low, 3) if controller != "proposed" else "", "diff_ci_high": round(d_high, 3) if controller != "proposed" else ""})
    with SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--parse-check", action="store_true")
    group.add_argument("--dev", action="store_true")
    group.add_argument("--table", action="store_true")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS))
    parser.add_argument("--limit", type=int, help="instances per family (default all 40)")
    args = parser.parse_args()
    if args.parse_check:
        rows = parse_check(dev_instances(args.families, args.limit or PER_FAMILY))
        print(f"{sum(r['correct'] for r in rows)} of {len(rows)} intents parsed as stated")
        return
    if args.dev:
        if args.limit is None and set(args.families) == set(FAMILIES) and set(args.controllers) == set(CONTROLLERS):
            guard_final(commit())
        play_matrix(args.families, args.controllers, args.limit)
        return
    rows = write_table()
    for r in summarise(rows):
        diff = f"  LLM+RL minus this {r['proposed_minus_this']} [{r['diff_ci_low']}, {r['diff_ci_high']}]" if r["proposed_minus_this"] != "" else ""
        print(f"{r['family']} {r['controller']:12s} n={r['n']:3d} median {r['median_met']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]{diff}")


if __name__ == "__main__":
    main()
