"""The broad benchmark (M09b): every controller on the generated instances, on the calibrated simulator.

  uv run python -m eval.benchmark --parse-check                 the real model parses every dev intent; mismatches drop the instance
  uv run python -m eval.benchmark --dev [--families f1 f3] [--controllers static proposed] [--limit 5]
  uv run python -m eval.benchmark --table                       the table and the per-family summary, from finished runs
  uv run python -m eval.benchmark --dev --controllers proposed --policy <zip> --study m09c_<name>   M09c: another policy
  uv run python -m eval.benchmark --compare m09c_<name>          M09c: LLM + RL with it against v7's runs (selection rule)

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

RESULTS = Path("eval/results/m09b")
PARSE = Path("docs/learn/data/M09b-parse.csv")
RUNS = Path("docs/learn/data/M09b-runs.csv")
SUMMARY = Path("docs/learn/data/M09b-summary.csv")
SELECTION = Path("docs/learn/data/M09c-selection.csv")
FAMILY_DROP = 0.02  # M09c: a new policy may not lower any family's mean intent met by more than this (DECISIONS)
CONTROLLERS = ("static", "rl_only", "llm_only", "llm_only_v2", "proposed", "formula", "oracle", "formula_a10")
STUDY = "m09b_dev"
# Controllers added after M09b keep their dev runs in their own folder, so M09b's runs (and their fingerprint in
# the evidence manifest) stay as they were; the table reads them from there, next to the others.
OWN_STUDY = {"formula_a10": "m09g_formula_a10"}
FEASIBLE = 0.9  # the oracle meets it in at least 90 % of the seconds (DECISIONS)
CEILING_MARGIN, CEILING_SHARE = 0.05, 0.05  # beaten by > 0.05 on > 5 % of intents: a reference, not a ceiling (DECISIONS)
BOOTSTRAP = 2000


def profile_cell(instance: Instance):
    return testbed_cell("m09", instance.seed)


def study_of(controller: str, study: str = STUDY) -> str:
    return OWN_STUDY.get(controller, study) if study == STUDY else study


def work_of(instance: Instance, controller: str, study: str = STUDY, root: Path = RESULTS) -> Path:
    """Where a run lives. `root` lets M09d keep the test runs apart (eval/results/m09d)."""
    return root / RunSpec(study_of(controller, study), instance.name, controller, instance.seed).run_id


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


def play(instance: Instance, controller: str, policy: Path, llm, study: str = STUDY, root: Path = RESULTS) -> Path:
    work = work_of(instance, controller, study, root)
    cell = profile_cell(instance)
    traffic = instance.traffic(noise_std())
    if (work / "run.json").exists():
        return work
    delay = 0 if controller == "oracle" else KPI_DELAY_TICKS  # the oracle reads the cell itself, no late view
    spec = RunSpec(study_of(controller, study), instance.name, controller, instance.seed, alpha=SERVED_SMOOTHING_ALPHA,
                   delay_ticks=delay)
    return run_once(spec, instance.scenario(), work, policy, llm if controller in USES_AGENT else None,
                    cfg=cell, traffic=traffic, clock_offset_s=CLOCK_OFFSET_S)


def play_matrix(families, controllers, limit: int | None, policy_file: Path | None = None, study: str = STUDY) -> None:
    skip = dropped()
    instances = [i for i in dev_instances(families, limit or PER_FAMILY) if i.name not in skip]
    print(f"{len(instances)} instances ({len(skip)} dropped by the parse check), {len(controllers)} controllers", flush=True)
    llm, policy = OllamaClient(), policy_file or served_policy()
    for folder in {study_of(c, study) for c in controllers}:  # only the folders this call plays into (M09g)
        (RESULTS / folder).mkdir(parents=True, exist_ok=True)
        (RESULTS / folder / "policy.json").write_text(json.dumps({"policy": str(policy)}))  # what --compare names
    total = len(instances) * len(controllers)
    for n, (inst, controller) in enumerate(((i, c) for i in instances for c in controllers), 1):
        work = play(inst, controller, policy, llm, study)
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


def table_rows(instances: list[Instance], controllers=CONTROLLERS, study: str = STUDY, root: Path = RESULTS) -> list[dict]:
    """One row per finished run and intent, with the oracle's numbers next to it (M09b; M09d reuses it)."""
    rows = []
    for inst in instances:
        oracle_work = work_of(inst, "oracle", study, root)
        if not (oracle_work / "run.json").exists():
            continue
        oracle_kpis = load_run(oracle_work)[0]
        oracle_met = [measure_run(oracle_work, inst.scenario())["intents"][k]["intent_met"] for k in range(len(inst.intents))]
        joint = jointly_feasible(oracle_kpis, inst.intents) if len(inst.intents) > 1 else None
        cell = profile_cell(inst)
        for controller in controllers:
            work = work_of(inst, controller, study, root)
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
    return rows


def write_table() -> list[dict]:
    rows = table_rows(dev_instances())
    if rows:
        RUNS.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with RUNS.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def bootstrap_ci(values: np.ndarray, seed: int = 0, stat=np.median) -> tuple[float, float]:
    """95 % interval of a statistic (median by default) by resampling instances (the unit of the benchmark)."""
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = stat(rng.choice(values, size=(BOOTSTRAP, len(values)), replace=True), axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarise(rows: list[dict]) -> list[dict]:
    """Per family and controller: mean and median intent met with their intervals; and the paired difference with
    LLM + RL, intent by intent, as a mean and a median with intervals. The mean is the one to read: intent met is
    often exactly 0 or 1, so a median difference jumps between ties (dev analysis; M09d pre-registers its own)."""
    out = []
    for family in FAMILIES:
        mine = [r for r in rows if r["family"] == family]
        ours = {(r["instance"], r["intent"]): r["intent_met"] for r in mine if r["controller"] == "proposed"}
        for controller in CONTROLLERS:
            theirs = {(r["instance"], r["intent"]): r["intent_met"] for r in mine if r["controller"] == controller}
            values = np.array(list(theirs.values()))
            if not len(values):
                continue
            row = {"family": family, "controller": controller, "n": len(values)}
            for name, stat in (("mean", np.mean), ("median", np.median)):
                low, high = bootstrap_ci(values, stat=stat)
                row |= {f"{name}_met": round(float(stat(values)), 3), f"{name}_low": round(low, 3), f"{name}_high": round(high, 3)}
            paired = np.array([ours[k] - v for k, v in theirs.items() if k in ours])
            for name, stat in (("mean", np.mean), ("median", np.median)):
                if controller == "proposed":
                    row |= {f"{name}_diff": "", f"{name}_diff_low": "", f"{name}_diff_high": ""}
                    continue
                low, high = bootstrap_ci(paired, stat=stat)
                row |= {f"{name}_diff": round(float(stat(paired)), 3), f"{name}_diff_low": round(low, 3), f"{name}_diff_high": round(high, 3)}
            # What intent met can't see: URLLC's seconds over the bound in the 60 s after a bounded intent's window
            # (M08d's swing at expiry). Only intents with a bound whose window ends before the run does.
            after = np.array([float(r["over_bound_after_s"]) for r in mine
                              if r["controller"] == controller and r["over_bound_after_s"] != ""])
            row |= {"n_after": len(after), "mean_over_bound_after_s": round(float(after.mean()), 2) if len(after) else ""}
            out.append(row)
    with SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    return out


def ceiling_check(rows: list[dict]) -> dict[str, float]:
    """Per family, the share of intents on which some controller beats the oracle by more than CEILING_MARGIN.
    Above CEILING_SHARE overall, the oracle is reported as a reference, not a ceiling (DECISIONS)."""
    oracle = {(r["instance"], r["intent"]): r["intent_met"] for r in rows if r["controller"] == "oracle"}
    beaten: dict[tuple, bool] = {k: False for k in oracle}
    for r in rows:
        k = (r["instance"], r["intent"])
        if r["controller"] != "oracle" and k in oracle and r["intent_met"] > oracle[k] + CEILING_MARGIN:
            beaten[k] = True
    family = {k: next(r["family"] for r in rows if (r["instance"], r["intent"]) == k) for k in oracle}
    out = {f: float(np.mean([b for k, b in beaten.items() if family[k] == f])) for f in FAMILIES if any(family[k] == f for k in oracle)}
    out["all"] = float(np.mean(list(beaten.values())))
    return out


def proposed_met(study: str) -> dict[tuple[str, int], tuple[str, float]]:
    """LLM + RL's intent met per (instance, intent) in a study's finished runs, with the family."""
    out = {}
    for inst in dev_instances():
        work = work_of(inst, "proposed", study)
        if (work / "run.json").exists():
            for k, row in enumerate(measure_run(work, inst.scenario())["intents"]):
                out[(inst.name, k)] = (inst.family, row["intent_met"])
    return out


def compare(study: str) -> dict:
    """M09c's selection rule: LLM + RL with the study's policy against v7's runs (the M09b dev benchmark), paired by
    intent. It beats v7 if its mean intent met is higher and no family's mean is lower by more than FAMILY_DROP.
    Appends to docs/learn/data/M09c-selection.csv and records the verdict for the registry gate (mlops.registry)."""
    from mlops import tracking

    new, old = proposed_met(study), proposed_met(STUDY)
    keys = sorted(set(new) & set(old))
    if not keys:
        raise SystemExit(f"no LLM + RL runs in both {study} and {STUDY}")
    policy = Path(json.loads((RESULTS / study / "policy.json").read_text())["policy"])
    row: dict = {"study": study, "policy": policy.stem, "intents": len(keys),
                 "mean_met": round(float(np.mean([new[k][1] for k in keys])), 4),
                 "v7_mean_met": round(float(np.mean([old[k][1] for k in keys])), 4)}
    below = 0
    for family in FAMILIES:
        mine = [k for k in keys if new[k][0] == family]
        if mine:
            a, b = float(np.mean([new[k][1] for k in mine])), float(np.mean([old[k][1] for k in mine]))
            row |= {f"{family}_met": round(a, 4), f"{family}_v7": round(b, 4)}
            below += a < b
            row.setdefault("drops", [])
            if a < b - FAMILY_DROP:
                row["drops"].append(family)
    drops = row.pop("drops", [])
    row |= {"families_below_v7": below, "families_dropped": " ".join(drops),
            "beats_champion": row["mean_met"] > row["v7_mean_met"] and not drops}
    new_file = not SELECTION.exists()
    with SELECTION.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    tracking.log_evaluation("benchmark", policy.stem, {k: v for k, v in row.items() if isinstance(v, float)},
                            {"study": study, "beats_champion": str(row["beats_champion"]), "champion_runs": STUDY})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--parse-check", action="store_true")
    group.add_argument("--dev", action="store_true")
    group.add_argument("--table", action="store_true")
    group.add_argument("--compare", metavar="STUDY", help="M09c: LLM + RL in STUDY against v7's dev runs")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS))
    parser.add_argument("--limit", type=int, help="instances per family (default all 40)")
    parser.add_argument("--policy", type=Path, help="M09c: play with this policy file instead of the served v7")
    parser.add_argument("--study", default=STUDY, help="M09c: results folder for a --policy run (e.g. m09c_v8)")
    args = parser.parse_args()
    if args.compare:
        print(compare(args.compare))
        return
    if args.policy is not None and args.study == STUDY:
        parser.error("--policy needs its own --study, so v7's runs are never overwritten")
    if args.parse_check:
        rows = parse_check(dev_instances(args.families, args.limit or PER_FAMILY))
        print(f"{sum(r['correct'] for r in rows)} of {len(rows)} intents parsed as stated")
        return
    if args.dev:
        full = args.limit is None and set(args.families) == set(FAMILIES)
        if full and (set(args.controllers) == set(CONTROLLERS) or args.policy is not None or set(args.controllers) & set(OWN_STUDY)):
            guard_final(commit())  # a full matrix, a selection run (M09c) or a diagnostic (M09g) comes after its rules
        play_matrix(args.families, args.controllers, args.limit, args.policy, args.study)
        return
    rows = write_table()
    for r in summarise(rows):
        diff = f"  LLM+RL minus this {r['mean_diff']:+.3f} [{r['mean_diff_low']:+.3f}, {r['mean_diff_high']:+.3f}]" if r["mean_diff"] != "" else ""
        print(f"{r['family']} {r['controller']:12s} n={r['n']:3d} mean {r['mean_met']:.2f} [{r['mean_low']:.2f}, {r['mean_high']:.2f}]{diff}")
    check = ceiling_check(rows)
    verdict = "a ceiling" if check["all"] <= CEILING_SHARE else "a reference, not a ceiling"
    print("intents where a controller beats the oracle by > 0.05:", {k: round(v, 3) for k, v in check.items()}, f"-> the oracle is {verdict}")


if __name__ == "__main__":
    main()
