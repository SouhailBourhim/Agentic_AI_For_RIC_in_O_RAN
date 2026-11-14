"""M10b: the non-learned executors on the m10 cell, and the two checks that decide whether M10 goes on to train RL.

  uv run python -m eval.m10b --tune --revision N --note "what changed"   formula_est on the even dev instances
  uv run python -m eval.m10b --check [--cell-revision N]                  all 7 executors on the odd ones, once per cell revision

No LLM and no GPU. Every run is the harness's "scripted" controller with an executor (eval.harness.EXECUTORS): each
intent's stated facts go through the real verifier and apply step with the default weights, so the objective is the
same for every executor and only the step from objective to split differs.
- Tuning (even indices): formula_est's estimate and margin (rl/baselines.py) change within a budget of
  MAX_REVISIONS, each logged in docs/learn/data/M10b-formula-revisions.csv with what it gave.
- Checks (odd indices), rules in DECISIONS before the run (`checks` below):
  difficulty: the oracle meets >= 0.90 of the window on >= 80 % of the accepted intents;
  headroom: mean over accepted intents of the best any executor got, minus the best non-learned candidate's mean,
  >= 0.05; comparator for M10c: the best candidate, the simpler one within 0.01.
Raw runs: eval/results/m10b/ (git-ignored). Tables: docs/learn/data/M10b-*-r<cell revision>.csv and .json.
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval import benchmark
from eval.generator import FAMILIES, Instance, dev_instances
from eval.harness import EXECUTORS, RunSpec, run_once
from eval.m09d import bootstrap_means, guard, interval
from eval.metrics import load_run, measure_run
from eval.sim_profiles import KPI_DELAY_TICKS, channel_of, noise_std
from eval.validate_sim import CLOCK_OFFSET_S, served_policy
from rl import baselines
from rl.runner import SERVED_SMOOTHING_ALPHA

PROFILE = "m10"
RESULTS = Path("eval/results/m10b")
DATA = Path("docs/learn/data")
REVISIONS_CSV = DATA / "M10b-formula-revisions.csv"
MAX_REVISIONS = 3
WORKERS = 8  # processes: run_once sets RIC_DB / RIC_EVENTS in its environment, so no threads
CANDIDATES = ("rev4", "formula_est", "mpc_est")  # simplest first: the tie rule prefers the earlier one
DIFFICULTY_MET, DIFFICULTY_SHARE = 0.90, 0.80
HEADROOM = 0.05
TIE = 0.01
TUNE_REFERENCES = ("rev4", "oracle")  # played once on the even indices, for context next to each revision


def half(parity: int) -> list[Instance]:
    """The even (0) or odd (1) dev instances of every family: tuning on one half, checks on the other."""
    return [inst for inst in dev_instances() if inst.index % 2 == parity]


def work_of(inst: Instance, executor: str, study: str) -> Path:
    return RESULTS / RunSpec(study, inst.name, "scripted", inst.seed, executor=executor).run_id


def play(job: tuple[Instance, str, str]) -> Path:
    """One executor on one instance of the m10 cell. A finished run is not played again."""
    inst, executor, study = job
    work = work_of(inst, executor, study)
    if (work / "run.json").exists():
        return work
    delay = 0 if executor == "oracle" else KPI_DELAY_TICKS  # the oracle reads the cell itself, as in M09b
    spec = RunSpec(study, inst.name, "scripted", inst.seed, alpha=SERVED_SMOOTHING_ALPHA, delay_ticks=delay,
                   executor=executor)
    return run_once(spec, inst.scenario(), work, served_policy() if executor == "v7" else None,
                    cfg=benchmark.profile_cell(inst, PROFILE), traffic=inst.traffic(noise_std()),
                    clock_offset_s=CLOCK_OFFSET_S, channel=channel_of(PROFILE, inst.seed))


def play_all(jobs: list[tuple[Instance, str, str]]) -> None:
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for n, work in enumerate(pool.map(play, jobs, chunksize=4), 1):
            if n % 50 == 0 or n == len(jobs):
                print(f"[{n}/{len(jobs)}] {work}", flush=True)


# ---------------------------------------------------------------- scoring


def accepted_flags(events: list[dict]) -> list[bool]:
    """The verifier's answer to each scripted intent, in submission order (apply_scripted logs one per intent)."""
    return [bool(e["data"]["verifier"]["accepted"]) for e in events
            if e["event"] == "agent_step" and e["data"].get("tool") == "set_rl_objective"]


def score(instances: list[Instance], executors, study: str) -> list[dict]:
    """One row per instance, intent and executor. `accepted` comes from each run's own log; the verifier is
    deterministic and sees the same applied intents in every run, so it must agree across executors (checked)."""
    rows = []
    for inst in instances:
        flags = None
        for executor in executors:
            work = work_of(inst, executor, study)
            measured = measure_run(work, inst.scenario())["intents"]
            mine = accepted_flags(load_run(work)[1])
            if flags is not None and mine != flags:
                raise RuntimeError(f"{inst.name}: the verifier answered differently in the {executor} run")
            flags = mine
            for k, row in enumerate(measured):
                rows.append({"family": inst.family, "variant": inst.variant, "instance": inst.name, "intent": k,
                             "executor": executor, "accepted": flags[k], "intent_met": row["intent_met"]})
    return rows


def checks(rows: list[dict]) -> dict:
    """The pre-registered checks, on accepted intents only. Pure: a test runs it on a synthetic table."""
    kept = [r for r in rows if r["accepted"]]
    executors = list(dict.fromkeys(r["executor"] for r in kept))
    by = {(r["instance"], r["intent"], r["executor"]): r["intent_met"] for r in kept}
    intents = sorted({(r["instance"], r["intent"]) for r in kept})
    met = {e: np.array([by[(i, k, e)] for i, k in intents]) for e in executors}
    mean = {e: float(v.mean()) for e, v in met.items()}
    envelope = np.max(np.array([met[e] for e in executors]), axis=0)
    best = max(mean[c] for c in CANDIDATES)
    comparator = next(c for c in CANDIDATES if mean[c] >= best - TIE)  # the simplest within TIE of the best
    oracle_feasible = float((met["oracle"] >= DIFFICULTY_MET).mean())
    headroom = float(envelope.mean()) - best
    return {"intents": len(intents), "means": mean, "envelope": float(envelope.mean()),
            "oracle_feasible_share": oracle_feasible, "difficulty_pass": oracle_feasible >= DIFFICULTY_SHARE,
            "best_candidate_mean": best, "headroom": headroom, "headroom_pass": headroom >= HEADROOM,
            "comparator": comparator}


def paired_interval(rows: list[dict], a: str, b: str) -> tuple[float, float, float]:
    """Mean of a minus b over accepted intents, with a 95 % cluster-bootstrap interval by instance (M09d)."""
    kept = [r for r in rows if r["accepted"]]
    by = {(r["instance"], r["intent"], r["executor"]): r["intent_met"] for r in kept}
    groups = []
    for name in dict.fromkeys(r["instance"] for r in kept):
        ks = sorted({r["intent"] for r in kept if r["instance"] == name})
        groups.append(np.array([by[(name, k, a)] - by[(name, k, b)] for k in ks]))
    draws = bootstrap_means(groups)
    total = np.concatenate(groups)
    return float(total.mean()), *interval(draws, 0.95)


# ---------------------------------------------------------------- tuning and checks


def family_means(rows: list[dict], executor: str) -> dict[str, float]:
    kept = [r for r in rows if r["accepted"] and r["executor"] == executor]
    out = {"all": float(np.mean([r["intent_met"] for r in kept]))}
    for f in FAMILIES:
        mine = [r["intent_met"] for r in kept if r["family"] == f]
        out[f] = float(np.mean(mine)) if mine else float("nan")
    return out


def tune(revision: int, note: str) -> None:
    if not 1 <= revision <= MAX_REVISIONS:
        raise SystemExit(f"the budget is {MAX_REVISIONS} revisions")
    done = list(csv.DictReader(REVISIONS_CSV.open())) if REVISIONS_CSV.exists() else []
    if any(int(r["revision"]) == revision for r in done):
        raise SystemExit(f"revision {revision} is already logged: the next one is {len(done) + 1}")
    instances = half(0)
    study = f"tune_rev{revision}"
    play_all([(i, e, "tune_ref") for i in instances for e in TUNE_REFERENCES] + [(i, "formula_est", study) for i in instances])
    rows = score(instances, ["formula_est"], study) + score(instances, TUNE_REFERENCES, "tune_ref")
    est = family_means([r for r in rows if r["executor"] == "formula_est"], "formula_est")
    refs = {e: family_means(rows, e)["all"] for e in TUNE_REFERENCES}
    row = {"revision": revision, "note": note, "estimate": baselines.FORMULA_EST_ESTIMATE,
           "margin": round(baselines.FORMULA_EST_MARGIN, 4), "volatility": baselines.FORMULA_EST_VOLATILITY,
           **{f"met_{k}": round(v, 4) for k, v in est.items()}, **{f"{e}_all": round(v, 4) for e, v in refs.items()}}
    write_rows(REVISIONS_CSV, done + [row])
    print(json.dumps(row, indent=1))


def check(cell_revision: int) -> None:
    study = f"check_r{cell_revision}"
    revision = guard(name=f"m10b/{study}")  # code and DECISIONS committed before the run
    result = RESULTS / study / "result.json"
    if result.exists():
        raise SystemExit(f"{result} exists: the check runs once per cell revision")
    instances = half(1)
    (RESULTS / study).mkdir(parents=True, exist_ok=True)
    play_all([(i, e, study) for i in instances for e in EXECUTORS])
    rows = score(instances, EXECUTORS, study)
    verdicts = checks(rows)
    verdicts["foresight"] = paired_interval(rows, "oracle", "formula_told")
    verdicts["estimation"] = paired_interval(rows, "formula_told", "formula_est")
    verdicts["families"] = {e: family_means(rows, e) for e in EXECUTORS}
    verdicts["commit"] = revision
    write_rows(DATA / f"M10b-runs-r{cell_revision}.csv", rows)  # one pair of tables per cell revision
    write_rows(DATA / f"M10b-checks-r{cell_revision}.csv", [{"executor": e, **{k: round(v, 4) for k, v in m.items()}} for e, m in verdicts["families"].items()])
    result.write_text(json.dumps(verdicts, indent=1))
    (DATA / f"M10b-result-r{cell_revision}.json").write_text(json.dumps(verdicts, indent=1))  # committed copy
    print(json.dumps({k: v for k, v in verdicts.items() if k != "families"}, indent=1))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--revision", type=int)
    parser.add_argument("--note", default="")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--cell-revision", type=int, default=1)
    args = parser.parse_args()
    if args.tune:
        if args.revision is None:
            parser.error("--tune needs --revision")
        tune(args.revision, args.note)
    elif args.check:
        check(args.cell_revision)
    else:
        parser.error("choose --tune or --check")


if __name__ == "__main__":
    main()
