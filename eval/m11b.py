"""M11b and M11b′: the runtime monitor with formula fallback, on the dev instances of two cells, then M11b′'s test split.

  uv run python -m eval.m11b --probe          one instance per cell and label, timed: the cost of the whole run
  uv run python -m eval.m11b --dev [--half even|odd]    dev runs (a finished run is not played again)
  uv run python -m eval.m11b --score [--half even|odd]  the frozen rule on dev (DECISIONS 2026-11-20) -> docs/learn/data/M11b-*
From revision 3 on, triggers are tuned on the even dev indices only and the chosen one is checked once on the odd
indices before the test split (DECISIONS 2026-11-20: Souhail lifted the one-revision limit).
  uv run python -m eval.m11b --play           M11b′'s test split, once (guard: committed code and DECISIONS)
  uv run python -m eval.m11b --score --final  the rule on the test split, once; marks m11b/test spent

Same mode as M10b/M10c′: no LLM; each intent's stated facts go through the real verifier and apply step with the
default weights, and only who turns the objective into a split differs. Three labels per instance:
  rl            the cell's RL policy alone (as before M11b);
  rl_mon        the same policy with the runtime monitor (rl/monitor.py): formula_est takes over a missed segment;
  formula_est   the formula alone (M10b's frozen executor), the reference the review's promise is about.
Cells: m10 (M10c′'s 160 dev instances, F1-F3 and F5, its median-seed model) and m09 (M09b's 200 dev instances,
F1-F5, v7, the champion). Revisions 0-1 are M11b's trigger (dev only, failed as registered); 2 (and 3, if its one
revision is needed) is M11b′'s, which alone is played on the fresh m11b_test split.
Raw runs: eval/results/m11b/r<revision>/ and eval/results/m11b/test/ (git-ignored).
"""

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval import benchmark, m10b
from eval.generator import FAMILIES, M10C_FAMILIES, Instance, dev_instances, m11b_test_instances
from eval.harness import RunSpec, run_once
from eval.m09d import guard
from eval.metrics import load_run, measure_run
from eval.sim_profiles import KPI_DELAY_TICKS, channel_of, noise_std
from eval.spent import mark_spent
from eval.validate_sim import CLOCK_OFFSET_S, served_policy
from rl import m09
from rl.monitor import M11B_PRIME_STATED, Trigger
from rl.runner import SERVED_SMOOTHING_ALPHA

# 0 = M11b, 20 s in a row; 1 = M11b, 10 s in a row (both failed rule A, DECISIONS 2026-11-20); 2 = M11b′ (A and B
# failed on all of dev); 3 = M11b′ watching the stated targets only (tuned on the even half from here on).
REVISION = 3
TRIGGER: Trigger = M11B_PRIME_STATED  # the trigger of this revision, and the one the test split plays
HALVES = {"even": 0, "odd": 1}
ROOT = Path("eval/results/m11b")
RESULTS = ROOT / f"r{REVISION}"
SPENT_NAME = "m11b/test"
DATA = Path("docs/learn/data")
LABELS = ("rl", "rl_mon", "formula_est")
WORKERS = 8
# The frozen rule (DECISIONS 2026-11-20), per family on the mean intent_met of accepted intents.
PROMISE_MARGIN = 0.05  # A: rl_mon >= formula_est - 0.05 (M10c′'s equivalence margin)
NO_HARM_MARGIN = 0.02  # B: where rl >= formula_est, rl_mon >= rl - 0.02


def cells() -> dict[str, dict]:
    """Each cell's families, RL executor, policy file and the alpha that policy is served at."""
    m10c = json.loads((DATA / "M10c-policy.json").read_text())["file"]
    return {"m10": {"families": M10C_FAMILIES, "executor": "m10_rl", "policy": m10c, "alpha": m09.M10C_ALPHA},
            "m09": {"families": FAMILIES, "executor": "v7", "policy": str(served_policy()),
                    "alpha": SERVED_SMOOTHING_ALPHA}}


def spec_of(cell: dict, inst: Instance, label: str, study: str = "dev") -> RunSpec:
    executor = "formula_est" if label == "formula_est" else cell["executor"]
    alpha = cell["alpha"] if label != "formula_est" else SERVED_SMOOTHING_ALPHA
    return RunSpec(study, inst.name, "scripted", inst.seed, alpha=alpha, delay_ticks=KPI_DELAY_TICKS,
                   executor=executor, monitor=TRIGGER.name if label == "rl_mon" else None)


def results_of(study: str) -> Path:
    return RESULTS if study == "dev" else ROOT / "test"


def work_of(name: str, cell: dict, inst: Instance, label: str, study: str = "dev") -> Path:
    return results_of(study) / name / spec_of(cell, inst, label, study).run_id


def instances_of(cell: dict, study: str, half: str | None = None) -> list[Instance]:
    """The dev instances (all, or the even or odd indices of each family) or the test split's."""
    if study != "dev":
        return m11b_test_instances(cell["families"])
    every = dev_instances(cell["families"])
    return every if half is None else [i for i in every if int(i.name.rsplit("_", 1)[1]) % 2 == HALVES[half]]


def play(job: tuple[str, dict, Instance, str, str]) -> Path:
    """One label on one instance of one cell. A finished run is not played again."""
    name, cell, inst, label, study = job
    work = work_of(name, cell, inst, label, study)
    if (work / "run.json").exists():
        return work
    file = Path(cell["policy"]) if label != "formula_est" else None
    return run_once(spec_of(cell, inst, label, study), inst.scenario(), work, file,
                    cfg=benchmark.profile_cell(inst, name), traffic=inst.traffic(noise_std()),
                    clock_offset_s=CLOCK_OFFSET_S, channel=channel_of(name, inst.seed))


def jobs(study: str = "dev", half: str | None = None) -> list[tuple[str, dict, Instance, str, str]]:
    return [(name, cell, inst, label, study) for name, cell in cells().items()
            for inst in instances_of(cell, study, half) for label in LABELS]


def play_all(todo: list) -> None:
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for n, work in enumerate(pool.map(play, todo, chunksize=2), 1):
            if n % 100 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] {work}", flush=True)


def probe() -> None:
    """The first instance of each cell, every label, one process: seconds per run, then the whole run's estimate."""
    for name, cell in cells().items():
        inst = dev_instances(cell["families"], 1)[0]
        for label in LABELS:
            work = work_of(name, cell, inst, label)
            for old in work.glob("run.json"):
                old.unlink()
            started = time.monotonic()
            play((name, cell, inst, label, "dev"))
            seconds = time.monotonic() - started
            total = len(dev_instances(cell["families"]))
            print(f"{name} {label}: {seconds:.1f} s per run; {total} runs ~ {seconds * total / WORKERS / 60:.1f} min "
                  f"on {WORKERS} workers", flush=True)


# ---------------------------------------------------------------- the numbers


def fallbacks(events: list[dict]) -> list[dict]:
    return [e for e in events if e["event"] == "executor_fallback"]


def score(name: str, cell: dict, study: str = "dev", half: str | None = None) -> list[dict]:
    """One row per instance, intent and label; the verifier must answer the same in every label's run (checked)."""
    rows = []
    for inst in instances_of(cell, study, half):
        flags = None
        scenario = inst.scenario()
        for label in LABELS:
            work = work_of(name, cell, inst, label, study)
            measured = measure_run(work, scenario)["intents"]
            events = load_run(work)[1]
            mine = m10b.accepted_flags(events)
            if flags is not None and mine != flags:
                raise RuntimeError(f"{inst.name}: the verifier answered differently in the {label} run")
            flags = mine
            fired = fallbacks(events)
            for k, row in enumerate(measured):
                start, end = scenario.intents[k].window
                rows.append({"cell": name, "family": inst.family, "variant": inst.variant, "instance": inst.name,
                             "intent": k, "label": label, "accepted": flags[k], "intent_met": row["intent_met"],
                             "window_s": end - start, "fallbacks_in_run": len(fired),
                             # the event file is already on the scenario's clock (eval.harness.rebase_event_file)
                             "first_fallback_s": "" if not fired else fired[0]["ts_backend"]})
    return rows


def family_means(rows: list[dict], families) -> dict[str, dict[str, float]]:
    kept = [r for r in rows if r["accepted"]]
    out = {}
    for f in (*families, "all"):
        mine = [r for r in kept if f == "all" or r["family"] == f]
        out[f] = {label: float(np.mean([r["intent_met"] for r in mine if r["label"] == label])) for label in LABELS}
        out[f]["intents"] = len(mine) // len(LABELS)
    return out


def verdicts(means: dict[str, dict[str, float]]) -> dict[str, dict]:
    """The frozen rule. A: the review's promise; B: no harm where the RL alone was at least as good as the formula."""
    out = {}
    for f, m in means.items():
        if f == "all":
            continue
        a = m["rl_mon"] >= m["formula_est"] - PROMISE_MARGIN
        b_applies = m["rl"] >= m["formula_est"]
        b = (m["rl_mon"] >= m["rl"] - NO_HARM_MARGIN) if b_applies else None
        out[f] = {"A": a, "B": b}
    return out


def deficits(rows: list[dict]) -> list[dict]:
    """Reported, not gating: each accepted intent's shortfall of rl_mon against formula_est, next to the detection
    bound (W + KPI delay + 1 s) / window: at most the seconds before the formula can act, as a fraction of the window."""
    by = {(r["instance"], r["intent"], r["label"]): r for r in rows if r["accepted"]}
    out = []
    for (inst, k, label), r in by.items():
        if label != "rl_mon":
            continue
        gap = by[(inst, k, "formula_est")]["intent_met"] - r["intent_met"]
        bound = (TRIGGER.window_s + KPI_DELAY_TICKS + 1) / r["window_s"] if r["window_s"] > 0 else float("nan")
        out.append({"cell": r["cell"], "family": r["family"], "instance": inst, "intent": k, "deficit": gap,
                    "detection_bound": bound, "fallbacks_in_run": r["fallbacks_in_run"],
                    "first_fallback_s": r["first_fallback_s"]})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_all(study: str = "dev", half: str | None = None) -> dict:
    out, all_rows, all_deficits = {}, [], []
    for name, cell in cells().items():
        rows = score(name, cell, study, half)
        means = family_means(rows, cell["families"])
        gaps = deficits(rows)
        runs = {(r["instance"], r["label"]): r["fallbacks_in_run"] for r in rows}
        fired = sum(1 for (_, label), n in runs.items() if label == "rl_mon" and n > 0)
        over = [g for g in gaps if g["deficit"] > g["detection_bound"]]
        out[name] = {"policy": cell["policy"], "means": means, "verdicts": verdicts(means),
                     "runs_with_fallback": fired, "runs": sum(1 for (_, label) in runs if label == "rl_mon"),
                     "intents_over_detection_bound": len(over),
                     "worst_deficit": max((g["deficit"] for g in gaps), default=float("nan"))}
        all_rows += rows
        all_deficits += gaps
    out["rule"] = {"trigger": TRIGGER.__dict__, "promise_margin": PROMISE_MARGIN, "no_harm_margin": NO_HARM_MARGIN,
                   "revision": REVISION, "half": half}
    tag = "test" if study != "dev" else f"dev-r{REVISION}" + (f"-{half}" if half is not None else "")
    write_csv(DATA / f"M11b-{tag}-runs.csv", all_rows)
    write_csv(DATA / f"M11b-{tag}-deficits.csv", all_deficits)
    (DATA / f"M11b-{tag}.json").write_text(json.dumps(out, indent=1))
    return out


def missing_runs() -> list[str]:
    return [str(work_of(n, c, i, label, "test")) for n, c, i, label, _ in jobs("test")
            if not (work_of(n, c, i, label, "test") / "run.json").exists()]


def play_test() -> None:
    revision = guard(name=SPENT_NAME)
    (ROOT / "test").mkdir(parents=True, exist_ok=True)
    (ROOT / "test" / "trigger.json").write_text(json.dumps({"trigger": TRIGGER.__dict__, "revision": REVISION,
                                                            "commit": revision}))
    play_all(jobs("test"))


def score_test(final: bool) -> dict:
    if not final:
        raise SystemExit("the test split is scored once: add --final")
    revision = guard(name=SPENT_NAME)
    if missing := missing_runs():
        raise SystemExit(f"{len(missing)} runs missing, e.g. {missing[0]}: run --play first")
    out = score_all("test")
    out["played"] = json.loads((ROOT / "test" / "trigger.json").read_text())
    out["commit"] = revision
    result = DATA / "M11b-test.json"
    result.write_text(json.dumps(out, indent=1))
    verdict = {c: out[c]["verdicts"] for c in cells()}
    mark_spent(SPENT_NAME, result, note=f"rule A/B per family: {json.dumps(verdict)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--probe", action="store_true")
    group.add_argument("--dev", action="store_true")
    group.add_argument("--play", action="store_true")
    group.add_argument("--score", action="store_true")
    parser.add_argument("--final", action="store_true", help="with --score: the test split, once")
    parser.add_argument("--half", choices=sorted(HALVES), default=None, help="dev: the even or odd indices only")
    args = parser.parse_args()
    if args.probe:
        probe()
    elif args.dev:
        play_all(jobs("dev", args.half))
    elif args.play:
        play_test()
    elif args.final:
        print(json.dumps(score_test(final=True), indent=1, default=float))
    else:
        print(json.dumps(score_all("dev", args.half), indent=1, default=float))


if __name__ == "__main__":
    main()
