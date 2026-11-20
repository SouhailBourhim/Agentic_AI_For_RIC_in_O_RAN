"""M11b: the runtime monitor with formula fallback, on the dev instances of two cells. Development evidence only.

  uv run python -m eval.m11b --probe        one instance per cell and label, timed: the cost of the whole run
  uv run python -m eval.m11b --dev          every run (a finished run is not played again)
  uv run python -m eval.m11b --score        the frozen rule (DECISIONS 2026-11-20) -> docs/learn/data/M11b-*

Same mode as M10b/M10c′: no LLM; each intent's stated facts go through the real verifier and apply step with the
default weights, and only who turns the objective into a split differs. Three labels per instance:
  rl            the cell's RL policy alone (as before M11b);
  rl_mon        the same policy with the runtime monitor (rl/monitor.py): formula_est takes over a missed segment;
  formula_est   the formula alone (M10b's frozen executor), the reference the review's promise is about.
Cells: m10 (M10c′'s 160 dev instances, F1-F3 and F5, its median-seed model) and m09 (M09b's 200 dev instances,
F1-F5, v7, the champion). No test split is touched.
Raw runs: eval/results/m11b/r<revision>/ (git-ignored).
"""

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval import benchmark, m10b
from eval.generator import FAMILIES, M10C_FAMILIES, Instance, dev_instances
from eval.harness import RunSpec, run_once
from eval.metrics import load_run, measure_run
from eval.sim_profiles import KPI_DELAY_TICKS, channel_of, noise_std
from eval.validate_sim import CLOCK_OFFSET_S, served_policy
from rl import m09
from rl.monitor import FALLBACK_AFTER_S
from rl.runner import SERVED_SMOOTHING_ALPHA

# Revision of the frozen rule's one allowed revision: 0 = N 20 s (rule A failed), 1 = N 10 s (DECISIONS 2026-11-20).
REVISION = 1
RESULTS = Path(f"eval/results/m11b/r{REVISION}")
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


def spec_of(cell: dict, inst: Instance, label: str) -> RunSpec:
    executor = "formula_est" if label == "formula_est" else cell["executor"]
    alpha = cell["alpha"] if label != "formula_est" else SERVED_SMOOTHING_ALPHA
    return RunSpec("dev", inst.name, "scripted", inst.seed, alpha=alpha, delay_ticks=KPI_DELAY_TICKS,
                   executor=executor, monitor=label == "rl_mon")


def work_of(name: str, cell: dict, inst: Instance, label: str) -> Path:
    return RESULTS / name / spec_of(cell, inst, label).run_id


def play(job: tuple[str, dict, Instance, str]) -> Path:
    """One label on one instance of one cell. A finished run is not played again."""
    name, cell, inst, label = job
    work = work_of(name, cell, inst, label)
    if (work / "run.json").exists():
        return work
    file = Path(cell["policy"]) if label != "formula_est" else None
    return run_once(spec_of(cell, inst, label), inst.scenario(), work, file, cfg=benchmark.profile_cell(inst, name),
                    traffic=inst.traffic(noise_std()), clock_offset_s=CLOCK_OFFSET_S, channel=channel_of(name, inst.seed))


def jobs(per_family: int | None = None) -> list[tuple[str, dict, Instance, str]]:
    out = []
    for name, cell in cells().items():
        instances = dev_instances(cell["families"]) if per_family is None else dev_instances(cell["families"], per_family)
        out += [(name, cell, inst, label) for inst in instances for label in LABELS]
    return out


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
            play((name, cell, inst, label))
            seconds = time.monotonic() - started
            total = len(dev_instances(cell["families"]))
            print(f"{name} {label}: {seconds:.1f} s per run; {total} runs ~ {seconds * total / WORKERS / 60:.1f} min "
                  f"on {WORKERS} workers", flush=True)


# ---------------------------------------------------------------- the numbers


def fallbacks(events: list[dict]) -> list[dict]:
    return [e for e in events if e["event"] == "executor_fallback"]


def score(name: str, cell: dict) -> list[dict]:
    """One row per instance, intent and label; the verifier must answer the same in every label's run (checked)."""
    rows = []
    for inst in dev_instances(cell["families"]):
        flags = None
        scenario = inst.scenario()
        for label in LABELS:
            work = work_of(name, cell, inst, label)
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
                             "first_fallback_s": "" if not fired else fired[0]["ts_backend"] - CLOCK_OFFSET_S})
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
    bound (N + KPI delay + 1 s) / window: the seconds before the formula can act, as a fraction of the window."""
    by = {(r["instance"], r["intent"], r["label"]): r for r in rows if r["accepted"]}
    out = []
    for (inst, k, label), r in by.items():
        if label != "rl_mon":
            continue
        gap = by[(inst, k, "formula_est")]["intent_met"] - r["intent_met"]
        bound = (FALLBACK_AFTER_S + KPI_DELAY_TICKS + 1) / r["window_s"] if r["window_s"] > 0 else float("nan")
        out.append({"cell": r["cell"], "family": r["family"], "instance": inst, "intent": k, "deficit": gap,
                    "detection_bound": bound, "fallbacks_in_run": r["fallbacks_in_run"],
                    "first_fallback_s": r["first_fallback_s"]})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_all() -> dict:
    out, all_rows, all_deficits = {}, [], []
    for name, cell in cells().items():
        rows = score(name, cell)
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
    out["rule"] = {"fallback_after_s": FALLBACK_AFTER_S, "promise_margin": PROMISE_MARGIN,
                   "no_harm_margin": NO_HARM_MARGIN}
    out["rule"]["revision"] = REVISION
    write_csv(DATA / f"M11b-dev-runs-r{REVISION}.csv", all_rows)
    write_csv(DATA / f"M11b-deficits-r{REVISION}.csv", all_deficits)
    (DATA / f"M11b-dev-r{REVISION}.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--probe", action="store_true")
    group.add_argument("--dev", action="store_true")
    group.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.probe:
        probe()
    elif args.dev:
        play_all(jobs())
    else:
        print(json.dumps(score_all(), indent=1, default=float))


if __name__ == "__main__":
    main()
