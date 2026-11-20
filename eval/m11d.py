"""M11d: the shielded RL executor (rl/shield.py) on the dev instances of two cells, against the RL alone, the formula,
and with the M11b′ monitor. Rules in DECISIONS (2026-11-20), written before any shield run.

  uv run python -m eval.m11d --reproduce      the brief's step 1: m10 dev means 0.766 (RL) and 0.826 (formula_est)
  uv run python -m eval.m11d --dev            every dev run (a finished run is not played again)
  uv run python -m eval.m11d --score          S1-S3 (the brief's, m10) and D (serve it?), -> docs/learn/data/M11d-*

Labels: rl, shield, formula_est, rl_mon (M11b′ r6, served now), shield_mon (the shield with the same monitor).
Cells and policies as eval.m11b: m10 (M10c′'s 160 dev instances, its median seed at alpha 0.25), m09 (200, v7).
Raw runs: eval/results/m11d/ (git-ignored).
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval import benchmark, m10b
from eval.generator import Instance, dev_instances
from eval.harness import RunSpec, run_once
from eval.m11b import cells
from eval.metrics import in_window, load_run, measure_run
from eval.sim_profiles import KPI_DELAY_TICKS, channel_of, noise_std
from eval.validate_sim import CLOCK_OFFSET_S
from rl.monitor import SERVED_TRIGGER
from rl.runner import SERVED_SMOOTHING_ALPHA

RESULTS = Path("eval/results/m11d")
DATA = Path("docs/learn/data")
LABELS = ("rl", "shield", "formula_est", "rl_mon", "shield_mon")
WORKERS = 8
REPRODUCE = {"rl": 0.766, "formula_est": 0.826}  # M10c′'s dev means on m10 (docs/learn/data/M10c-dev.json)
# The frozen rules (DECISIONS 2026-11-20)
OVER_BOUND_POINTS = 0.05  # S1: shield's F3 URLLC over-bound share <= formula_est's + 5 points (m10)
NO_HARM = 0.02  # S2 (shield vs rl on F2, F5, m10) and D (shield_mon vs rl_mon, every family, both cells)
SOME_GAIN = 0.01  # D: shield_mon > rl_mon + 0.01 pooled on at least one cell


def spec_of(cell: dict, inst: Instance, label: str, study: str = "dev") -> RunSpec:
    if label == "formula_est":
        return RunSpec(study, inst.name, "scripted", inst.seed, alpha=SERVED_SMOOTHING_ALPHA,
                       delay_ticks=KPI_DELAY_TICKS, executor="formula_est")
    executor = "shielded_rl" if label.startswith("shield") else cell["executor"]
    monitor = SERVED_TRIGGER.name if label.endswith("_mon") else None
    return RunSpec(study, inst.name, "scripted", inst.seed, alpha=cell["alpha"], delay_ticks=KPI_DELAY_TICKS,
                   executor=executor, monitor=monitor)


def work_of(name: str, cell: dict, inst: Instance, label: str, study: str = "dev") -> Path:
    return RESULTS / study / name / spec_of(cell, inst, label, study).run_id


def play(job: tuple[str, dict, Instance, str, str]) -> Path:
    name, cell, inst, label, study = job
    work = work_of(name, cell, inst, label, study)
    if (work / "run.json").exists():
        return work
    file = None if label == "formula_est" else Path(cell["policy"])
    return run_once(spec_of(cell, inst, label, study), inst.scenario(), work, file,
                    cfg=benchmark.profile_cell(inst, name), traffic=inst.traffic(noise_std()),
                    clock_offset_s=CLOCK_OFFSET_S, channel=channel_of(name, inst.seed))


def instances_of(cell: dict, study: str) -> list[Instance]:
    if study != "dev":
        raise SystemExit("M11d's test split exists only once D has passed on dev (DECISIONS 2026-11-20)")
    return dev_instances(cell["families"])


def jobs(study: str = "dev", labels=LABELS, only: str | None = None) -> list:
    return [(name, cell, inst, label, study) for name, cell in cells().items() if only in (None, name)
            for inst in instances_of(cell, study) for label in labels]


def play_all(todo: list) -> None:
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for n, work in enumerate(pool.map(play, todo, chunksize=2), 1):
            if n % 200 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] {work}", flush=True)


# ---------------------------------------------------------------- the numbers


def over_bound_share(kpis: dict, intent) -> float | None:
    """Share of the intent's window seconds with URLLC over its latency bound; None if the intent has no bound."""
    if intent.max_latency_ms is None:
        return None
    start, end = intent.window
    window = in_window(kpis["urllc"]["t"], start, end)
    return float((kpis["urllc"]["lat"][window] > intent.max_latency_ms).mean())


def score(name: str, cell: dict, study: str = "dev", labels=LABELS) -> list[dict]:
    rows = []
    for inst in instances_of(cell, study):
        scenario, flags = inst.scenario(), None
        for label in labels:
            work = work_of(name, cell, inst, label, study)
            kpis, events = load_run(work)
            mine = m10b.accepted_flags(events)
            if flags is not None and mine != flags:
                raise RuntimeError(f"{inst.name}: the verifier answered differently in the {label} run")
            flags = mine
            for k, row in enumerate(measure_run(work, scenario)["intents"]):
                ob = over_bound_share(kpis, scenario.intents[k])
                rows.append({"cell": name, "family": inst.family, "variant": inst.variant, "instance": inst.name,
                             "intent": k, "label": label, "accepted": flags[k], "intent_met": row["intent_met"],
                             "urllc_over_bound": "" if ob is None else ob,
                             "fallbacks_in_run": sum(e["event"] == "executor_fallback" for e in events)})
    return rows


def means(rows: list[dict], families, labels=LABELS) -> dict:
    kept = [r for r in rows if r["accepted"]]
    out = {}
    for f in (*families, "all"):
        mine = [r for r in kept if f == "all" or r["family"] == f]
        out[f] = {label: float(np.mean([r["intent_met"] for r in mine if r["label"] == label])) for label in labels}
        for label in labels:
            ob = [r["urllc_over_bound"] for r in mine if r["label"] == label and r["urllc_over_bound"] != ""]
            out[f][f"{label}_over_bound"] = float(np.mean(ob)) if ob else None
        out[f]["intents"] = len(mine) // len(labels)
    return out


def rules(by_cell: dict[str, dict]) -> dict:
    """S1-S3 on m10 (the brief), D on both cells (whether to serve the shield)."""
    m = by_cell["m10"]
    s1 = m["f3"]["shield_over_bound"] <= m["f3"]["formula_est_over_bound"] + OVER_BOUND_POINTS
    s2 = all(m[f]["shield"] >= m[f]["rl"] - NO_HARM for f in ("f2", "f5"))
    s3 = m["all"]["shield"] - m["all"]["formula_est"] >= 0.0
    no_harm = {c: {f: v["shield_mon"] >= v["rl_mon"] - NO_HARM for f, v in fam.items() if f != "all"}
               for c, fam in by_cell.items()}
    gain = {c: fam["all"]["shield_mon"] > fam["all"]["rl_mon"] + SOME_GAIN for c, fam in by_cell.items()}
    d = all(all(v.values()) for v in no_harm.values()) and any(gain.values())
    return {"S1": s1, "S2": s2, "S3": s3, "D": d, "D_no_harm": no_harm, "D_gain": gain}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_all(study: str = "dev") -> dict:
    by_cell, all_rows, out = {}, [], {}
    for name, cell in cells().items():
        rows = score(name, cell, study)
        by_cell[name] = means(rows, cell["families"])
        fired = {label: sum(1 for r in rows if r["label"] == label and r["intent"] == 0 and r["fallbacks_in_run"] > 0)
                 for label in ("rl_mon", "shield_mon")}
        out[name] = {"policy": cell["policy"], "means": by_cell[name], "runs_with_fallback": fired,
                     "runs": len(instances_of(cell, study))}
        all_rows += rows
    out["rules"] = rules(by_cell)
    out["monitor_trigger"] = SERVED_TRIGGER.name
    write_csv(DATA / f"M11d-{study}-runs.csv", all_rows)
    (DATA / f"M11d-{study}.json").write_text(json.dumps(out, indent=1))
    return out


def reproduce() -> dict:
    """The brief's step 1, on m10 only: the RL alone and the formula alone give M10c′'s dev means."""
    play_all(jobs("dev", ("rl", "formula_est"), only="m10"))
    cell = cells()["m10"]
    m = means(score("m10", cell, "dev", ("rl", "formula_est")), cell["families"], ("rl", "formula_est"))["all"]
    got = {label: round(m[label], 3) for label in REPRODUCE}
    if got != REPRODUCE:
        raise SystemExit(f"not reproduced: {got} against {REPRODUCE}; stop and find out why")
    return got


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reproduce", action="store_true")
    group.add_argument("--dev", action="store_true")
    group.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.reproduce:
        print("reproduced:", reproduce())
    elif args.dev:
        play_all(jobs("dev"))
    else:
        print(json.dumps(score_all("dev"), indent=1, default=float))


if __name__ == "__main__":
    main()
