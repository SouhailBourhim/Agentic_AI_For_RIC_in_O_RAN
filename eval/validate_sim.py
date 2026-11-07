"""M09a: replay M08c's and M08d's testbed scenarios on the calibrated simulator, and compare with the testbed.

  uv run python -m eval.validate_sim --dev            one seed, M08d's S1 and S3, the 4 controllers: for bugs only
  uv run python -m eval.validate_sim --final          240 runs (2 studies x 3 scenarios x 4 controllers x 10 seeds); needs Ollama
  uv run python -m eval.validate_sim --compare        the table and the verdict, from finished runs

The rule (docs/DECISIONS.md, M09a pre-registration) was written before any replay. --final refuses uncommitted
code. A finished run is never played again, so a stopped command can simply be started again. Raw runs go to
eval/results/m09a/ (git-ignored); the tables to docs/learn/data/M09a-sim-runs.csv and M09a-validation.csv.
"""

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from agent.llm import OllamaClient
from eval.harness import CONTROLLERS, USES_AGENT, RunSpec, run_once
from eval.metrics import load_run, measure_run, over_bound_after
from eval.plot_testbed_m08d import load_gaps
from eval.scenarios import Scenario
from eval.sim_profiles import KPI_DELAY_TICKS, noise_std, testbed_cell
from eval.testbed_control import commit, guard_final
from eval.testbed_scenarios import SCENARIOS as M08D_SCENARIOS
from eval.testbed_scenarios import SEGMENTS as M08D_SEGMENTS
from eval.testbed_scenarios_m08c import SCENARIOS as M08C_SCENARIOS
from eval.testbed_scenarios_m08c import SEGMENTS as M08C_SEGMENTS
from rl.runner import SERVED_SMOOTHING_ALPHA
from sim.traffic import TrafficPattern
from store.events import read_events
from verifier.objective import SLICES

RESULTS = Path("eval/results/m09a")
RUNS_TABLE = Path("docs/learn/data/M09a-sim-runs.csv")
VALIDATION = Path("docs/learn/data/M09a-validation.csv")
TESTBED_TABLES = {"m08c": Path("docs/learn/data/M08c-runs.csv"), "m08d": Path("docs/learn/data/M08d-runs.csv")}
TESTBED_RESULTS = {"m08c": Path("eval/results/m08c"), "m08d": Path("eval/results/m08d")}
STUDIES = ("m08c", "m08d")
SEEDS = range(9000, 9010)
DEV_SEED = 8999
PANELS = [("s1", 1), ("s2", 1), ("s3", 1), ("s3", 2)]
TIE = 0.05  # medians closer than this are not ordered
CELL_TOLERANCE = 0.15
NEED_PANELS, NEED_CELLS = 7, 26


def scenarios(study: str) -> dict[str, Scenario]:
    return M08C_SCENARIOS if study == "m08c" else M08D_SCENARIOS


def segments(study: str) -> dict:
    return M08C_SEGMENTS if study == "m08c" else M08D_SEGMENTS


def played_starts(study: str, scenario: str, controller: str) -> list[float]:
    """When each segment started on the testbed: M08d on the clock; M08c as played, the median over its repeats
    (each segment waited for the previous iperf3 to close, by an amount that depended on the controller)."""
    nominal = list(np.cumsum([0] + [seconds for seconds, _ in segments(study)[scenario][:-1]]).astype(float))
    if study == "m08d":
        return [float(x) for x in nominal]
    starts = []
    for work in sorted((TESTBED_RESULTS[study] / "testbed" / scenario / controller).glob("repeat?")):
        events = read_events(work / "events.jsonl")
        t0 = next(e["ts_wall"] for e in events if e["event"] == "scenario_start")
        starts.append([e["ts_wall"] - t0 for e in events if e["event"] == "segment_start"])
    medians = [float(np.median(column)) for column in zip(*starts, strict=True)]
    return [0.0, *medians[1:]]  # the first segment started within 10 ms of t0


def replay_traffic(study: str, scenario: str, controller: str) -> dict[str, TrafficPattern]:
    """Each slice's offered load as iperf3 played it: a segment's rate from its start for its length, then
    nothing until the next one starts (M08c's gap; on M08d's clock there is none)."""
    starts = played_starts(study, scenario, controller)
    steps: dict[str, list[tuple[float, float]]] = {s: [] for s in SLICES}
    for i, (start, (seconds, rates)) in enumerate(zip(starts, segments(study)[scenario], strict=True)):
        end = start + seconds
        for s in SLICES:
            steps[s].append((float(start), rates[s]))
            if i + 1 == len(starts) or starts[i + 1] > end:  # a gap before the next segment, or the end
                steps[s].append((float(end), 0.0))
    return {s: TrafficPattern("steps", steps=tuple(v), noise_std=noise_std()) for s, v in steps.items()}


def replay_scenario(study: str, scenario: str, controller: str) -> Scenario:
    """The testbed's scenario; S1's reaction clock starts when the spike was played, as in eval.testbed_control."""
    played = scenarios(study)[scenario]
    if scenario != "s1":
        return played
    spike = played_starts(study, scenario, controller)[1]
    return dataclasses.replace(played, intents=(dataclasses.replace(played.intents[0], reaction_from_s=spike),))


def spec_of(study: str, scenario: str, controller: str, seed: int, dev: bool = False) -> RunSpec:
    label = f"dev_{study}" if dev else study
    return RunSpec(label, scenario, controller, seed, alpha=SERVED_SMOOTHING_ALPHA, delay_ticks=KPI_DELAY_TICKS)


def play(spec: RunSpec, study: str, policy_file: Path, llm) -> Path:
    work = RESULTS / spec.run_id
    if (work / "run.json").exists():
        return work
    return run_once(spec, replay_scenario(study, spec.scenario, spec.controller), work, policy_file,
                    llm if spec.controller in USES_AGENT else None,
                    cfg=testbed_cell(study, spec.seed), traffic=replay_traffic(study, spec.scenario, spec.controller))


def served_policy() -> Path:
    cache = Path("var/policy_cache")
    return cache / json.loads((cache / "served.json").read_text())["file"]


def play_matrix(dev: bool) -> None:
    llm = OllamaClient()
    policy = served_policy()
    seeds = [DEV_SEED] if dev else list(SEEDS)
    studies = ("m08d",) if dev else STUDIES
    names = ("s1", "s3") if dev else ("s1", "s2", "s3")
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not dev:
        (RESULTS / "commit.txt").write_text(commit())
    specs = [(study, spec_of(study, n, c, seed, dev)) for study in studies for n in names for c in CONTROLLERS for seed in seeds]
    for i, (study, spec) in enumerate(specs, 1):
        work = play(spec, study, policy, llm)
        print(f"[{i}/{len(specs)}] {spec.run_id}: {json.loads((work / 'run.json').read_text())['wall_s']} s", flush=True)


def write_runs_table() -> list[dict]:
    rows = []
    for study in STUDIES:
        for path in sorted((RESULTS / study).glob("*/*/seed*/run.json")):
            spec = json.loads(path.read_text())
            work = path.parent
            played = replay_scenario(study, spec["scenario"], spec["controller"])
            measured = measure_run(work, played)
            kpis = load_run(work)[0]
            for intent, scenario_intent in zip(measured["intents"], played.intents, strict=True):
                after = over_bound_after(kpis, scenario_intent)
                rows.append({"study": study, "scenario": spec["scenario"], "controller": spec["controller"], "seed": spec["seed"],
                             **intent, "over_bound_after_s": "" if after is None else after, **measured["run"]})
    if rows:
        RUNS_TABLE.parent.mkdir(parents=True, exist_ok=True)
        with RUNS_TABLE.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def testbed_values(study: str) -> dict[tuple[str, int, str], list[float]]:
    """intent_met per (scenario, intent, controller): the runs each testbed repeat ended with. M08d's two S3 LLM only
    runs that lost their offered load take their filled values (the controller's own behaviour; DECISIONS)."""
    with TESTBED_TABLES[study].open() as f:
        rows = [r for r in csv.DictReader(f) if r["study"] == "testbed" and r.get("invalid_copy", "False") == "False"]
    filled = {}
    if study == "m08d":
        filled = {run.split("/")[-1]: value for run, _, _, value in load_gaps() if "s3/llm_only" in run}
    out: dict[tuple[str, int, str], list[float]] = {}
    for r in rows:
        value = float(r["intent_met"])
        if (r["scenario"], r["controller"], r["intent"]) == ("s3", "llm_only", "1") and f"repeat{r['seed']}" in filled:
            value = filled[f"repeat{r['seed']}"]
        out.setdefault((r["scenario"], int(r["intent"]), r["controller"]), []).append(value)
    return out


def panel_mismatches(testbed: dict[str, float], sim: dict[str, float]) -> list[str]:
    """Pairs ordered one way on the testbed and the other way in the simulator, each by more than TIE."""
    out = []
    for i, a in enumerate(CONTROLLERS):
        for b in CONTROLLERS[i + 1:]:
            t, s = testbed[a] - testbed[b], sim[a] - sim[b]
            if (t > TIE and s < -TIE) or (t < -TIE and s > TIE):
                out.append(f"{a}/{b}")
    return out


def compare(rows: list[dict]) -> dict:
    cells, panels = [], []
    for study in STUDIES:
        real = testbed_values(study)
        for scenario, intent in PANELS:
            tb = {c: float(np.median(real[(scenario, intent, c)])) for c in CONTROLLERS}
            sim_values = {c: [float(r["intent_met"]) for r in rows if r["study"] == study and r["scenario"] == scenario
                              and r["intent"] == intent and r["controller"] == c] for c in CONTROLLERS}
            if not all(sim_values.values()):
                continue
            sm = {c: float(np.median(v)) for c, v in sim_values.items()}
            mismatches = panel_mismatches(tb, sm)
            panels.append({"study": study, "scenario": scenario, "intent": intent, "order_matches": not mismatches,
                           "mismatched_pairs": " ".join(mismatches)})
            for c in CONTROLLERS:
                cells.append({"study": study, "scenario": scenario, "intent": intent, "controller": c, "testbed_median": round(tb[c], 3),
                              "sim_median": round(sm[c], 3), "sim_min": round(min(sim_values[c]), 3), "sim_max": round(max(sim_values[c]), 3),
                              "abs_diff": round(abs(sm[c] - tb[c]), 3), "within": abs(sm[c] - tb[c]) <= CELL_TOLERANCE, "n_sim": len(sim_values[c])})
    good_panels = sum(p["order_matches"] for p in panels)
    good_cells = sum(c["within"] for c in cells)
    not_static = [c for c in cells if c["controller"] != "static"]
    verdict = {"panels": len(panels), "panels_matching": good_panels, "cells": len(cells), "cells_within": good_cells,
               "cells_within_without_static": f"{sum(c['within'] for c in not_static)}/{len(not_static)}",
               "pass": len(panels) == 8 and good_panels >= NEED_PANELS and good_cells >= NEED_CELLS}
    VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    with VALIDATION.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    return {"verdict": verdict, "panels": panels, "cells": cells}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dev", action="store_true")
    group.add_argument("--final", action="store_true")
    group.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.final:
        guard_final(commit())
    if args.dev or args.final:
        play_matrix(dev=args.dev)
    if args.dev:
        return
    rows = write_runs_table()
    result = compare(rows)
    for p in result["panels"]:
        print(f"{p['study']} {p['scenario']} goal {p['intent']}: order {'matches' if p['order_matches'] else 'MISMATCH ' + p['mismatched_pairs']}")
    for c in result["cells"]:
        print(f"  {c['study']} {c['scenario']}/{c['intent']} {c['controller']:8s} testbed {c['testbed_median']:.2f}  sim {c['sim_median']:.2f} "
              f"[{c['sim_min']:.2f}-{c['sim_max']:.2f}]  {'ok' if c['within'] else 'OFF'}")
    print(json.dumps(result["verdict"], indent=1))


if __name__ == "__main__":
    main()
