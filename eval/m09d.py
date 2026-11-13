"""M09d: the pre-registered simulator result on the test seeds, scored once (rules: docs/DECISIONS.md, 2026-11-11).

  uv run python -m eval.m09d --play            the 200 test instances x 6 controllers (~1 h 45, Ollama on the GPU)
  uv run python -m eval.m09d --repeats         2 more plays of the 8 M09e instances, LLM controllers (~15 min)
  uv run python -m eval.m09d --score --final   the hypotheses P1-P6 and the M09e predictions, ONCE

Policy v7 as served, M09a's cell per seed, M09b's frozen prompts and references. Nothing is tuned here: every
number this file decides on was written in DECISIONS and committed before the first test run (the guard below
refuses an uncommitted tree). `--play` prints only wall times, so nobody sees a result before `--score`.
Raw runs: eval/results/m09d/ (git-ignored). Tables: docs/learn/data/M09d-*.csv.
"""

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agent.llm import OllamaClient
from eval.benchmark import FEASIBLE, ceiling_check, play, table_rows, work_of
from eval.generator import PER_FAMILY, Instance, draw, test_instances
from eval.metrics import load_run, measure_run
from eval.spent import mark_spent, refuse_if_spent
from eval.testbed_control import commit, guard_final
from eval.validate_sim import CELL_TOLERANCE, served_policy

RESULTS = Path("eval/results/m09d")
STUDY = "m09d_test"
SPENT_NAME = "m09d/test"
DATA = Path("docs/learn/data")
RUNS, HYPOTHESES_CSV, P6_CSV, PREDICTIONS_CSV = (DATA / f"M09d-{n}.csv" for n in ("runs", "hypotheses", "p6", "predictions"))
RESULT = RESULTS / "result.json"

CONTROLLERS = ("static", "rl_only", "llm_only_v2", "proposed", "formula", "oracle")
AGENT_CONTROLLERS = ("llm_only_v2", "proposed", "formula")  # the ones whose decisions go through the verifier
RESAMPLES = 10_000
LEVELS = (0.95, 0.99)  # decide on 95 %; 99 % = Bonferroni over P1-P5, reported only
MARGIN = 0.05  # non-inferiority margin (P4, P5)
P6_PRECISION = 0.95

# M09e's sample: 2 instances from each of these families, drawn with a seed fixed before any test run.
M09E_FAMILIES = ("f1", "f2", "f3", "f5")
M09E_PER_FAMILY = 2
M09E_SEED = 20261111
M09E_CONTROLLERS = ("static", "llm_only_v2", "formula", "proposed")
EXTRA_PLAYS = 2
TIE = 0.05  # a predicted difference this small is predicted as a tie


@dataclass(frozen=True)
class Hypothesis:
    name: str
    family: str
    other: str  # LLM + RL is compared with this controller
    kind: str  # "superior": lower bound > 0; "noninferior": lower bound > -MARGIN; "faster": upper bound < 0
    measure: str  # "intent_met" or "reaction" (seconds, LLM + RL minus the other)


HYPOTHESES = (
    Hypothesis("P1", "f1", "static", "superior", "intent_met"),
    Hypothesis("P2", "f2", "rl_only", "superior", "intent_met"),
    Hypothesis("P3a", "f3", "llm_only_v2", "superior", "intent_met"),
    Hypothesis("P3b", "f3", "llm_only_v2", "faster", "reaction"),
    Hypothesis("P4", "f3", "formula", "noninferior", "intent_met"),
    Hypothesis("P5", "f5", "static", "noninferior", "intent_met"),
)


# ---------------------------------------------------------------- the guard


def guard(register: Path | None = None, name: str = SPENT_NAME) -> str:
    """Refuse unless the set is unspent and the code and the protocol are committed. Returns the commit.
    M09e reuses it with its own set name."""
    refuse_if_spent(name, register)
    revision = commit()
    guard_final(revision)  # code directories (eval/testbed_control.CODE_DIRS)
    if subprocess.run(["git", "status", "--porcelain", "--", "docs/DECISIONS.md"], capture_output=True, text=True,
                      check=False).stdout.strip():
        raise SystemExit("commit docs/DECISIONS.md first: the pre-registration must come before the runs in the history")
    return revision


# ---------------------------------------------------------------- the runs


def m09e_sample() -> list[tuple[str, int]]:
    """The 8 instances M09e plays on the testbed: (family, index). Drawn from a fixed seed, not from any result."""
    rng = np.random.default_rng(M09E_SEED)
    return [(f, int(i)) for f in M09E_FAMILIES for i in sorted(rng.choice(PER_FAMILY, M09E_PER_FAMILY, replace=False))]


def play_test(register: Path | None = None) -> None:
    revision = guard(register)
    llm, policy = OllamaClient(), served_policy()
    (RESULTS / STUDY).mkdir(parents=True, exist_ok=True)
    (RESULTS / STUDY / "policy.json").write_text(json.dumps({"policy": str(policy), "commit": revision}))
    jobs = [(inst, c) for inst in test_instances() for c in CONTROLLERS]
    for n, (inst, controller) in enumerate(jobs, 1):
        work = play(inst, controller, policy, llm, STUDY, RESULTS)  # skips a finished run: resumable
        wall = json.loads((work / "run.json").read_text()).get("wall_s", "")
        print(f"[{n}/{len(jobs)}] {inst.name} {controller}: {wall} s", flush=True)


def play_repeats(register: Path | None = None) -> None:
    """More plays of the M09e sample with the LLM controllers: what the model's nondeterminism does to the
    prediction (M09b's method). static is deterministic and has no LLM: its one play is enough."""
    guard(register)
    llm, policy = OllamaClient(), served_policy()
    jobs = [(draw(f, i, "test"), c, k) for f, i in m09e_sample() for c in AGENT_CONTROLLERS for k in range(1, EXTRA_PLAYS + 1)]
    for n, (inst, controller, k) in enumerate(jobs, 1):
        play(inst, controller, policy, llm, f"m09d_rep{k}", RESULTS)
        print(f"[{n}/{len(jobs)}] {inst.name} {controller} play {k + 1}", flush=True)


def missing_runs() -> list[str]:
    main = [work_of(inst, c, STUDY, RESULTS) for inst in test_instances() for c in CONTROLLERS]
    extra = [work_of(draw(f, i, "test"), c, f"m09d_rep{k}", RESULTS)
             for f, i in m09e_sample() for c in AGENT_CONTROLLERS for k in range(1, EXTRA_PLAYS + 1)]
    return [str(w) for w in main + extra if not (w / "run.json").exists()]


# ---------------------------------------------------------------- the statistics


def bootstrap_means(groups: list[np.ndarray], seed: int = 0, resamples: int = RESAMPLES) -> np.ndarray:
    """The mean over all values, recomputed on `resamples` resamples of the groups (one group = one instance).
    Instances are the unit drawn by the generator; the 2-3 intents of an F1b or F4a instance share its traffic
    and its cell, so resampling intents one by one would treat them as independent and narrow the interval."""
    sums = np.array([g.sum() for g in groups], dtype=float)
    counts = np.array([len(g) for g in groups], dtype=float)
    picks = np.random.default_rng(seed).integers(0, len(groups), size=(resamples, len(groups)))
    return sums[picks].sum(axis=1) / counts[picks].sum(axis=1)


def interval(draws: np.ndarray, level: float) -> tuple[float, float]:
    tail = 100 * (1 - level) / 2
    return float(np.percentile(draws, tail)), float(np.percentile(draws, 100 - tail))


def verdict(low: float, high: float, kind: str) -> bool:
    """The pass rules of the pre-registration, on one interval."""
    if kind == "superior":
        return low > 0
    if kind == "noninferior":
        return low > -MARGIN
    if kind == "faster":
        return high < 0
    raise ValueError(kind)


def reaction_or_window(row: dict, instance: Instance) -> float:
    """Reaction time in seconds; an intent never reached counts as the whole time it had (P3b's rule), so a
    controller that never gets there can't look faster by being left out."""
    if row["reaction_s"] != "":
        return float(row["reaction_s"])
    intent = instance.intents[int(row["intent"]) - 1]
    return intent.window[1] - intent.reaction_from_s


def paired_groups(rows: list[dict], h: Hypothesis, instances: dict[str, Instance]) -> list[np.ndarray]:
    """LLM + RL minus the other controller, intent by intent, grouped by instance."""
    def value(r: dict) -> float:
        return float(r["intent_met"]) if h.measure == "intent_met" else reaction_or_window(r, instances[r["instance"]])

    ours = {(r["instance"], r["intent"]): value(r) for r in rows if r["family"] == h.family and r["controller"] == "proposed"}
    theirs = {(r["instance"], r["intent"]): value(r) for r in rows if r["family"] == h.family and r["controller"] == h.other}
    groups: dict[str, list[float]] = {}
    for key in sorted(set(ours) & set(theirs)):
        groups.setdefault(key[0], []).append(ours[key] - theirs[key])
    return [np.array(v) for v in groups.values()]


def score_hypotheses(rows: list[dict], instances: dict[str, Instance]) -> list[dict]:
    out = []
    for h in HYPOTHESES:
        groups = paired_groups(rows, h, instances)
        draws = bootstrap_means(groups)
        mean = float(np.concatenate(groups).mean())
        row = {"hypothesis": h.name, "family": h.family, "other": h.other, "kind": h.kind, "measure": h.measure,
               "instances": len(groups), "intents": sum(len(g) for g in groups), "mean_diff": round(mean, 4)}
        for level in LEVELS:
            low, high = interval(draws, level)
            tag = f"{round(level * 100)}"
            row |= {f"low_{tag}": round(low, 4), f"high_{tag}": round(high, 4), f"pass_{tag}": verdict(low, high, h.kind)}
        out.append(row)
    return out


# ---------------------------------------------------------------- P6, the verifier


def feasible(oracle_met: float, static_met: float) -> bool:
    """Some LLM-free controller met it in at least 90 % of its window. The oracle alone is a reference, not a
    ceiling (it can let URLLC queue before a bounded intent arrives; DECISIONS 2026-11-09), and the equal split
    meeting it is just as good a proof that a split exists."""
    return max(oracle_met, static_met) >= FEASIBLE


def unverified_applies(events: list[dict], instance: Instance) -> dict[str, int]:
    """P6c, from the run's own log. Every applied objective must follow an accepted re-verification, and every
    applied split (LLM only's set_slice_share) an accepted verification of the same shares; each verification
    serves one apply. And no F4b request (above the policy's range by construction) may reach an applied objective."""
    unverified = over_range = 0
    objective_ok, shares_ok, last_args = False, None, {}
    over = instance.intents[0].targets if instance.variant == "b" and instance.family == "f4" else {}
    for e in events:
        data = e["data"]
        if e["event"] == "agent_step" and data.get("node") == "act" and data.get("tool") == "set_rl_objective":
            last_args = data.get("arguments", {})
        elif e["event"] == "reverified":
            objective_ok = bool(data.get("accepted"))
        elif e["event"] == "verified_shares":
            shares_ok = data.get("shares") if data.get("accepted") else None
        elif e["event"] == "applied":
            unverified += not objective_ok
            targets = last_args.get("min_thr_mbps") or {}
            over_range += any((targets.get(s) or 0) >= t for s, t in over.items())
            objective_ok = False
        elif e["event"] == "shares_applied":
            unverified += shares_ok is None or shares_ok != data.get("shares")
            shares_ok = None
    return {"unverified": unverified, "over_range": over_range}


def p6(rows: list[dict], instances: dict[str, Instance], study: str = STUDY, root: Path = RESULTS) -> dict:
    static = {(r["instance"], r["intent"]): float(r["intent_met"]) for r in rows if r["controller"] == "static"}
    mine = [r for r in rows if r["controller"] == "proposed"]
    out: dict = {"intents": len(mine), "accepted": sum(r["outcome"] == "applied" for r in mine)}
    for label, is_feasible in (("oracle_or_static", lambda r: feasible(float(r["oracle_met"]), static[(r["instance"], r["intent"])])),
                               ("oracle_only", lambda r: float(r["oracle_met"]) >= FEASIBLE)):
        by_instance: dict[str, list[tuple[bool, bool]]] = {}
        for r in mine:
            by_instance.setdefault(r["instance"], []).append((r["outcome"] == "applied", is_feasible(r)))
        accepted = [np.array([f for a, f in v if a], dtype=float) for v in by_instance.values()]
        accepted = [g for g in accepted if len(g)]
        feasible_ones = [np.array([a for a, f in v if f], dtype=float) for v in by_instance.values()]
        feasible_ones = [g for g in feasible_ones if len(g)]
        precision = float(np.concatenate(accepted).mean())
        low, high = interval(bootstrap_means(accepted), 0.95)
        out |= {f"{label}_feasible": int(sum(is_feasible(r) for r in mine)),
                f"{label}_precision": round(precision, 4), f"{label}_precision_low": round(low, 4),
                f"{label}_precision_high": round(high, 4),
                f"{label}_recall": round(float(np.concatenate(feasible_ones).mean()), 4)}
    for controller in AGENT_CONTROLLERS:
        counts = {"unverified": 0, "over_range": 0}
        for inst in instances.values():
            work = work_of(inst, controller, study, root)
            for k, v in unverified_applies(load_run(work)[1], inst).items():
                counts[k] += v
        out |= {f"{controller}_unverified_applies": counts["unverified"], f"{controller}_over_range_applied": counts["over_range"]}
    out["p6a_pass"] = out["oracle_or_static_precision"] >= P6_PRECISION
    out["p6c_pass"] = out["proposed_unverified_applies"] == 0 and out["proposed_over_range_applied"] == 0
    return out


# ---------------------------------------------------------------- the predictions for M09e


def instance_score(work: Path, instance: Instance) -> float:
    """One number per run: the mean intent met over the instance's intents (F1b has two)."""
    return float(np.mean([r["intent_met"] for r in measure_run(work, instance.scenario())["intents"]]))


def predictions() -> list[dict]:
    """Per M09e instance and controller: the simulator's score over its plays, an interval widened by M09a's cell
    tolerance (how far the calibrated simulator may sit from the testbed), the ranking, and the sign of LLM + RL
    minus each other controller, "tie" within TIE. LLM only v2's rows are weak: no simulator reproduces it (M09a)."""
    rows = []
    for f, i in m09e_sample():
        inst = draw(f, i, "test")
        scores = {}
        for c in M09E_CONTROLLERS:
            studies = [STUDY] + ([f"m09d_rep{k}" for k in range(1, EXTRA_PLAYS + 1)] if c in AGENT_CONTROLLERS else [])
            scores[c] = np.array([instance_score(work_of(inst, c, s, RESULTS), inst) for s in studies])
        means = {c: float(v.mean()) for c, v in scores.items()}
        top = max(means.values())
        order = sorted(means, key=lambda c: -means[c])
        for c in M09E_CONTROLLERS:
            diff = means["proposed"] - means[c]
            sign = "" if c == "proposed" else ("tie" if abs(diff) <= TIE else ("+" if diff > 0 else "-"))
            rows.append({"instance": inst.name, "family": f, "seed": inst.seed, "controller": c, "plays": len(scores[c]),
                         "mean": round(means[c], 3), "min": round(float(scores[c].min()), 3), "max": round(float(scores[c].max()), 3),
                         "low": round(max(0.0, float(scores[c].min()) - CELL_TOLERANCE), 3),
                         "high": round(min(1.0, float(scores[c].max()) + CELL_TOLERANCE), 3),
                         "rank": order.index(c) + 1, "best": top - means[c] <= TIE,
                         "proposed_minus_this": "" if c == "proposed" else sign, "weak": c == "llm_only_v2"})
    return rows


# ---------------------------------------------------------------- scoring, once


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def score(final: bool, register: Path | None = None) -> dict:
    if not final:
        raise SystemExit("the test set is scored once: pass --final when you mean it")
    revision = guard(register)
    if missing := missing_runs():
        raise SystemExit(f"{len(missing)} runs missing (first: {missing[0]}); finish --play and --repeats first")
    instances = {inst.name: inst for inst in test_instances()}
    rows = table_rows(list(instances.values()), CONTROLLERS, STUDY, RESULTS)
    static = {(r["instance"], r["intent"]): float(r["intent_met"]) for r in rows if r["controller"] == "static"}
    for r in rows:
        r["static_met"] = static[(r["instance"], r["intent"])]
        r["feasible_oracle_or_static"] = feasible(float(r["oracle_met"]), r["static_met"])
    write_csv(RUNS, rows)
    hypotheses = score_hypotheses(rows, instances)
    write_csv(HYPOTHESES_CSV, hypotheses)
    verifier = p6(rows, instances)
    write_csv(P6_CSV, [verifier])
    predicted = predictions()
    write_csv(PREDICTIONS_CSV, predicted)
    play_commit = json.loads((RESULTS / STUDY / "policy.json").read_text()).get("commit", "")
    parse = {c: f"{sum(r['parse_ok'] is True for r in rows if r['controller'] == c)}/{sum(r['controller'] == c for r in rows)}"
             for c in AGENT_CONTROLLERS}
    result = {"commit_play": play_commit, "commit_score": revision, "hypotheses": hypotheses, "p6": verifier,
              "ceiling_check": ceiling_check(rows), "parse_ok": parse, "m09e_sample": m09e_sample()}
    RESULT.write_text(json.dumps(result, indent=1))
    p = {h["hypothesis"]: h["pass_95"] for h in hypotheses}
    mark_spent(SPENT_NAME, RESULT, f"M09d, v7: P1 {p['P1']}, P2 {p['P2']}, P3 {p['P3a'] and p['P3b']}, P4 {p['P4']}, "
               f"P5 {p['P5']}, P6a {verifier['p6a_pass']}, P6c {verifier['p6c_pass']}", register)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--play", action="store_true")
    group.add_argument("--repeats", action="store_true")
    group.add_argument("--score", action="store_true")
    parser.add_argument("--final", action="store_true", help="with --score: score the test set, once")
    args = parser.parse_args()
    if args.play:
        play_test()
        return
    if args.repeats:
        play_repeats()
        return
    result = score(args.final)
    for h in result["hypotheses"]:
        print(f"{h['hypothesis']:4s} {h['family']} LLM+RL - {h['other']:12s} ({h['measure']}) n={h['intents']:3d}  "
              f"{h['mean_diff']:+.3f} [{h['low_95']:+.3f}, {h['high_95']:+.3f}]  pass {h['pass_95']}  (99 %: {h['pass_99']})")
    v = result["p6"]
    print(f"P6a precision {v['oracle_or_static_precision']:.3f} [{v['oracle_or_static_precision_low']:.3f}, "
          f"{v['oracle_or_static_precision_high']:.3f}] pass {v['p6a_pass']}; P6b recall {v['oracle_or_static_recall']:.3f}; "
          f"P6c pass {v['p6c_pass']}; oracle-only precision {v['oracle_only_precision']:.3f}")
    print("parse_ok:", result["parse_ok"], " ceiling check:", {k: round(x, 3) for k, x in result["ceiling_check"].items()})
    print(f"marked spent: {SPENT_NAME}. Commit docs/learn/data/spent_sets.json and the M09d-*.csv tables.")


if __name__ == "__main__":
    main()
