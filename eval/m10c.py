"""M10c': the RL policy trained on the m10 cell against formula_est, on F1, F2, F3 and F5, pre-registered.

  uv run python -m eval.m10c --choose            the median seed of models/m10c (rl.sweep's rule) -> M10c-policy.json
  uv run python -m eval.m10c --dev               every executor on the 160 dev instances: the expectations, not a claim
  uv run python -m eval.m10c --play              the 160 test instances, once (guard: committed code and DECISIONS)
  uv run python -m eval.m10c --score --final     H1 and the reported numbers, once; marks m10c/test spent

Same mode as M10b: no LLM; each intent's stated facts go through the real verifier and apply step with the default
weights, and only the executor differs. The rules are in docs/DECISIONS.md (pre-registration of 2026-11-14), written
before the policy was trained. F4 is left out after M10b's gate, as that line says.
Raw runs: eval/results/m10c/ (git-ignored). Tables: docs/learn/data/M10c-*.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


from eval import benchmark, m10b
from eval.generator import M10C_FAMILIES, Instance, dev_instances, m10c_test_instances
from eval.harness import RunSpec, run_once
from eval.m09d import guard
from eval.sim_profiles import KPI_DELAY_TICKS, channel_of, noise_std
from eval.spent import mark_spent
from eval.validate_sim import CLOCK_OFFSET_S, served_policy
from rl import m09
from rl.runner import SERVED_SMOOTHING_ALPHA
from rl.sweep import checkpoint_reward, median_seeds

PROFILE = "m10"
RESULTS = Path("eval/results/m10c")
DATA = Path("docs/learn/data")
CHOICE = DATA / "M10c-policy.json"
POLICY_DIR = Path("models/m10c")
SPENT_NAME = "m10c/test"
RL, COMPARATOR = "m10_rl", "formula_est"
EXECUTORS = ("static", "v7", "rev4", "formula_est", "mpc_est", RL, "formula_told", "oracle")
EQUIVALENCE = 0.05
WORKERS = 8


# ---------------------------------------------------------------- the policy


def choose() -> dict:
    """The median seed by validation reward (rl.sweep's rule). If it kept no checkpoint, its final model, as the
    pre-registration says, so the test runs whatever training gave."""
    runs = sorted(r for r in POLICY_DIR.glob("td3_m10c_seed*.zip") if not r.stem.endswith("_best"))
    if len(runs) != 5:
        raise SystemExit(f"expected 5 trained seeds in {POLICY_DIR}, found {len(runs)}")
    best = median_seeds(POLICY_DIR, m09.M10_RECIPES)["m10c"]
    ranked = sorted(runs, key=lambda r: -1e9 if checkpoint_reward(r) is None else checkpoint_reward(r))
    median = ranked[(len(ranked) - 1) // 2]
    choice = {"median_seed": median.stem, "file": str(best if best is not None else median),
              "fallback_to_final_model": best is None,
              "validation_reward": {r.stem: checkpoint_reward(r) for r in runs}}
    CHOICE.write_text(json.dumps(choice, indent=1))
    return choice


def policy_file() -> Path:
    if not CHOICE.exists():
        raise SystemExit("run --choose first: the policy tested is the median seed")
    return Path(json.loads(CHOICE.read_text())["file"])


# ---------------------------------------------------------------- the runs


def play(job: tuple[Instance, str, str, str]) -> Path:
    """One executor on one instance of the m10 cell, as eval.m10b.play; m10_rl at its own alpha."""
    inst, executor, study, policy = job
    work = m10b.work_of(inst, executor, study, RESULTS)
    if (work / "run.json").exists():
        return work
    alpha = m09.M10C_ALPHA if executor == RL else SERVED_SMOOTHING_ALPHA  # the alpha each policy is served at
    spec = RunSpec(study, inst.name, "scripted", inst.seed, alpha=alpha,
                   delay_ticks=0 if executor == "oracle" else KPI_DELAY_TICKS, executor=executor)
    file = Path(policy) if executor == RL else served_policy() if executor == "v7" else None
    return run_once(spec, inst.scenario(), work, file, cfg=benchmark.profile_cell(inst, PROFILE),
                    traffic=inst.traffic(noise_std()), clock_offset_s=CLOCK_OFFSET_S, channel=channel_of(PROFILE, inst.seed))


def play_all(instances: list[Instance], study: str) -> None:
    policy = str(policy_file())
    jobs = [(i, e, study, policy) for i in instances for e in EXECUTORS]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for n, work in enumerate(pool.map(play, jobs, chunksize=4), 1):
            if n % 100 == 0 or n == len(jobs):
                print(f"[{n}/{len(jobs)}] {work}", flush=True)


def m10c_dev_instances() -> list[Instance]:
    return dev_instances(M10C_FAMILIES)


# ---------------------------------------------------------------- the numbers


def reading(low: float, high: float, margin: float = EQUIVALENCE) -> str:
    """The pre-registered readings of H1's 95 % interval, in their order."""
    if low > 0:
        return "RL superior"
    if high < 0:
        return "formula better"
    if low > -margin and high < margin:
        return "equivalent"
    return "inconclusive"


def summary(rows: list[dict]) -> dict:
    """H1 with its reading, then everything reported only."""
    mean, low, high = m10b.paired_interval(rows, RL, COMPARATOR)
    _, low99, high99 = m10b.paired_interval(rows, RL, COMPARATOR, level=0.99)
    families = {}
    for f in M10C_FAMILIES:
        mine = [r for r in rows if r["family"] == f]
        families[f] = m10b.paired_interval(mine, RL, COMPARATOR)
    means = {e: m10b.family_means(rows, e) for e in EXECUTORS}
    told, oracle, rl = (means[e]["all"] for e in ("formula_told", "oracle", RL))
    return {"h1": {"mean": mean, "ci95": [low, high], "ci99": [low99, high99], "reading": reading(low, high)},
            "h1_by_family": {f: {"mean": m, "ci95": [lo, hi]} for f, (m, lo, hi) in families.items()},
            "means": means, "foresight_closed": (rl - told) / (oracle - told) if oracle > told else float("nan"),
            "intents": len({(r["instance"], r["intent"]) for r in rows if r["accepted"]})}


def dev() -> dict:
    instances = m10c_dev_instances()
    play_all(instances, "dev")
    rows = m10b.score(instances, EXECUTORS, "dev", RESULTS)
    out = {**summary(rows), "policy": json.loads(CHOICE.read_text())["file"]}
    m10b.write_rows(DATA / "M10c-dev-runs.csv", rows)
    (DATA / "M10c-dev.json").write_text(json.dumps(out, indent=1))
    return out


def missing_runs() -> list[str]:
    return [str(m10b.work_of(i, e, "test", RESULTS)) for i in m10c_test_instances() for e in EXECUTORS
            if not (m10b.work_of(i, e, "test", RESULTS) / "run.json").exists()]


def play_test() -> None:
    revision = guard(name=SPENT_NAME)
    (RESULTS / "test").mkdir(parents=True, exist_ok=True)
    (RESULTS / "test" / "policy.json").write_text(json.dumps({"policy": str(policy_file()), "commit": revision}))
    play_all(m10c_test_instances(), "test")


def score_test(final: bool) -> dict:
    if not final:
        raise SystemExit("the test is scored once: add --final")
    revision = guard(name=SPENT_NAME)
    if missing := missing_runs():
        raise SystemExit(f"{len(missing)} runs missing, e.g. {missing[0]}: run --play first")
    rows = m10b.score(m10c_test_instances(), EXECUTORS, "test", RESULTS)
    out = {**summary(rows), "policy": json.loads((RESULTS / "test" / "policy.json").read_text()), "commit": revision}
    m10b.write_rows(DATA / "M10c-runs.csv", rows)
    result = DATA / "M10c-result.json"
    result.write_text(json.dumps(out, indent=1))
    mark_spent(SPENT_NAME, result, note=f"H1: {out['h1']['reading']}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--choose", action="store_true")
    group.add_argument("--dev", action="store_true")
    group.add_argument("--play", action="store_true")
    group.add_argument("--score", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.choose:
        out = choose()
    elif args.dev:
        out = dev()
    elif args.play:
        play_test()
        return
    else:
        out = score_test(args.final)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
