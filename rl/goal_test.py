"""Goal-conditioning test v2 (M02b): does telling the policy the objective help it reach it?

Run with:
  uv run python -m rl.goal_test --check                     validity checks, no trained model needed
  uv run python -m rl.goal_test --split validation          development: per model and per recipe
  uv run python -m rl.goal_test --split test --final        the held-out test, run ONCE at the end
  ... --min-share 0.05                                      same test with the per-slice PRB floor
                                                            (M04a round; results in runs/m04a_floor/)
  uv run python -m rl.goal_test --split m09_validation --min-share 0.05          M09c, calibrated cell
  uv run python -m rl.goal_test --split m09_test --final --min-share 0.05        M09c held-out, ONCE

Why a new test. M02's test asked for a URLLC share gap >= 0.10 between "protect URLLC" and
"maximise eMBB" on one scenario. A reward-optimal controller scores anywhere in 0.034-0.120
on it, depending only on how it breaks ties (URLLC and BE weigh the same under "maximise
eMBB"), so the test could fail a perfect policy and pass a sloppy one. It is still reported.

The idea. For a scenario X and two objectives A != B:
    policy advantage = R_A(policy told A) - R_A(policy told B)
    oracle advantage = R_A(oracle for A)  - R_A(oracle for B)
R_A(... told B) = run the controller with B in its observation, score the episode with A's
reward (mean reward per step, reward v2). One rollout per (scenario, objective told) is
scored under every A by recomputing the reward from the logged KPIs.
A case is *material* when the oracle advantage is >= MATERIAL_THRESHOLD: the two objectives
really call for different actions there. Cases where they don't (ties) are left out.

Pass criteria, per trained seed, every seed must pass (fixed before any M02b training):
  T1a  policy advantage > 0 in >= 90 % of material cases
  T1b  mean over material cases of clip(policy advantage / oracle advantage, -1, 1) >= 0.5
  T2   mean reward on random episodes above the equal split's (test: env seeds 1000-1099,
       the M02 episodes; validation: 2000-2049)
A controller that ignores the objective (equal split, proportional to load, a policy whose
objective input is frozen) has advantage 0 in every case, so it fails by construction.

Two disjoint case sets: VALIDATION (used to develop: choose checkpoints and the recipe) and
TEST (held out; used once on the confirmation seeds).

M09c adds two splits on the calibrated cell, m09_validation and m09_test (rl/m09.py's setting: true
capacity drawn per episode and told 79 Mb/s, KPIs 2 ticks late, load history, filter at the served
alpha 0.10, 120-step episodes with random and on/off loads). The old sets are spent; these use fresh
seeds (7000s, 8000s) and the same criteria, scored with reward v2 at the told capacity. m09_test runs
once, with --final, on the confirmation seeds.

With min_share (M04a), every rollout (policy, oracle, equal split, reward episodes) runs in an env
that applies the per-slice PRB floor, as the deployed runner does. Everything else (scenarios,
objectives, seeds, thresholds) is unchanged. The M04a round is the SECOND use of the test set.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval.spent import mark_spent, refuse_if_spent
from mlops import tracking
from rl.baselines import equal_split, proportional_to_load
from rl.evaluate import OBJECTIVES, SPIKE_STEPS, episode_metrics
from rl import m09
from rl.oracle import GreedyOracle
from rl.policy import SMOOTHING_ALPHA, SlicePolicy
from sim.cell import CellConfig
from sim.env import OBJECTIVE_SLICE, SlicingEnv, objective_arrays, objective_to_vector, sample_objective
from sim.reward import compute_reward_batch
from sim.scenarios import urllc_spike
from sim.traffic import TrafficPattern, sample_traffic
from verifier.objective import SLICES, RLObjective, SliceValues

MATERIAL_THRESHOLD = 0.02  # reward per step
T1A_MIN_SHARE = 0.90
T1B_MIN_CAPTURED = 0.50
EPISODE_STEPS = SPIKE_STEPS  # 40 s per scenario
TEST_REWARD_SEEDS = range(1000, 1100)  # the M02 random episodes
VALIDATION_REWARD_SEEDS = range(2000, 2050)
RESULT_DIR = Path("runs/m02b")
FLOOR_RESULT_DIR = Path("runs/m04a_floor")  # the M04a retraining with the per-slice floor
M09_RESULT_DIR = Path("runs/m09c")  # M09c: the calibrated cell's splits
M09_SCENARIOS, M09_OBJECTIVES = 8, 6
M09_SEEDS = {"m09_validation": 7000, "m09_test": 8000}  # scenarios +i, objectives +100+i, reward episodes +200..

CFG = CellConfig()
C = CFG.capacity_mbps

Controller = Callable[[np.ndarray], np.ndarray]


def _steady(fraction: float) -> TrafficPattern:
    return TrafficPattern("steady", fraction * C, fraction * C)


def _spike(base: float, peak: float, start: float, end: float) -> TrafficPattern:
    return TrafficPattern("spike", base * C, peak * C, start, end)


def _ramp(base: float, peak: float, start: float, end: float) -> TrafficPattern:
    return TrafficPattern("ramp", base * C, peak * C, start, end)


def held_out_scenarios() -> dict[str, dict[str, TrafficPattern]]:
    """Held-out scenarios: the M02 URLLC spike unchanged, plus three other situations."""
    return {
        "URLLC spike": urllc_spike()[0],
        "eMBB surge": {"embb": _spike(0.40, 0.90, 10, 20), "urllc": _steady(0.10), "be": _steady(0.30)},
        "BE ramp": {"embb": _steady(0.45), "urllc": _steady(0.10), "be": _ramp(0.20, 0.70, 5, 30)},
        "steady overload": {"embb": _steady(0.55), "urllc": _steady(0.15), "be": _steady(0.40)},
    }


def held_out_objectives() -> dict[str, RLObjective]:
    """Held-out objectives: the two M02 ones unchanged, plus one favouring eMBB, one BE."""
    return {
        **OBJECTIVES,
        "guarantee eMBB": RLObjective(weights=SliceValues(embb=0.4, urllc=0.2, be=0.4),
                                      min_thr_mbps=SliceValues(embb=0.50 * C)),
        "protect BE": RLObjective(weights=SliceValues(embb=0.2, urllc=0.2, be=0.6),
                                  min_thr_mbps=SliceValues(be=0.30 * C)),
    }


def validation_scenarios() -> dict[str, dict[str, TrafficPattern]]:
    """Same families as the test with other levels and timings, plus 4 random patterns."""
    scenarios = {
        "val URLLC spike": {"embb": _steady(0.45), "urllc": _spike(0.08, 0.30, 15, 25), "be": _steady(0.30)},
        "val eMBB surge": {"embb": _spike(0.35, 0.80, 8, 22), "urllc": _steady(0.12), "be": _steady(0.25)},
        "val BE ramp": {"embb": _steady(0.40), "urllc": _steady(0.08), "be": _ramp(0.15, 0.60, 10, 35)},
        "val steady overload": {"embb": _steady(0.50), "urllc": _steady(0.12), "be": _steady(0.45)},
    }
    for i in range(4):
        scenarios[f"val random {i}"] = sample_traffic(np.random.default_rng(3000 + i), EPISODE_STEPS, C)
    return scenarios


def validation_objectives() -> dict[str, RLObjective]:
    """Six random objectives (same distribution as training), fixed by their seeds."""
    return {f"val objective {i}": sample_objective(np.random.default_rng(4000 + i), CFG) for i in range(6)}


@dataclass(frozen=True)
class CaseSet:
    name: str
    scenarios: dict[str, dict[str, TrafficPattern]]
    objectives: dict[str, RLObjective]
    reward_seeds: range


@dataclass(frozen=True)
class Setting:
    """The cell and actuator rules a case set is played in: M02b's (the defaults) or M09c's."""

    env: dict = field(default_factory=dict)  # SlicingEnv keyword arguments
    steps: int = EPISODE_STEPS
    alpha: float = SMOOTHING_ALPHA  # the filter of the policy and the oracle
    capacity_mbps: float = C  # what rewards are normalised by: the capacity the policy is told


M09_CELL = m09.cell("m09")
M09_SETTING = Setting({**m09.env_options("m09", "on"), "kpi_delay_range": m09.TEST_KPI_DELAY},
                      m09.EPISODE_STEPS, m09.FILTER_ALPHA, M09_CELL.told_capacity_mbps)


def m09_case_set(split: str) -> CaseSet:
    """Random scenarios (a third with on/off loads) and objectives on the told capacity, fixed by fresh seeds."""
    base, told = M09_SEEDS[split], M09_CELL.told_capacity_mbps
    scenarios = {f"{split} scenario {i}": sample_traffic(np.random.default_rng(base + i), m09.EPISODE_STEPS, told,
                                                         (1.5, 5.0), toggle_prob=0.34, noise_std=m09.noise_std())
                 for i in range(M09_SCENARIOS)}
    objectives = {f"{split} objective {i}": sample_objective(np.random.default_rng(base + 100 + i), M09_CELL)
                  for i in range(M09_OBJECTIVES)}
    return CaseSet(split, scenarios, objectives, range(base + 200, base + (250 if split == "m09_validation" else 300)))


def case_set(split: str) -> CaseSet:
    if split == "test":
        return CaseSet("test", held_out_scenarios(), held_out_objectives(), TEST_REWARD_SEEDS)
    if split == "validation":
        return CaseSet("validation", validation_scenarios(), validation_objectives(), VALIDATION_REWARD_SEEDS)
    if split in M09_SEEDS:
        return m09_case_set(split)
    raise ValueError(f"unknown split {split!r}")


def setting_of(split: str) -> Setting:
    return M09_SETTING if split in M09_SEEDS else Setting()


# --- Rollouts and scoring ------------------------------------------------------------------


def rollout(act: Callable[[SlicingEnv, np.ndarray], np.ndarray], traffic: dict[str, TrafficPattern],
            told: RLObjective, reset: Callable[[], None] | None = None,
            min_share: float | None = None, setting: Setting | None = None) -> dict[str, np.ndarray]:
    """Run one episode with `told` as the objective in the observation; log the KPIs."""
    setting = setting or Setting()
    env = SlicingEnv(episode_steps=setting.steps, min_share=min_share, **setting.env)
    obs, _ = env.reset(seed=0, options={"objective": told, "traffic": traffic})
    if reset is not None:
        reset()
    log: dict[str, list] = {"thr": [], "load": [], "lat_urllc": [], "share_urllc": []}
    for _ in range(setting.steps):
        obs, _, _, _, info = env.step(np.asarray(act(env, obs), dtype=np.float32))
        log["thr"].append([info["throughput_mbps"][s] for s in SLICES])
        log["load"].append([info["load_mbps"][s] for s in SLICES])
        log["lat_urllc"].append(info["latency_ms"]["urllc"])
        log["share_urllc"].append(info["shares"]["urllc"])
    return {k: np.array(v) for k, v in log.items()}


def score(log: dict[str, np.ndarray], objective: RLObjective, capacity_mbps: float = C) -> float:
    """Mean reward per step of a logged episode under `objective` (reward v2)."""
    w, t, b = objective_arrays(objective)
    n = len(log["lat_urllc"])
    rewards = compute_reward_batch(np.repeat(w, n, axis=0), np.repeat(t, n, axis=0), np.repeat(b, n),
                                   log["thr"], log["load"], log["lat_urllc"], capacity_mbps)
    return float(rewards.mean())


def controller_rollouts(controller: Controller, cases: CaseSet, min_share: float | None = None,
                        setting: Setting | None = None) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """One rollout per (scenario, objective told) for an obs -> action controller."""
    reset = getattr(controller, "reset", None)
    return {(sc, told): rollout(lambda _env, obs: controller(obs), traffic, obj, reset, min_share, setting)
            for sc, traffic in cases.scenarios.items() for told, obj in cases.objectives.items()}


def oracle_rollouts(cases: CaseSet, min_share: float | None = None,
                    setting: Setting | None = None) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """One rollout per (scenario, objective) of the oracle optimising that objective."""
    setting = setting or Setting()
    out = {}
    for sc, traffic in cases.scenarios.items():
        for told, obj in cases.objectives.items():
            oracle = GreedyOracle(obj, alpha=setting.alpha, min_share=min_share)
            out[(sc, told)] = rollout(lambda env, _obs: oracle.act(env), traffic, obj, oracle.reset, min_share, setting)  # noqa: B023 (called within this iteration)
    return out


def advantages(rollouts: dict, cases: CaseSet, capacity_mbps: float = C) -> dict[tuple[str, str, str], float]:
    """R_A(told A) - R_A(told B) for every scenario and ordered pair (A, B), A != B."""
    out = {}
    for sc in cases.scenarios:
        for a, obj_a in cases.objectives.items():
            own = score(rollouts[(sc, a)], obj_a, capacity_mbps)
            for b in cases.objectives:
                if b != a:
                    out[(sc, a, b)] = own - score(rollouts[(sc, b)], obj_a, capacity_mbps)
    return out


def goal_metrics(policy_adv: dict, oracle_adv: dict) -> dict[str, float]:
    """T1a and T1b over the material cases."""
    material = [k for k, v in oracle_adv.items() if v >= MATERIAL_THRESHOLD]
    if not material:
        raise ValueError("no material case: the case set cannot test goal conditioning")
    positive = np.mean([policy_adv[k] > 0 for k in material])
    captured = np.mean([np.clip(policy_adv[k] / oracle_adv[k], -1.0, 1.0) for k in material])
    return {"material_cases": len(material), "t1a_share_positive": float(positive),
            "t1b_captured": float(captured)}


def mean_reward(controller: Controller, seeds: range, min_share: float | None = None,
                setting: Setting | None = None) -> float:
    setting = setting or Setting()
    steps = {} if setting.env == {} else {"steps": setting.steps, "env_options": setting.env}  # M02b: as before
    return float(np.mean([episode_metrics(controller, seed, min_share=min_share, **steps)["reward"] for seed in seeds]))


class GoalTest:
    """Holds a case set and the oracle advantages (computed once), then scores controllers."""

    def __init__(self, split: str, min_share: float | None = None) -> None:
        self.cases = case_set(split)
        self.setting = setting_of(split)
        self.alpha = self.setting.alpha  # the filter a policy is scored with (rl.train's checkpoint uses it)
        self.min_share = min_share  # None: no floor (M02b); 0.05: the M04a actuator rules
        self.oracle_adv = advantages(oracle_rollouts(self.cases, min_share, self.setting), self.cases,
                                     self.setting.capacity_mbps)
        self.equal_reward = mean_reward(equal_split, self.cases.reward_seeds, min_share, self.setting)

    def evaluate(self, controller: Controller, with_reward: bool = True) -> dict[str, float | bool]:
        policy_adv = advantages(controller_rollouts(controller, self.cases, self.min_share, self.setting), self.cases,
                                self.setting.capacity_mbps)
        m: dict[str, float | bool] = goal_metrics(policy_adv, self.oracle_adv)
        m["t1a_pass"] = m["t1a_share_positive"] >= T1A_MIN_SHARE
        m["t1b_pass"] = m["t1b_captured"] >= T1B_MIN_CAPTURED
        if with_reward:
            m["reward"] = mean_reward(controller, self.cases.reward_seeds, self.min_share, self.setting)
            m["equal_split_reward"] = self.equal_reward
            m["t2_pass"] = m["reward"] > self.equal_reward
            m["pass"] = bool(m["t1a_pass"] and m["t1b_pass"] and m["t2_pass"])
        return m


class FrozenObjective:
    """A policy whose objective input is always the same: blind to the real objective."""

    def __init__(self, policy: SlicePolicy, objective: RLObjective) -> None:
        self.policy = policy
        self.vector = objective_to_vector(objective, CFG).astype(np.float32)

    def reset(self) -> None:
        self.policy.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs.copy()
        x[OBJECTIVE_SLICE] = self.vector
        return self.policy(x)


def original_criterion(policy: SlicePolicy, min_share: float | None = None) -> float:
    """The M02 measure, still reported: URLLC share gap on the spike, protect - maximise eMBB."""
    traffic = urllc_spike()[0]
    share = {name: rollout(lambda _env, obs: policy(obs), traffic, obj, policy.reset, min_share)["share_urllc"].mean()
             for name, obj in OBJECTIVES.items()}
    return float(share["protect URLLC"] - share["maximise eMBB"])


# --- Command line ---------------------------------------------------------------------------


def recipe_of(name: str) -> str:
    return name.split("_seed")[0]


def runs_without_best(models: list[Path]) -> list[str]:
    """Trained runs next to the given `_best` checkpoints that have none: no validation ever beat the equal split
    (rl.train.ValidationCheckpoint saves one only then). On a held-out split each is a failing seed, so "every seed
    must pass" is judged over all the seeds trained, not only the ones that left a checkpoint (M09c seed 40)."""
    folders = {path.parent for path in models}
    return sorted(final.stem + "_best" for folder in folders for final in folder.glob("*_seed*.zip")
                  if not final.stem.endswith("_best") and not final.with_name(final.stem + "_best.zip").exists())


def check(models: list[Path], min_share: float | None = None) -> None:
    """Validity of both case sets: enough material cases, blind controllers fail. Uses no trained
    policy unless models are given (then only with its objective input frozen)."""
    for split in ("validation", "test"):
        test = GoalTest(split, min_share)
        print(f"{split}: {len(test.cases.scenarios)} scenarios x {len(test.cases.objectives)} objectives, "
              f"{len(test.oracle_adv)} ordered cases")
        blind: dict[str, Controller] = {"equal split": equal_split, "proportional to load": proportional_to_load}
        if models:
            blind[f"{models[0].stem}, objective frozen"] = FrozenObjective(SlicePolicy.load(models[0]),
                                                                          OBJECTIVES["maximise eMBB"])
        for name, ctl in blind.items():
            m = test.evaluate(ctl, with_reward=False)
            print(f"  {name:<36} material {m['material_cases']:3d}  T1a {m['t1a_share_positive']:.2f}  "
                  f"T1b {m['t1b_captured']:.2f}  -> {'PASS' if m['t1a_pass'] and m['t1b_pass'] else 'fail'}")


FINAL_SPLITS = ("test", "m09_test")  # held out: run once, with --final, on the confirmation seeds


def result_dir_of(split: str, min_share: float | None) -> Path:
    if split in M09_SEEDS:
        return M09_RESULT_DIR
    return RESULT_DIR if min_share is None else FLOOR_RESULT_DIR


def run_split(split: str, models: list[Path], final: bool, min_share: float | None = None) -> None:
    result_dir = result_dir_of(split, min_share)
    final_result = result_dir / "test_result.json"
    if split in FINAL_SPLITS:
        if not final:
            raise SystemExit("the held-out test runs once, on the confirmation seeds: add --final")
        if final_result.exists():
            raise SystemExit(f"{final_result} exists: the held-out test has already been run")
        refuse_if_spent(f"goal_test/{split}")  # the committed register: holds even if runs/ is gone
        if split == "test" and min_share is not None:
            print("NOTE: second use of the held-out test set (first: M02b, runs/m02b/test_result.json)")
    test = GoalTest(split, min_share)
    print(f"{split}: material cases {sum(v >= MATERIAL_THRESHOLD for v in test.oracle_adv.values())} "
          f"of {len(test.oracle_adv)}; equal split reward {test.equal_reward:.3f}")
    print(f"{'model':<28}{'T1a':>6}{'T1b':>7}{'reward':>8}{'orig gap':>10}  verdict")
    results = {}
    for path in models:
        policy = SlicePolicy.load(path, test.alpha)
        m = test.evaluate(policy)
        # The M02 criterion is on the M01 cell's URLLC spike: not meaningful for the M09c splits.
        m["original_gap"] = original_criterion(policy, min_share) if split not in M09_SEEDS else float("nan")
        results[path.stem] = m
        print(f"{path.stem:<28}{m['t1a_share_positive']:6.2f}{m['t1b_captured']:7.2f}{m['reward']:8.3f}"
              f"{m['original_gap']:10.3f}  {'PASS' if m['pass'] else 'FAIL'}")
    if split in FINAL_SPLITS:
        for name in runs_without_best(models):
            results[name] = {"pass": False, "no_checkpoint": True, "t1b_captured": float("nan")}
            print(f"{name:<28}  no checkpoint (never beat the equal split on validation)  FAIL")
    recipes: dict[str, list[str]] = {}
    for name in results:
        recipes.setdefault(recipe_of(name), []).append(name)
    summary = {r: {"all_pass": all(results[n]["pass"] for n in names),
                   "mean_t1b": float(np.nanmean([results[n]["t1b_captured"] for n in names])),
                   "seeds": len(names)} for r, names in recipes.items()}
    print("\nper recipe (every seed must pass):")
    for r, s in summary.items():
        print(f"  {r:<26} seeds {s['seeds']}  mean T1b {s['mean_t1b']:.2f}  {'PASS' if s['all_pass'] else 'FAIL'}")
    out = {"split": split, "material_threshold": MATERIAL_THRESHOLD, "t1a_min": T1A_MIN_SHARE,
           "t1b_min": T1B_MIN_CAPTURED, "min_share": min_share, "models": results, "recipes": summary}
    if split == "test" and min_share is not None:
        out["test_set_use"] = "second (first: M02b)"
    if split == "validation":
        passing = {r: s for r, s in summary.items() if s["all_pass"]}
        chosen = max(passing, key=lambda r: passing[r]["mean_t1b"]) if passing else None
        out["chosen_recipe"] = chosen
        print("\nchosen recipe (rule: all seeds pass on validation, then highest mean T1b):",
              chosen or "NONE: stop and report before touching the test")
    path = final_result if split in FINAL_SPLITS else result_dir / f"{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    if split in FINAL_SPLITS:
        mark_spent(f"goal_test/{split}", path, note=f"{len(models)} models, min_share {min_share}")
    # One MLflow evaluation run per model; the registry gate reads the final ones (mlops/registry.py).
    for name, m in results.items():
        tags = {"split": split, "final": str(split in FINAL_SPLITS).lower(),
                "recipe_all_pass": str(summary[recipe_of(name)]["all_pass"]),
                "min_share": str(min_share)}
        tracking.log_evaluation("goal-test", name, m, tags, result_file=path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["validation", "test", *M09_SEEDS])
    parser.add_argument("--check", action="store_true", help="validity checks of both case sets")
    parser.add_argument("--models", default=None, help="glob of model zips (default depends on --split)")
    parser.add_argument("--final", action="store_true", help="required to run the held-out test")
    parser.add_argument("--min-share", type=float, default=None, help="per-slice PRB floor (M04a: 0.05)")
    args = parser.parse_args()
    default = {"validation": "models/m02b/dev/*_best.zip", "test": "models/m02b/confirm/*_best.zip",
               "m09_validation": "models/m09c/dev/*_best.zip", "m09_test": "models/m09c/confirm/*_best.zip"}
    pattern = args.models or default.get(args.split or "", "models/m02/*_seed*.zip")
    models = sorted(Path().glob(pattern))
    if args.check:
        check(models, args.min_share)
        return
    if not args.split:
        parser.error("give --split or --check")
    if not models:
        raise SystemExit(f"no models match {pattern}")
    run_split(args.split, models, args.final, args.min_share)


if __name__ == "__main__":
    main()
