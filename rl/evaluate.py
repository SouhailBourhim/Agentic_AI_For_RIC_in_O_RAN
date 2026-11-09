"""Evaluate the trained policies against the baselines, and test goal conditioning.

Run with:  uv run python -m rl.evaluate        (after training; reads models/m02/*.zip)
Writes runs/m02/eval.json and prints two tables plus the pass/fail of both criteria.

1. Goal conditioning: the same URLLC-spike traffic under two objectives ("protect URLLC" vs
   "maximise eMBB"). A goal-conditioned policy must split the PRBs differently.
   Criterion: the mean URLLC share differs by at least 0.10, for every trained seed.
2. Random episodes: 100 episodes with a random objective and random traffic (env seeds
   1000-1099). Every controller sees exactly the same episodes (paired comparison).
   Criterion: the policy's mean reward beats the equal split, for every trained seed.
Both criteria were fixed before any result existed, and are judged per training round
(all seeds of one group). Round 1 (ddpg, td3) registered them on the raw actor; round 2
(td3v2) on the deployed policy with the smoothing filter. CRITERION_VARIANT records that, so
each group is judged on the variant fixed for it; the other variant is printed for information.

3. Diagnostic, added after the first DDPG results (not a criterion): over 2 000 real states,
   how much the URLLC share moves when only the objective part of the observation is swapped
   between the two objectives above. It tells whether a policy ignores the objective in
   general, or only in the states the spike scenario reaches. It uses the raw actor: the
   smoothing filter works over time and means nothing across unrelated states.

Rewards use the env default, reward v2 since M02b (sim/reward.py); the M02 round 1-2 numbers in
the learning note were computed with reward v1.

Policies are evaluated as deployed, i.e. with the output smoothing filter of rl/policy.py
(the criteria use these), and also raw ("<name> raw") for comparison.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from mlops import tracking
from rl.baselines import equal_split, proportional_to_load
from rl.policy import SlicePolicy
from sim.cell import CellConfig, shares_from_action
from sim.env import OBJECTIVE_SLICE, SlicingEnv, objective_to_vector
from sim.plot import run_episode
from sim.scenarios import urllc_spike
from verifier.objective import SLICES, RLObjective, SliceValues

MODEL_DIR = Path("models/m02")
OUT_PATH = Path("runs/m02/eval.json")
EVAL_SEEDS = range(1000, 1100)
EPISODE_STEPS = 200
SPIKE_STEPS = 40  # the URLLC-spike scenario: spike at 10-20 s, then 20 s of recovery
SHARE_GAP_CRITERION = 0.10
TARGET_TOLERANCE = 0.99  # a target counts as met at >= 99 % of it (float noise)
# The policy variant each group's criteria were registered on, before its results existed.
CRITERION_VARIANT = {"ddpg": "raw", "td3": "raw", "td3v2": "filtered"}

CAPACITY = CellConfig().capacity_mbps
OBJECTIVES = {
    "protect URLLC": RLObjective(
        weights=SliceValues(embb=0.2, urllc=0.6, be=0.2),
        min_thr_mbps=SliceValues(urllc=0.30 * CAPACITY),
        max_latency_ms=10.0,
    ),
    "maximise eMBB": RLObjective(weights=SliceValues(embb=0.8, urllc=0.1, be=0.1)),
}

Controller = Callable[[np.ndarray], np.ndarray]


def episode_metrics(controller: Controller, seed: int, steps: int = EPISODE_STEPS,
                    min_share: float | None = None, env_options: dict | None = None) -> dict[str, float]:
    """Run one random episode and summarise it.

    - reward: mean reward per step
    - target_met: share of (step, slice) pairs with a minimum target where
      throughput >= min(target, offered load). NaN if the objective has no target.
    - latency_met: share of steps with URLLC latency <= bound. NaN if there is no bound.
    - split_change: mean step-to-step change of the PRB split, sum(|share_t - share_t-1|) / 2,
      i.e. the fraction of the cell's PRBs moved each step. Diagnostic added after the M02
      figures showed a policy flipping its split every step.
    """
    # min_share: the M04a per-slice floor; env_options: the M09c cell (rl.goal_test.M09_SETTING)
    env = SlicingEnv(episode_steps=steps, min_share=min_share, **(env_options or {}))
    obs, _ = env.reset(seed=seed)
    if hasattr(controller, "reset"):  # the smoothed policy starts each episode fresh
        controller.reset()
    objective = env.objective
    targets = dict(zip(SLICES, objective.min_thr_mbps.as_list()))
    rewards, target_hits, latency_hits, changes = [], [], [], []
    previous = None
    for _ in range(steps):
        obs, reward, _, _, info = env.step(np.asarray(controller(obs), dtype=np.float32))
        rewards.append(reward)
        shares = np.array([info["shares"][s] for s in SLICES])
        if previous is not None:
            changes.append(np.abs(shares - previous).sum() / 2)
        previous = shares
        for s in SLICES:
            if targets[s] > 0:
                needed = min(targets[s], info["load_mbps"][s])
                target_hits.append(info["throughput_mbps"][s] >= TARGET_TOLERANCE * needed)
        if objective.max_latency_ms is not None:
            latency_hits.append(info["latency_ms"]["urllc"] <= objective.max_latency_ms)
    return {
        "reward": float(np.mean(rewards)),
        "target_met": float(np.mean(target_hits)) if target_hits else float("nan"),
        "latency_met": float(np.mean(latency_hits)) if latency_hits else float("nan"),
        "split_change": float(np.mean(changes)),
    }


def evaluate_random(controller: Controller) -> dict[str, list[float]]:
    """All metrics, one value per evaluation episode, in EVAL_SEEDS order."""
    per_episode = [episode_metrics(controller, seed) for seed in EVAL_SEEDS]
    return {key: [m[key] for m in per_episode] for key in per_episode[0]}


def goal_conditioning(policy: SlicePolicy) -> dict[str, dict[str, float]]:
    """Mean PRB share per slice, URLLC throughput and latency under each objective."""
    traffic, _ = urllc_spike()
    out = {}
    for name, objective in OBJECTIVES.items():
        run = run_episode(policy, traffic, objective, steps=SPIKE_STEPS)
        summary = {f"share_{s}": float(run[f"share_{s}"].mean()) for s in SLICES}
        summary["urllc_thr_mbps"] = float(run["thr_urllc"].mean())
        summary["urllc_lat_ms_median"] = float(np.median(run["lat_urllc"]))
        out[name] = summary
    return out


def collect_states(n_episodes: int = 20, steps: int = 100) -> np.ndarray:
    """Real observations from equal-split episodes on the first evaluation seeds."""
    states = []
    for seed in list(EVAL_SEEDS)[:n_episodes]:
        env = SlicingEnv(episode_steps=steps)
        obs, _ = env.reset(seed=seed)
        for _ in range(steps):
            obs, *_ = env.step(equal_split(obs))
            states.append(obs.copy())
    return np.array(states)


def objective_sensitivity(policy: SlicePolicy, states: np.ndarray) -> dict[str, float]:
    """URLLC share under "protect URLLC" minus under "maximise eMBB", same KPIs, per state."""
    cfg = CellConfig()
    gaps = []
    for obs in states:
        shares = {}
        for name, objective in OBJECTIVES.items():
            x = obs.copy()
            x[OBJECTIVE_SLICE] = objective_to_vector(objective, cfg)
            shares[name] = shares_from_action(policy.raw_action(x))[1]
        gaps.append(shares["protect URLLC"] - shares["maximise eMBB"])
    gaps = np.array(gaps)
    return {"mean": float(gaps.mean()), "median": float(np.median(gaps)),
            "share_of_states_gap_ge_0.10": float(np.mean(gaps >= SHARE_GAP_CRITERION))}


def algo_of(name: str) -> str:
    return name.split("_seed")[0]


def judged_name(name: str) -> str:
    """The evaluation entry a policy's pre-registered criteria are judged on."""
    return f"{name} raw" if CRITERION_VARIANT.get(algo_of(name), "filtered") == "raw" else name


def summarise(values: list[float]) -> str:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return f"{arr.mean():7.3f} ± {arr.std():.3f}" if arr.size else "    n/a"


def main() -> None:
    model_paths = sorted(MODEL_DIR.glob("*_seed*.zip"))
    if not model_paths:
        raise SystemExit(f"no models in {MODEL_DIR}; train first (uv run python -m rl.train)")
    policies = {p.stem: SlicePolicy.load(p) for p in model_paths}  # with the smoothing filter
    raw_policies = {f"{p.stem} raw": SlicePolicy.load(p, alpha=1.0) for p in model_paths}

    results: dict = {"eval_seeds": [EVAL_SEEDS.start, EVAL_SEEDS.stop], "goal_conditioning": {}, "random": {}}

    print("1) Goal conditioning: URLLC-spike traffic, mean over", SPIKE_STEPS, "s")
    print(f"{'policy':<20}{'objective':<16}{'eMBB':>7}{'URLLC':>7}{'BE':>7}{'URLLC Mb/s':>12}{'URLLC lat ms (median)':>23}")
    gaps = {}
    for name, policy in {**policies, **raw_policies}.items():
        gc = goal_conditioning(policy)
        results["goal_conditioning"][name] = gc
        for obj_name, row in gc.items():
            print(f"{name:<20}{obj_name:<16}{row['share_embb']:7.2f}{row['share_urllc']:7.2f}"
                  f"{row['share_be']:7.2f}{row['urllc_thr_mbps']:12.1f}{row['urllc_lat_ms_median']:23.1f}")
        gaps[name] = gc["protect URLLC"]["share_urllc"] - gc["maximise eMBB"]["share_urllc"]
    print("URLLC share gap (protect - maximise eMBB):", {k: round(v, 3) for k, v in gaps.items()})
    gaps = {judged_name(name): gaps[judged_name(name)] for name in policies}
    algos = sorted({algo_of(n) for n in policies})
    gc_pass = {a: all(g >= SHARE_GAP_CRITERION for n, g in gaps.items() if algo_of(n) == a) for a in algos}
    for a in algos:
        print(f"  {a}: criterion gap >= {SHARE_GAP_CRITERION} on every seed "
              f"(judged {CRITERION_VARIANT.get(a, 'filtered')}):", "PASS" if gc_pass[a] else "FAIL")

    print(f"\n2) {len(EVAL_SEEDS)} random episodes (objective + traffic), same episodes for every controller")
    controllers: dict[str, Controller] = {
        "equal split": equal_split,
        "proportional to load": proportional_to_load,
        **policies,
        **raw_policies,
    }
    print(f"{'controller':<24}{'reward/step':>18}{'targets met':>18}{'URLLC latency met':>20}{'split change/step':>20}")
    for name, controller in controllers.items():
        metrics = evaluate_random(controller)
        results["random"][name] = metrics
        print(f"{name:<24}{summarise(metrics['reward']):>18}{summarise(metrics['target_met']):>18}"
              f"{summarise(metrics['latency_met']):>20}{summarise(metrics['split_change']):>20}")
    equal_mean = np.mean(results["random"]["equal split"]["reward"])
    beats = {judged_name(n): float(np.mean(results["random"][judged_name(n)]["reward"]) - equal_mean)
             for n in policies}
    print("mean reward minus equal split:", {k: round(v, 4) for k, v in beats.items()})
    rand_pass = {a: all(d > 0 for n, d in beats.items() if algo_of(n) == a) for a in algos}
    for a in algos:
        print(f"  {a}: criterion beats equal split on every seed "
              f"(judged {CRITERION_VARIANT.get(a, 'filtered')}):", "PASS" if rand_pass[a] else "FAIL")

    print("\n3) Diagnostic (added after the first DDPG results, not a criterion):")
    print("   URLLC share gap over 2 000 real states, objective part swapped")
    states = collect_states()
    results["objective_sensitivity"] = {}
    for name, policy in policies.items():
        sens = objective_sensitivity(policy, states)
        results["objective_sensitivity"][name] = sens
        print(f"   {name:<14} mean {sens['mean']:6.3f}  median {sens['median']:6.3f}  "
              f"gap >= 0.10 in {sens['share_of_states_gap_ge_0.10']:.0%} of states")

    results["criteria"] = {
        "variant": CRITERION_VARIANT,
        "goal_conditioning_share_gap": {"gaps": gaps, "threshold": SHARE_GAP_CRITERION, "pass": gc_pass},
        "beats_equal_split": {"differences": beats, "pass": rand_pass},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}")
    for name in policies:  # one MLflow evaluation run per policy (mlops/tracking.py)
        metrics = {f"{k}_mean": float(np.nanmean(v)) for k, v in results["random"][name].items()}
        metrics["urllc_share_gap"] = float(gaps.get(judged_name(name), np.nan))
        tracking.log_evaluation("m02-eval", name, metrics, {"split": "m02-eval", "reward_version": "v2",
                                                            "judged_variant": CRITERION_VARIANT.get(algo_of(name), "filtered")})


if __name__ == "__main__":
    main()
