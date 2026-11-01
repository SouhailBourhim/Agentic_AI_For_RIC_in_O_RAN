"""Choose the policy to deploy, by a rule fixed before running it, on validation episodes only.

Run with:  uv run python -m rl.select        (after training; reads models/m02/*.zip)
Writes runs/m02/selection.json (the chosen policy is under "deployed").

Why this exists. The goal-conditioning criterion failed in both training rounds: some seeds
follow the objective on the URLLC-spike scenario, some do not (see rl/evaluate.py). We keep
that FAIL on record and still have to deploy one policy. Picking the seed that looked best on
the test scenario would turn the test into a selection step, so the choice uses episodes that
no criterion has ever used:

- validation reward: 100 random episodes (objective + traffic), env seeds 2000-2099;
- validation goal gap: 50 random traffic patterns (env seeds 2100-2149), each run twice with
  the same noise, under "protect URLLC" and under "maximise eMBB". Per pattern, the gap is the
  mean URLLC share under the first minus under the second.

The rule (fixed before any validation number existed), applied to every trained policy as
deployed (with the smoothing filter):
1. Eligible: mean validation reward above the equal split's on the same episodes.
2. Among the eligible, the highest share of validation patterns with a gap >= 0.10. This
   targets what failed: goal conditioning that holds in some scenarios but not in others.
3. Tie: the higher mean validation reward.
The test numbers of every policy were already seen when this rule was written, so the
selected policy's test result is not an unbiased estimate; M08 re-checks it on fresh episodes.
"""

import json
from pathlib import Path

import numpy as np

from rl.baselines import equal_split
from rl.evaluate import MODEL_DIR, OBJECTIVES, SHARE_GAP_CRITERION, Controller, episode_metrics
from rl.policy import SlicePolicy
from sim.env import SlicingEnv
from sim.plot import run_episode

OUT_PATH = Path("runs/m02/selection.json")
VALIDATION_REWARD_SEEDS = range(2000, 2100)  # disjoint from the test seeds 1000-1099
VALIDATION_TRAFFIC_SEEDS = range(2100, 2150)
GAP_EPISODE_STEPS = 100


def validation_reward(controller: Controller) -> float:
    """Mean reward per step over the validation episodes."""
    return float(np.mean([episode_metrics(controller, seed)["reward"] for seed in VALIDATION_REWARD_SEEDS]))


def validation_gaps(policy: SlicePolicy) -> np.ndarray:
    """One URLLC share gap per validation traffic pattern (protect URLLC - maximise eMBB)."""
    gaps = []
    for seed in VALIDATION_TRAFFIC_SEEDS:
        env = SlicingEnv(episode_steps=GAP_EPISODE_STEPS)
        env.reset(seed=seed)  # only to draw a random traffic pattern
        share = {
            name: run_episode(policy, env.traffic, objective, steps=GAP_EPISODE_STEPS, seed=seed)["share_urllc"].mean()
            for name, objective in OBJECTIVES.items()
        }
        gaps.append(share["protect URLLC"] - share["maximise eMBB"])
    return np.array(gaps)


def select(candidates: dict[str, dict[str, float]], equal_reward: float) -> str:
    """Apply the rule of the module docstring to the per-policy validation numbers."""
    eligible = {n: c for n, c in candidates.items() if c["reward"] > equal_reward}
    if not eligible:
        raise SystemExit("no policy beats the equal split on validation; nothing to deploy")
    return max(eligible, key=lambda n: (eligible[n]["share_gap_ge_0.10"], eligible[n]["reward"]))


def main() -> None:
    model_paths = sorted(MODEL_DIR.glob("*_seed*.zip"))
    if not model_paths:
        raise SystemExit(f"no models in {MODEL_DIR}; train first (uv run python -m rl.train)")
    equal_reward = validation_reward(equal_split)
    print(f"validation: rewards on env seeds {VALIDATION_REWARD_SEEDS.start}-{VALIDATION_REWARD_SEEDS.stop - 1}, "
          f"goal gap on {len(VALIDATION_TRAFFIC_SEEDS)} traffic patterns")
    print(f"{'policy':<14}{'reward/step':>12}{'mean gap':>10}{'gap >= 0.10':>13}")
    print(f"{'equal split':<14}{equal_reward:12.3f}")
    candidates = {}
    for path in model_paths:
        policy = SlicePolicy.load(path)
        gaps = validation_gaps(policy)
        candidates[path.stem] = {
            "reward": validation_reward(policy),
            "mean_gap": float(gaps.mean()),
            "share_gap_ge_0.10": float(np.mean(gaps >= SHARE_GAP_CRITERION)),
        }
        c = candidates[path.stem]
        print(f"{path.stem:<14}{c['reward']:12.3f}{c['mean_gap']:10.3f}{c['share_gap_ge_0.10']:13.0%}")
    deployed = select(candidates, equal_reward)
    print(f"\ndeployed policy (rule in rl/select.py): {deployed}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "validation_reward_seeds": [VALIDATION_REWARD_SEEDS.start, VALIDATION_REWARD_SEEDS.stop],
        "validation_traffic_seeds": [VALIDATION_TRAFFIC_SEEDS.start, VALIDATION_TRAFFIC_SEEDS.stop],
        "equal_split_reward": equal_reward,
        "candidates": candidates,
        "deployed": deployed,
    }, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
