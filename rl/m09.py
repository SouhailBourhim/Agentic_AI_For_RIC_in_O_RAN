"""The M09c training setting: the cell as the policy will meet it, in one place for rl.train, rl.goal_test and rl.sweep.

What v8 trains on, against v7 (M04a floor round), and the M09b finding each change answers:
- the calibrated cell (eval.sim_profiles, M09a): flat RLC delay, 408 Mb buffer, true capacity drawn per episode in
  73.5-80.2 Mb/s while the policy is told 79 (v7 trained on an M/M/1 cell that is exactly what it's told, so it gave
  URLLC a big margin and settled about 1 Mb/s short of targets: F4a, F2);
- KPIs 1-3 ticks late, drawn per episode (the testbed's are 2 ticks late; v7 saw them current);
- the served filter, alpha 0.10, in the loop (v7 trained at 0.25 and is served at 0.10);
- objectives that switch every ~50 s, half the time back to the default (an intent expiring);
- on/off loads on a quarter of the slices, 5-40 s phases (F3's fast dynamics; never eval.generator's instances);
- the load history in the observation, and reward v3 (sim/reward.py), which charges surplus while a target is missed.
120-step episodes: long enough for a 30 s history, a few toggles and an expiry.

The four recipes of the dev round (DECISIONS) differ only by `RECIPES`; everything else is shared.
"""

from typing import Any

from eval.sim_profiles import TRUE_CAPACITY_MBPS, noise_std, testbed_cell
from rl.runner import SERVED_SMOOTHING_ALPHA
from sim.cell import CellConfig

EPISODE_STEPS = 120
KPI_DELAY_RANGE = (1, 3)
SWITCH_PROB = 0.02  # an objective change every ~50 s on average
EXPIRY_SHARE = 0.5
TOGGLE_PROB = 0.25
TEST_KPI_DELAY = (2, 2)  # the goal test scores at the testbed's delay, fixed
FILTER_ALPHA = SERVED_SMOOTHING_ALPHA  # trained with the filter it is served with

# The dev round's ablation: the full recipe, and three versions each missing one change.
RECIPES: dict[str, dict[str, Any]] = {
    "v8": {"profile": "m09", "reward": "v3", "history": "on"},
    "v8_mm1": {"profile": "m01", "reward": "v3", "history": "on"},
    "v8_nooverserve": {"profile": "m09", "reward": "v1", "history": "on"},
    "v8_nohistory": {"profile": "m09", "reward": "v3", "history": "off"},  # 22 inputs, as deployed for such a policy
}


def cell(profile: str) -> CellConfig:
    """m09: the calibrated cell, told 79 Mb/s (its true capacity is redrawn per episode); m01: the M01 cell."""
    return testbed_cell("m09", 0) if profile == "m09" else CellConfig()


def env_options(profile: str, history: str) -> dict[str, Any]:
    """SlicingEnv keyword arguments shared by training and the goal test (not the training-only ones).
    The goal test always uses ("m09", "on"): every recipe is scored on the cell it will be deployed on, and a
    policy trained without the history reads only its first 22 inputs (rl.policy.SlicePolicy)."""
    calibrated = profile == "m09"
    return {
        "cfg": cell(profile),
        "capacity_range": TRUE_CAPACITY_MBPS["m09"] if calibrated else None,
        "noise_std": noise_std() if calibrated else None,
        "history": history,
    }


def training_options(profile: str, history: str) -> dict[str, Any]:
    """Everything SlicingEnv gets in M09c training, on top of rl.train's usual options."""
    return {**env_options(profile, history), "kpi_delay_range": KPI_DELAY_RANGE, "objective_switch_prob": SWITCH_PROB,
            "expiry_share": EXPIRY_SHARE, "toggle_prob": TOGGLE_PROB}
