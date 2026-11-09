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

M09h (v9) keeps v8_nooverserve's setting (reward v1) and adds, after the M09g diagnostics (DECISIONS 2026-11-09):
- A1 a switching cost in the reward (SWITCH_COST x the share of the cell moved per step), served at the same alpha;
- A4 Ornstein-Uhlenbeck exploration noise, correlated over ~10 steps, so it isn't averaged away by the filter;
- A5 the split applied at the last step, undelayed, in the observation (25 -> 28 values);
- A6 gamma 0.98 (a ~50-step horizon, past the filter's ~10 s, the 2-tick delay and 5-15 s drains);
and as its own arm A7, the residual on the formula (rl/residual.py). The dev round is the full v9, one arm per change
removed, and the residual arm; every key below has a default, so the v8 recipes are unchanged.
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

# M09h: v9's options, with the value each has in the v8 recipes (V9_DEFAULTS) and in v9.
SWITCH_COST = 1.0  # at v7's 2.65 % moved per second, ~11 % of a typical step's reward (~0.24)
OU_THETA, OU_DT, OU_SIGMA = 0.1, 1.0, 0.1  # x' = 0.9 x + 0.1 N: ~10-step correlation, stationary std 0.23 ([-1, 1] scale)
V9_GAMMA = 0.98
V9_DEFAULTS: dict[str, Any] = {"applied": False, "switch_cost": 0.0, "noise": "normal", "gamma": 0.95, "residual": False}
V9: dict[str, Any] = {"profile": "m09", "reward": "v1", "history": "on",
                      "applied": True, "switch_cost": SWITCH_COST, "noise": "ou", "gamma": V9_GAMMA, "residual": False}
RECIPES |= {
    "v9": V9,
    "v9_noswitch": {**V9, "switch_cost": 0.0},  # A1 off
    "v9_noou": {**V9, "noise": "normal"},  # A4 off
    "v9_noapplied": {**V9, "applied": False},  # A5 off: 25 inputs
    "v9_gamma95": {**V9, "gamma": 0.95},  # A6 off
    "v9_residual": {**V9, "residual": True},  # A7: the residual arm, on top of the full v9
}
V8_RECIPES = ("v8", "v8_mm1", "v8_nooverserve", "v8_nohistory")
V9_RECIPES = tuple(r for r in RECIPES if r.startswith("v9"))


def option(recipe: str, key: str) -> Any:
    """One of v9's options for a recipe: its own value, else the v8 one."""
    return RECIPES[recipe].get(key, V9_DEFAULTS[key])


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


def training_options(profile: str, history: str, applied: bool = False, switch_cost: float = 0.0) -> dict[str, Any]:
    """Everything SlicingEnv gets in M09c/M09h training, on top of rl.train's usual options."""
    return {**env_options(profile, history), "kpi_delay_range": KPI_DELAY_RANGE, "objective_switch_prob": SWITCH_PROB,
            "expiry_share": EXPIRY_SHARE, "toggle_prob": TOGGLE_PROB,
            **({"applied_split": True} if applied else {}), **({"switch_cost": switch_cost} if switch_cost else {})}
