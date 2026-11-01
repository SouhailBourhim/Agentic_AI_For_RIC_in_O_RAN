"""Oracle controller: a reference for what a controller that knows the model can achieve.

Used only to evaluate the RL policy (rl/goal_test.py), never deployed. It is allowed to read
the simulator's internal state (queue backlogs), to use the simulator's own equations, and to
know the traffic of the next second exactly (it replays the env's random generator). No real
controller can do any of that, which is the point: it is a demanding reference, and the goal
test asks the policy to capture half of what it gains from knowing the objective.

Why it knows the next second. A first version assumed the next loads equal the last observed
ones. It was too weak a reference: at a surge onset it cannot react in time, and it never
keeps headroom, so on the URLLC spike a fixed generous split (0.3 / 0.5 / 0.2) beat it under
"protect URLLC". A weak oracle would make the test easy; a strong one keeps it honest.

Every second it:
1. takes every PRB split on a 0.025 grid (861 splits summing to 1) as a candidate target;
2. passes each through the same smoothing filter as the deployed policy (rl/policy.py), so
   it faces the same actuator lag;
3. predicts the next second for each applied split with the cell model and the true loads
   (sim.cell.step_slices_batch) and scores it with the reward of its objective;
4. applies the best one. Ties go to the split closest to the current one (deterministic).

With min_share (M04a), step 3 scores each candidate after the per-slice floor
(sim.cell.apply_min_share), exactly as the env will apply it: the oracle plays by the same
actuator rules as the policy it is compared with. Without it, the M02b oracle is unchanged.

It is greedy (one second ahead), so it is not a perfect optimum: it does not plan queue
draining over several seconds. The policy can beat it in some cases (the test clips at 1).
"""

import copy

import numpy as np

from rl.policy import SMOOTHING_ALPHA
from sim.cell import CellConfig, apply_min_share, smooth_split, step_slices_batch
from sim.env import SlicingEnv, objective_arrays
from sim.reward import DEFAULT_REWARD_VERSION, RewardVersion, compute_reward_batch
from verifier.objective import SLICES, RLObjective

GRID_STEP = 0.025
TIE_TOLERANCE = 1e-9


def split_grid(step: float = GRID_STEP) -> np.ndarray:
    """All PRB splits (eMBB, URLLC, BE) on a grid of `step`, summing to 1: shape (n, 3)."""
    n = round(1.0 / step)
    rows = [(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)]
    return np.array(rows, dtype=np.float64) / n


class GreedyOracle:
    """One-second-ahead greedy controller for a given objective."""

    def __init__(
        self,
        objective: RLObjective,
        alpha: float | None = SMOOTHING_ALPHA,
        reward_version: RewardVersion = DEFAULT_REWARD_VERSION,
        min_share: float | None = None,
    ) -> None:
        self.weights, self.targets, self.bound = objective_arrays(objective)
        self.min_share = min_share
        self.alpha = alpha
        self.reward_version = reward_version
        self.candidates = split_grid()
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def act(self, env: SlicingEnv) -> np.ndarray:
        """The split to apply now (already smoothed), from the env's current state."""
        cfg: CellConfig = env.cfg
        filtered = (self.candidates if self.alpha is None
                    else smooth_split(self._previous, self.candidates, self.alpha))
        applied = filtered if self.min_share is None else apply_min_share(filtered, self.min_share)
        # The loads env.step() is about to draw: same generator state, same order of draws.
        rng = copy.deepcopy(env.np_random)
        loads = np.array([env.traffic[s].load_at(env.time_s, rng) for s in SLICES])
        backlog = np.array([env.last[s].backlog_mb for s in SLICES])
        base = np.array([cfg.base_latency_ms[s] for s in SLICES])
        pred = step_slices_batch(applied, loads, backlog, base, cfg)
        n = len(applied)
        rewards = compute_reward_batch(
            np.repeat(self.weights, n, axis=0), np.repeat(self.targets, n, axis=0),
            np.repeat(self.bound, n), pred["throughput_mbps"], np.broadcast_to(loads, (n, 3)),
            pred["latency_ms"][:, 1], cfg.capacity_mbps, self.reward_version,
        )
        best = rewards.max()
        tied = np.flatnonzero(rewards >= best - TIE_TOLERANCE)
        distance = np.abs(applied[tied] - env.shares).sum(axis=1)
        best_index = tied[np.argmin(distance)]
        self._previous = filtered[best_index]  # the filter's memory is before the floor, as in the env
        return filtered[best_index]  # the env applies the floor itself
