"""Gymnasium environment: one cell, three slices, a goal-conditioned observation.

- Action: 3 numbers in [0, 1], turned into PRB shares that sum to 1.
- Observation: KPIs of the last step per slice, plus the current objective.
- Reward: see sim/reward.py. It depends on the objective.
- Episode: a fixed number of steps, then `truncated` (the cell never "ends").

Throughputs and targets are divided by the cell capacity, so the policy sees fractions
and keeps working when the capacity is recalibrated to the testbed.
"""

from typing import Any

import gymnasium as gym
import numpy as np

from sim.cell import CellConfig, SliceStep, shares_from_action, step_slice
from sim.reward import compute_reward
from sim.traffic import TrafficPattern, sample_traffic
from verifier.objective import SLICES, RLObjective, SliceValues

LATENCY_SCALE_MS = 100.0  # latencies enter the observation as latency / 100 ms
OBS_CLIP = 10.0  # every observation value is clipped to [0, 10]
KPI_FEATURES_PER_SLICE = 5  # load, throughput, latency, prb_share, prb_used
OBJECTIVE_FEATURES = 7  # 3 weights, 3 minimum targets, 1 latency bound
OBS_SIZE = KPI_FEATURES_PER_SLICE * len(SLICES) + OBJECTIVE_FEATURES


def objective_to_vector(objective: RLObjective, cfg: CellConfig) -> np.ndarray:
    """The objective as 7 numbers for the policy input.

    A latency bound of 0 means "no bound": a real bound is always > 0 (the schema checks it).
    """
    weights = objective.weights.as_list()
    targets = [t / cfg.capacity_mbps for t in objective.min_thr_mbps.as_list()]
    bound = 0.0 if objective.max_latency_ms is None else objective.max_latency_ms / LATENCY_SCALE_MS
    return np.clip(np.array(weights + targets + [bound]), 0.0, OBS_CLIP)


def sample_objective(rng: np.random.Generator, cfg: CellConfig) -> RLObjective:
    """A random objective, so training covers many goals (goal-conditioned policy).

    Weights come from a flat Dirichlet distribution: every split of importance is equally
    likely. Each minimum target, and the URLLC latency bound, is present half of the time.
    """
    weights = rng.dirichlet(np.ones(len(SLICES)))
    weights = weights / weights.sum()  # guard against rounding before the schema check
    targets = [
        float(rng.uniform(0.0, 0.4 * cfg.capacity_mbps)) if rng.random() < 0.5 else 0.0
        for _ in SLICES
    ]
    bound = float(rng.uniform(5.0, 50.0)) if rng.random() < 0.5 else None
    return RLObjective(
        weights=SliceValues(**dict(zip(SLICES, map(float, weights)))),
        min_thr_mbps=SliceValues(**dict(zip(SLICES, targets))),
        max_latency_ms=bound,
    )


class SlicingEnv(gym.Env):
    """One cell with eMBB, URLLC and best-effort slices sharing a PRB budget."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: CellConfig | None = None,
        episode_steps: int = 200,
        load_scale: float = 1.0,
        objective_switch_prob: float = 0.0,
    ) -> None:
        """`load_scale` and `objective_switch_prob` are training options (M02, round 2):
        - load_scale multiplies the randomly sampled traffic, so more steps are in contention,
          which is where the objective changes the best action;
        - objective_switch_prob is the chance, each step, of drawing a new random objective in
          the middle of the episode, as when the LLM changes the goal at run time.
        Both are off by default, so evaluation episodes and scripted scenarios are unchanged.
        """
        self.cfg = cfg or CellConfig()
        self.episode_steps = episode_steps
        self.load_scale = load_scale
        self.objective_switch_prob = objective_switch_prob
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(len(SLICES),), dtype=np.float32)
        self.observation_space = gym.spaces.Box(0.0, OBS_CLIP, shape=(OBS_SIZE,), dtype=np.float32)
        # Filled by reset():
        self.objective: RLObjective
        self.traffic: dict[str, TrafficPattern]
        self.time_s = 0.0
        self.step_count = 0
        self.shares = np.full(len(SLICES), 1.0 / len(SLICES))
        self.loads: dict[str, float] = {}
        self.last: dict[str, SliceStep] = {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode. `options` may give "objective" and/or "traffic"; else random."""
        super().reset(seed=seed)
        options = options or {}
        episode_s = self.episode_steps * self.cfg.dt_s
        self.objective = options.get("objective") or sample_objective(self.np_random, self.cfg)
        self.traffic = options.get("traffic") or sample_traffic(
            self.np_random, episode_s, self.load_scale * self.cfg.capacity_mbps
        )
        self.time_s = 0.0
        self.step_count = 0
        self.shares = np.full(len(SLICES), 1.0 / len(SLICES))
        # Before the first step nothing has been sent: zero throughput, empty queues.
        self.loads = {s: self.traffic[s].mean_load_at(0.0) for s in SLICES}
        self.last = {
            s: SliceStep(0.0, 0.0, 0.0, 0.0, self.cfg.base_latency_ms[s]) for s in SLICES
        }
        return self._observation(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply a PRB split for dt seconds and report what happened."""
        self.shares = shares_from_action(action)
        self.loads = {s: self.traffic[s].load_at(self.time_s, self.np_random) for s in SLICES}
        self.last = {
            s: step_slice(
                share=float(self.shares[i]),
                load_mbps=self.loads[s],
                backlog_mb=self.last[s].backlog_mb,
                base_latency_ms=self.cfg.base_latency_ms[s],
                cfg=self.cfg,
            )
            for i, s in enumerate(SLICES)
        }
        reward = compute_reward(self.objective, self.last, self.loads, self.cfg)
        # Only touch the random generator when the option is on, so that with the default the
        # random stream (hence every evaluation episode) is exactly what it was before.
        if self.objective_switch_prob > 0 and self.np_random.random() < self.objective_switch_prob:
            self.objective = sample_objective(self.np_random, self.cfg)  # seen from the next step
        self.time_s += self.cfg.dt_s
        self.step_count += 1
        truncated = self.step_count >= self.episode_steps
        return self._observation(), reward, False, truncated, self._info()

    def set_objective(self, objective: RLObjective) -> None:
        """Change the objective mid-episode (a new intent arrives). Takes effect next step."""
        self.objective = objective

    def _observation(self) -> np.ndarray:
        """5 KPI features per slice, then the 7 objective features."""
        capacity = self.cfg.capacity_mbps
        kpis = []
        for i, s in enumerate(SLICES):
            kpis += [
                self.loads[s] / capacity,
                self.last[s].throughput_mbps / capacity,
                self.last[s].latency_ms / LATENCY_SCALE_MS,
                self.shares[i],
                self.last[s].prb_used,
            ]
        obs = np.concatenate([np.array(kpis), objective_to_vector(self.objective, self.cfg)])
        return np.clip(obs, 0.0, OBS_CLIP).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        """Raw values in real units, for logs, the demo and the KPI store later."""
        return {
            "time_s": self.time_s,
            "shares": dict(zip(SLICES, map(float, self.shares))),
            "load_mbps": dict(self.loads),
            "throughput_mbps": {s: self.last[s].throughput_mbps for s in SLICES},
            "latency_ms": {s: self.last[s].latency_ms for s in SLICES},
            "prb_used": {s: self.last[s].prb_used for s in SLICES},
            "backlog_mb": {s: self.last[s].backlog_mb for s in SLICES},
            "dropped_mb": {s: self.last[s].dropped_mb for s in SLICES},
        }
