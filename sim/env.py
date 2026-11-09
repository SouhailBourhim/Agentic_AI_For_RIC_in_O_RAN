"""Gymnasium environment: one cell, three slices, a goal-conditioned observation.

- Action: 3 numbers in [0, 1], turned into PRB shares that sum to 1.
- Observation: KPIs of the last step per slice, plus the current objective.
- Reward: see sim/reward.py. It depends on the objective.
- Episode: a fixed number of steps, then `truncated` (the cell never "ends").

Throughputs and targets are divided by the cell capacity, so the policy sees fractions
and keeps working when the capacity is recalibrated to the testbed.
"""

import dataclasses
from collections import deque
from collections.abc import Mapping
from typing import Any, Literal

import gymnasium as gym
import numpy as np

from sim.cell import CellConfig, SliceStep, apply_min_share, shares_from_action, smooth_split, step_slice
from sim.reward import DEFAULT_REWARD_VERSION, RewardVersion, compute_reward
from sim.traffic import DEFAULT_SURGE_RANGE, TrafficPattern, sample_traffic
from verifier.objective import (
    DEFAULT_OBJECTIVE,
    SLICES,
    TRAINED_LATENCY_BOUND_MS,
    TRAINED_MAX_TARGET_FRACTION,
    RLObjective,
    SliceValues,
)

LATENCY_SCALE_MS = 100.0  # latencies enter the observation as latency / 100 ms
OBS_CLIP = 10.0  # every observation value is clipped to [0, 10]
KPI_FEATURES_PER_SLICE = 5  # load, throughput, latency, prb_share, prb_used
OBJECTIVE_FEATURES = 7  # 3 weights, 3 minimum targets, 1 latency bound
OBS_SIZE = KPI_FEATURES_PER_SLICE * len(SLICES) + OBJECTIVE_FEATURES
# Where the objective sits in the observation, whatever follows it (M09c appends the load history after it).
OBJECTIVE_SLICE = slice(KPI_FEATURES_PER_SLICE * len(SLICES), OBS_SIZE)
# M09c: each slice's peak offered load over the last HISTORY_S seconds, / C, appended after the objective.
# A policy trained without it reads only the first OBS_SIZE values (rl.policy.SlicePolicy), so v7 is unchanged.
HISTORY_FEATURES = len(SLICES)
HISTORY_S = 30.0
OBS_SIZE_WITH_HISTORY = OBS_SIZE + HISTORY_FEATURES
HistoryMode = Literal["off", "on"]
# M09h: the split applied at the last step, undelayed, appended after the history. The KPIs' own prb_share is as late
# as the rest of the KPIs (2 ticks on the testbed), though the controller knows what it applied. 25 -> 28 values.
APPLIED_FEATURES = len(SLICES)
OBS_SIZE_WITH_APPLIED = OBS_SIZE_WITH_HISTORY + APPLIED_FEATURES


class LoadHistory:
    """Each slice's peak offered load over the last HISTORY_S seconds, from the loads the policy sees.

    One instance in SlicingEnv (training) and one in the RL runner (deployment): the same code
    gives the same features in both. It is fed the late KPIs the policy reads, not the true ones,
    so in training it has exactly the lag it will have when deployed. Why a peak: a latency bound
    under bursty load is met by sizing to the recent peak, which a one-tick observation can't
    see (M09b: F3, LLM + formula 0.92 against v7's 0.60).
    """

    def __init__(self, window_s: float = HISTORY_S, dt_s: float = 1.0) -> None:
        self.loads: deque[list[float]] = deque(maxlen=max(1, round(window_s / dt_s)))

    def reset(self) -> None:
        self.loads.clear()

    def update(self, loads_mbps: Mapping[str, float]) -> None:
        self.loads.append([float(loads_mbps[s]) for s in SLICES])

    def features(self, capacity_mbps: float) -> np.ndarray:
        if not self.loads:
            return np.zeros(HISTORY_FEATURES, dtype=np.float32)
        return np.clip(np.max(np.array(self.loads), axis=0) / capacity_mbps, 0.0, OBS_CLIP).astype(np.float32)


def applied_features(shares) -> np.ndarray:
    """The split applied at the last step as policy input (M09h): the same function in SlicingEnv and the runner."""
    return np.clip(np.asarray(shares, dtype=np.float64), 0.0, 1.0).astype(np.float32)


def objective_vectors(
    weights: np.ndarray, targets_mbps: np.ndarray, bound_ms: np.ndarray, cfg: CellConfig
) -> np.ndarray:
    """n objectives as an (n, 7) array for the policy input (weights, targets, bound).

    A latency bound of 0 means "no bound": a real bound is always > 0 (the schema checks it).
    bound_ms uses NaN for "no bound", as in sim/reward.py.
    """
    bound = np.where(np.isnan(bound_ms), 0.0, bound_ms / LATENCY_SCALE_MS)
    vectors = np.concatenate([weights, targets_mbps / cfg.capacity_mbps, bound[:, None]], axis=1)
    return np.clip(vectors, 0.0, OBS_CLIP)


def objective_arrays(objective: RLObjective) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One RLObjective as (weights (1, 3), targets_mbps (1, 3), bound_ms (1,)), NaN = no bound."""
    bound = np.nan if objective.max_latency_ms is None else objective.max_latency_ms
    return (np.array([objective.weights.as_list()]), np.array([objective.min_thr_mbps.as_list()]),
            np.array([bound]))


def objective_to_vector(objective: RLObjective, cfg: CellConfig) -> np.ndarray:
    """The objective as 7 numbers for the policy input."""
    return objective_vectors(*objective_arrays(objective), cfg)[0]


def build_observation(
    loads_mbps: Mapping[str, float],
    throughput_mbps: Mapping[str, float],
    latency_ms: Mapping[str, float],
    shares: Mapping[str, float],
    prb_used: Mapping[str, float],
    objective: RLObjective,
    cfg: CellConfig,
) -> np.ndarray:
    """The policy input: 5 KPI features per slice (SLICES order), then the 7 objective features.

    Per slice: load / C, throughput / C, latency / 100 ms, PRB share, PRB use. A pure function,
    shared by SlicingEnv (training) and the RL runner (rl/runner.py, deployment), so the policy
    sees exactly the same vector in both places; a test checks it.
    """
    capacity = cfg.capacity_mbps
    kpis = []
    for s in SLICES:
        kpis += [
            loads_mbps[s] / capacity,
            throughput_mbps[s] / capacity,
            latency_ms[s] / LATENCY_SCALE_MS,
            shares[s],
            prb_used[s],
        ]
    obs = np.concatenate([np.array(kpis), objective_to_vector(objective, cfg)])
    return np.clip(obs, 0.0, OBS_CLIP).astype(np.float32)


def sample_objective(rng: np.random.Generator, cfg: CellConfig) -> RLObjective:
    """A random objective, so training covers many goals (goal-conditioned policy).

    Weights come from a flat Dirichlet distribution: every split of importance is equally
    likely. Each minimum target, and the URLLC latency bound, is present half of the time.
    """
    weights = rng.dirichlet(np.ones(len(SLICES)))
    weights = weights / weights.sum()  # guard against rounding before the schema check
    max_target = TRAINED_MAX_TARGET_FRACTION * cfg.capacity_mbps
    targets = [float(rng.uniform(0.0, max_target)) if rng.random() < 0.5 else 0.0 for _ in SLICES]
    bound = float(rng.uniform(*TRAINED_LATENCY_BOUND_MS)) if rng.random() < 0.5 else None
    return RLObjective(
        weights=SliceValues(**dict(zip(SLICES, map(float, weights)))),
        min_thr_mbps=SliceValues(**dict(zip(SLICES, targets))),
        max_latency_ms=bound,
    )


def sample_objective_vectors(
    rng: np.random.Generator, n: int, cfg: CellConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n random objectives at once, same distribution as sample_objective.

    Used by the objective relabeling of the replay buffer (rl/relabel.py), which needs a
    batch of objectives per gradient step; building 256 pydantic objects there would be slow.
    A test checks the two samplers agree statistically. sample_objective itself is left as
    it is, so the evaluation episodes, which depend on its random draws, do not change.
    Returns weights (n, 3), targets_mbps (n, 3), bound_ms (n,) with NaN for "no bound".
    """
    k = len(SLICES)
    weights = rng.dirichlet(np.ones(k), size=n)
    max_target = TRAINED_MAX_TARGET_FRACTION * cfg.capacity_mbps
    targets = np.where(rng.random((n, k)) < 0.5, rng.uniform(0.0, max_target, (n, k)), 0.0)
    bound = np.where(rng.random(n) < 0.5, rng.uniform(*TRAINED_LATENCY_BOUND_MS, n), np.nan)
    return weights, targets, bound


class SlicingEnv(gym.Env):
    """One cell with eMBB, URLLC and best-effort slices sharing a PRB budget."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: CellConfig | None = None,
        episode_steps: int = 200,
        load_scale: float = 1.0,
        objective_switch_prob: float = 0.0,
        surge_range: tuple[float, float] = DEFAULT_SURGE_RANGE,
        action_filter_alpha: float | None = None,
        reward_version: RewardVersion = DEFAULT_REWARD_VERSION,
        min_share: float | None = None,
        capacity_range: tuple[float, float] | None = None,
        kpi_delay_range: tuple[int, int] | None = None,
        history: HistoryMode = "off",
        expiry_share: float = 0.0,
        toggle_prob: float = 0.0,
        noise_std: float | None = None,
        applied_split: bool = False,
        switch_cost: float = 0.0,
    ) -> None:
        """Training options (all off by default, so evaluation episodes and scripted
        scenarios are unchanged):
        - load_scale multiplies the randomly sampled traffic, so more steps are in contention,
          which is where the objective changes the best action (M02 round 2);
        - objective_switch_prob is the chance, each step, of drawing a new random objective in
          the middle of the episode, as when the LLM changes the goal at run time (round 2);
        - surge_range is the peak/base factor range of sampled spikes and ramps (M02b);
        - action_filter_alpha applies the deployed smoothing filter inside the environment, so
          the policy trains with the actuator lag it will have when deployed (M02b). Do not
          combine it with a smoothed SlicePolicy, or the split would be filtered twice.
        reward_version picks the reward formula (sim/reward.py); "v2" is the default.
        min_share applies the per-slice PRB floor (sim.cell.apply_min_share) after the filter, as
        the RL runner does when deployed (M04a). None = no floor, the M01-M02b behaviour, so the
        recorded evaluations of earlier policies reproduce exactly.
        M09c options (all off by default; the random generator is touched only when they're on):
        - capacity_range: the cell's true capacity is drawn per episode in this range, while the
          observation, the objective and the reward are normalised by what the cell reports
          (cfg.told_capacity_mbps), as the runner does through backend.info();
        - kpi_delay_range: the KPI part of the observation is from k ticks ago, k drawn per episode
          (the testbed's KPIs arrive 2 ticks late, M07b); the objective part is always current;
        - history: "on" appends LoadHistory's features after the objective;
        - expiry_share: at an objective switch, the chance of returning to DEFAULT_OBJECTIVE (an
          intent expiring) instead of drawing a new objective;
        - toggle_prob, noise_std: passed to sample_traffic (on/off loads, load noise).
        M09h options (off by default):
        - applied_split: append the split applied at the last step, undelayed, after the history (needs history "on",
          so every feature keeps its place: v7 reads 22 values, v8 25, v9 28);
        - switch_cost: the reward's switching cost (sim/reward.py), on the applied split.
        """
        if applied_split and history != "on":
            raise ValueError("applied_split goes after the load history: it needs history='on'")
        self.base_cfg = cfg or CellConfig()
        self.cfg = self.base_cfg  # this episode's cell: its capacity is redrawn per episode with capacity_range
        self.obs_cfg = CellConfig(capacity_mbps=self.base_cfg.told_capacity_mbps)  # what the policy is told
        self.capacity_range = capacity_range
        self.kpi_delay_range = kpi_delay_range
        self.kpi_delay = 0
        self.history = history
        self.load_history = LoadHistory(dt_s=self.base_cfg.dt_s)
        self.expiry_share = expiry_share
        self.toggle_prob = toggle_prob
        self.noise_std = noise_std
        self.applied_split = applied_split
        self.switch_cost = switch_cost
        self._seen: deque[dict[str, dict[str, float]]] = deque(maxlen=(kpi_delay_range or (0, 0))[1] + 1)
        self.episode_steps = episode_steps
        self.load_scale = load_scale
        self.objective_switch_prob = objective_switch_prob
        self.surge_range = surge_range
        self.action_filter_alpha = action_filter_alpha
        self.reward_version = reward_version
        self.min_share = min_share
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(len(SLICES),), dtype=np.float32)
        size = OBS_SIZE if history == "off" else (OBS_SIZE_WITH_APPLIED if applied_split else OBS_SIZE_WITH_HISTORY)
        self.observation_space = gym.spaces.Box(0.0, OBS_CLIP, shape=(size,), dtype=np.float32)
        # Filled by reset():
        self.objective: RLObjective
        self.traffic: dict[str, TrafficPattern]
        self.time_s = 0.0
        self.step_count = 0
        self.shares = np.full(len(SLICES), 1.0 / len(SLICES))  # the split applied (after the floor)
        self.previous_shares = self.shares  # the one applied the step before (switching cost, M09h)
        self._filtered: np.ndarray | None = None  # the filter's memory (before the floor), as in SlicePolicy
        self.loads: dict[str, float] = {}
        self.last: dict[str, SliceStep] = {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode. `options` may give "objective" and/or "traffic"; else random."""
        super().reset(seed=seed)
        options = options or {}
        episode_s = self.episode_steps * self.cfg.dt_s
        if self.capacity_range is not None:
            capacity = float(self.np_random.uniform(*self.capacity_range))
            self.cfg = dataclasses.replace(self.base_cfg, capacity_mbps=capacity,
                                           reported_capacity_mbps=self.base_cfg.told_capacity_mbps)
        self.objective = options.get("objective") or sample_objective(self.np_random, self.obs_cfg)
        self.traffic = options.get("traffic") or sample_traffic(
            self.np_random, episode_s, self.load_scale * self.cfg.capacity_mbps, self.surge_range,
            self.toggle_prob, self.noise_std,
        )
        if self.kpi_delay_range is not None:
            self.kpi_delay = int(self.np_random.integers(self.kpi_delay_range[0], self.kpi_delay_range[1] + 1))
        self.time_s = 0.0
        self.step_count = 0
        self.shares = np.full(len(SLICES), 1.0 / len(SLICES))
        self.previous_shares = self.shares
        self._filtered = None
        # Before the first step nothing has been sent: zero throughput, empty queues.
        self.loads = {s: self.traffic[s].mean_load_at(0.0) for s in SLICES}
        self.last = {
            s: SliceStep(0.0, 0.0, 0.0, 0.0, self.cfg.base_latency_ms[s]) for s in SLICES
        }
        self._seen.clear()
        self.load_history.reset()
        self._advance_view()
        return self._observation(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply a PRB split for dt seconds and report what happened."""
        target = shares_from_action(action)
        if self.action_filter_alpha is None:
            split = target
        else:  # same filter as the deployed policy; nothing to smooth with on the first step
            # The filter remembers its own output (before the floor), exactly like SlicePolicy;
            # without a floor that is the applied split, as before M04a.
            previous = None if self.step_count == 0 else self._filtered
            split = smooth_split(previous, target, self.action_filter_alpha)
        self._filtered = split
        self.previous_shares = self.shares
        self.shares = split if self.min_share is None else apply_min_share(split, self.min_share)
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
        reward = compute_reward(self.objective, self.last, self.loads, self.obs_cfg, self.reward_version,
                                shares=self.shares, urllc_base_ms=self.cfg.told_urllc_base_latency_ms,
                                switch_cost=self.switch_cost, previous_shares=self.previous_shares)
        # Only touch the random generator when the option is on, so that with the default the
        # random stream (hence every evaluation episode) is exactly what it was before.
        if self.objective_switch_prob > 0 and self.np_random.random() < self.objective_switch_prob:
            if self.expiry_share > 0 and self.np_random.random() < self.expiry_share:
                self.objective = DEFAULT_OBJECTIVE  # the intent expired: the runner's default again
            else:
                self.objective = sample_objective(self.np_random, self.obs_cfg)  # seen from the next step
        self.time_s += self.cfg.dt_s
        self.step_count += 1
        self._advance_view()
        truncated = self.step_count >= self.episode_steps
        return self._observation(), reward, False, truncated, self._info()

    def set_objective(self, objective: RLObjective) -> None:
        """Change the objective mid-episode (a new intent arrives). Takes effect next step."""
        self.objective = objective

    def _advance_view(self) -> None:
        """Record this step's KPIs, and feed the history what the policy sees (kpi_delay ticks late)."""
        self._seen.append({
            "loads_mbps": dict(self.loads),
            "throughput_mbps": {s: self.last[s].throughput_mbps for s in SLICES},
            "latency_ms": {s: self.last[s].latency_ms for s in SLICES},
            "shares": dict(zip(SLICES, map(float, self.shares))),
            "prb_used": {s: self.last[s].prb_used for s in SLICES},
        })
        self.load_history.update(self._view()["loads_mbps"])

    def _view(self) -> dict[str, dict[str, float]]:
        """The KPIs the policy sees: kpi_delay ticks ago, or the oldest kept at an episode's start."""
        return self._seen[max(0, len(self._seen) - 1 - self.kpi_delay)]

    def _observation(self) -> np.ndarray:
        """5 KPI features per slice, then the 7 objective features (see build_observation),
        then the load history if the env has it, then the split applied at the last step (M09h)."""
        obs = build_observation(**self._view(), objective=self.objective, cfg=self.obs_cfg)
        if self.history == "off":
            return obs
        parts = [obs, self.load_history.features(self.obs_cfg.capacity_mbps)]
        if self.applied_split:
            parts.append(applied_features(self.shares))
        return np.concatenate(parts).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        """Raw values in real units, for logs, the demo and the KPI store later."""
        return {
            "time_s": self.time_s,
            "shares": dict(zip(SLICES, map(float, self.shares))),
            "previous_shares": dict(zip(SLICES, map(float, self.previous_shares))),
            "load_mbps": dict(self.loads),
            "throughput_mbps": {s: self.last[s].throughput_mbps for s in SLICES},
            "latency_ms": {s: self.last[s].latency_ms for s in SLICES},
            "prb_used": {s: self.last[s].prb_used for s in SLICES},
            "backlog_mb": {s: self.last[s].backlog_mb for s in SLICES},
            "dropped_mb": {s: self.last[s].dropped_mb for s in SLICES},
        }
