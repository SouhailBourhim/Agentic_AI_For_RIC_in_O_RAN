"""Replay buffer with objective relabeling (M02b).

The problem it solves. In a normal replay buffer each stored step carries the one objective
that was active when it was collected. Most steps are uncongested, where every objective is
satisfied by almost any split, so few transitions tell the policy what each objective wants.

The idea. The objective does not change how the cell behaves (traffic, queues, throughput
and latency depend only on the split), it only changes how a step is scored. So a stored
transition (state, split, next state) is equally real under any other objective, as long as
the objective part of the observations and the reward are replaced consistently. Every time
a batch is sampled, RELABEL_FRACTION of it gets a fresh random objective: the 7 objective values of
obs and next_obs (sim.env.OBJECTIVE_SLICE) are overwritten, and the reward is recomputed from the step's raw KPIs with
the same reward function as the environment (sim.reward.compute_reward_batch). Each congested
state then teaches the policy what many objectives want. This is the idea behind hindsight
experience replay (Andrychowicz et al., 2017), applied to objectives instead of goal states.

Only for one environment at a time (n_envs = 1), which is how rl/train.py trains.
"""

from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from sim.cell import CellConfig
from sim.env import OBJECTIVE_SLICE, objective_vectors, sample_objective_vectors
from sim.reward import DEFAULT_REWARD_VERSION, RewardVersion, compute_reward_batch
from verifier.objective import SLICES

# throughput x3, load x3, URLLC latency, split x3, previous split x3: everything the reward needs (v3: the split;
# M09h's switching cost: the split before). The switching cost doesn't depend on the objective, but a relabeled
# reward is recomputed from these columns, so it must be in it too.
KPI_COLUMNS = 13


def kpis_from_info(info: dict[str, Any]) -> list[float]:
    """The raw KPIs of one step, from the env's info dict, in KPI column order. They are the true
    KPIs of the step, not the late ones the observation shows (M09c), as the env's reward uses."""
    return ([info["throughput_mbps"][s] for s in SLICES] + [info["load_mbps"][s] for s in SLICES]
            + [info["latency_ms"]["urllc"]] + [info["shares"][s] for s in SLICES]
            + [info["previous_shares"][s] for s in SLICES])


class RelabelReplayBuffer(ReplayBuffer):
    """SB3 ReplayBuffer that also stores raw KPIs and relabels objectives when sampling."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        device: str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        relabel_fraction: float = 0.8,
        capacity_mbps: float = CellConfig().capacity_mbps,
        reward_version: RewardVersion = DEFAULT_REWARD_VERSION,
        seed: int = 0,
        urllc_base_ms: float | None = None,
        switch_cost: float = 0.0,
    ) -> None:
        if n_envs != 1 or optimize_memory_usage:
            raise ValueError("RelabelReplayBuffer supports one env and optimize_memory_usage=False")
        super().__init__(buffer_size, observation_space, action_space, device, n_envs,
                         optimize_memory_usage, handle_timeout_termination)
        self.kpis = np.zeros((self.buffer_size, KPI_COLUMNS), dtype=np.float64)
        self.relabel_fraction = relabel_fraction
        self.cfg = CellConfig(capacity_mbps=capacity_mbps)
        self.reward_version = reward_version
        self.urllc_base_ms = urllc_base_ms  # reward v3 only: the URLLC base latency the verifier sizes with
        self.switch_cost = switch_cost  # M09h: the env's switching cost, so a relabeled reward keeps it
        self.rng = np.random.default_rng(seed)

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
            done: np.ndarray, infos: list[dict[str, Any]]) -> None:
        self.kpis[self.pos] = kpis_from_info(infos[0])  # before super().add moves self.pos
        super().add(obs, next_obs, action, reward, done, infos)

    def relabel(self, batch_inds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """obs, next_obs, rewards for the batch, with a share of them relabeled."""
        obs = self.observations[batch_inds, 0, :].copy()
        next_obs = self.next_observations[batch_inds, 0, :].copy()
        rewards = self.rewards[batch_inds, 0].copy()
        chosen = self.rng.random(len(batch_inds)) < self.relabel_fraction
        n = int(chosen.sum())
        if n:
            weights, targets, bound = sample_objective_vectors(self.rng, n, self.cfg)
            vectors = objective_vectors(weights, targets, bound, self.cfg).astype(np.float32)
            obs[chosen, OBJECTIVE_SLICE] = vectors
            next_obs[chosen, OBJECTIVE_SLICE] = vectors  # same objective on both sides
            k = self.kpis[batch_inds[chosen]]
            rewards[chosen] = compute_reward_batch(weights, targets, bound, k[:, 0:3], k[:, 3:6], k[:, 6],
                                                   self.cfg.capacity_mbps, self.reward_version,
                                                   shares=k[:, 7:10], urllc_base_ms=self.urllc_base_ms,
                                                   switch_cost=self.switch_cost, previous_shares=k[:, 10:13])
        return obs, next_obs, rewards

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None) -> ReplayBufferSamples:
        obs, next_obs, rewards = self.relabel(batch_inds)
        # Only use dones that are not due to timeouts (same as the parent class).
        dones = self.dones[batch_inds, 0] * (1 - self.timeouts[batch_inds, 0])
        data = (obs, self.actions[batch_inds, 0, :], next_obs, dones.reshape(-1, 1), rewards.reshape(-1, 1))
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
