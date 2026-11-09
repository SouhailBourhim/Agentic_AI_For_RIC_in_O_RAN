"""The residual arm of v9 (M09h, A7): the RL policy corrects the formula's split instead of choosing one from scratch.

M09b and M09g found LLM + formula ahead of LLM + RL on dev, mostly through *what split it asks for*. A residual
policy starts from that split and learns a bounded correction:

    split = normalise(max(0, formula(obs) + c)),   c = 0.2 * (2 * action - 1)   (each slice moved by at most 0.2)

then the usual filter and per-slice floor. The actor still sees the full observation (objective included), so it
stays goal-conditioned, and the Runner still sets the split (invariant 4): only what stands where the policy stands
changes. With the actor at 0.5 everywhere, the split is the formula's.

The same function builds the split in training (ResidualEnv, around SlicingEnv) and when served (ResidualPolicy),
and both feed the formula the observation the actor sees (late KPIs included), once per step. A residual model is
recognised by RESIDUAL_MARK in its file name (td3_v9_residual_seed50.zip): SlicePolicy.load then returns a
ResidualPolicy, so the runner, the harness, the goal test and the registry load it unchanged.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from rl.baselines import FormulaPolicy
from rl.policy import SMOOTHING_ALPHA, SlicePolicy
from sim.cell import smooth_split

RESIDUAL_BOUND = 0.2  # the largest correction per slice, as a share of the cell
RESIDUAL_MARK = "_residual"


def told_base_ms() -> float:
    """The URLLC base latency the formula sizes with: the calibrated cell's told value, as in M09b's formula."""
    from rl import m09  # here: rl.m09 imports rl.runner, which imports rl.policy

    return m09.cell("m09").told_urllc_base_latency_ms


def residual_shares(base: np.ndarray, action: np.ndarray, bound: float = RESIDUAL_BOUND) -> np.ndarray:
    """The formula's split plus the actor's correction, as shares summing to 1 (the formula's split if all vanish)."""
    correction = bound * (2.0 * np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0) - 1.0)
    shares = np.maximum(np.asarray(base, dtype=np.float64) + correction, 0.0)
    total = shares.sum()
    return shares / total if total > 1e-9 else np.asarray(base, dtype=np.float64) / np.sum(base)


class ResidualEnv(gym.Wrapper):
    """SlicingEnv whose action is a correction to the formula's split. Training only; the env's filter and floor
    then apply to the corrected split, exactly as ResidualPolicy's filter and the runner's floor when served."""

    def __init__(self, env: gym.Env, base_ms: float | None = None) -> None:
        super().__init__(env)
        self.base_ms = told_base_ms() if base_ms is None else base_ms
        self.formula = FormulaPolicy(self.base_ms)
        self.last_obs: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self.formula = FormulaPolicy(self.base_ms)  # its 30 s URLLC peak starts empty, as at a runner start
        self.last_obs = obs
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        shares = residual_shares(self.formula.shares(self.last_obs), action)
        obs, reward, terminated, truncated, info = self.env.step(shares.astype(np.float32))
        self.last_obs = obs
        return obs, reward, terminated, truncated, info


class ResidualPolicy(SlicePolicy):
    """A trained residual actor served as a controller: formula + correction, then the smoothing filter."""

    def __init__(self, model: Any, alpha: float = SMOOTHING_ALPHA, base_ms: float | None = None) -> None:
        super().__init__(model, alpha)
        self.base_ms = told_base_ms() if base_ms is None else base_ms
        self.formula = FormulaPolicy(self.base_ms)

    @classmethod
    def load(cls, path: str | Path, alpha: float = SMOOTHING_ALPHA) -> "ResidualPolicy":
        return super().load(path, alpha)  # type: ignore[return-value]

    def reset(self) -> None:
        super().reset()
        self.formula = FormulaPolicy(self.base_ms)

    def action(self, obs: np.ndarray) -> np.ndarray:
        target = residual_shares(self.formula.shares(obs), self.raw_action(obs))
        self._previous = smooth_split(self._previous, target, self.alpha)
        return self._previous.astype(np.float32)

    __call__ = action
