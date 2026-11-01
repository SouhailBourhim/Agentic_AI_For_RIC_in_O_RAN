"""The trained policy as a plain `observation -> PRB shares` controller.

This is the small interface the RL runner (M04a) will call every step. It hides
stable-baselines3: load a saved model once, then ask for an action. It always acts
deterministically (the actor's output, no exploration noise).

Output smoothing. The trained actors jitter: they move 13-17 % of the cell's PRBs every
second, flipping between two splits, and nothing in the reward charges for that. We pass the
actor's output through an exponential filter:

    a_t = alpha * actor(obs_t) + (1 - alpha) * a_{t-1},   alpha = 0.25

It is a low-pass filter with a time constant of about 1/alpha = 4 steps (4 s). Measured on
the 100 evaluation episodes, it cut the movement to ~2 % per second and did not lower the
reward (slightly raised it). A real RAN also cannot follow a split that flips every second.
Call reset() at the start of every episode so the filter does not carry over old actions.
"""

from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG, TD3

from sim.cell import shares_from_action

SMOOTHING_ALPHA = 0.25  # weight of the newest actor output; 1.0 turns the filter off


class SlicePolicy:
    """A trained goal-conditioned actor (DDPG or TD3), with output smoothing."""

    def __init__(self, model: DDPG | TD3, alpha: float = SMOOTHING_ALPHA) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.model = model
        self.alpha = alpha
        self._previous: np.ndarray | None = None

    @classmethod
    def load(cls, path: str | Path, alpha: float = SMOOTHING_ALPHA) -> "SlicePolicy":
        """The file name says which algorithm saved it: ddpg_seed0.zip or td3_seed0.zip."""
        algo = TD3 if Path(path).name.startswith("td3") else DDPG
        return cls(algo.load(str(path), device="cpu"), alpha)

    def reset(self) -> None:
        """Forget the previous action (start of an episode, or after a restart)."""
        self._previous = None

    def raw_action(self, obs: np.ndarray) -> np.ndarray:
        """The actor's output, unfiltered: 3 numbers in [0, 1]."""
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    def action(self, obs: np.ndarray) -> np.ndarray:
        """Smoothed action: 3 numbers in [0, 1]. The env divides them by their sum.

        A weighted average of two vectors in [0, 1] stays in [0, 1], so the filter never
        produces an invalid action.
        """
        raw = self.raw_action(obs)
        if self._previous is None:
            self._previous = raw
        else:
            self._previous = self.alpha * raw + (1.0 - self.alpha) * self._previous
        return self._previous.astype(np.float32)

    __call__ = action  # a SlicePolicy can be passed anywhere a controller function is expected

    def shares(self, obs: np.ndarray) -> np.ndarray:
        """PRB shares that sum to 1, exactly as the env would apply them (advances the filter)."""
        return shares_from_action(self.action(obs))
