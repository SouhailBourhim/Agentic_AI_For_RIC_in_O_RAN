"""The trained policy as a plain `observation -> PRB shares` function.

This is the small interface the RL runner (M04a) will call every step. It hides
stable-baselines3: load a saved model once, then ask for shares. It always acts
deterministically (the actor's output, no exploration noise).
"""

from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG

from sim.cell import shares_from_action


class SlicePolicy:
    """A trained goal-conditioned DDPG actor."""

    def __init__(self, model: DDPG) -> None:
        self.model = model

    @classmethod
    def load(cls, path: str | Path) -> "SlicePolicy":
        return cls(DDPG.load(str(path), device="cpu"))

    def action(self, obs: np.ndarray) -> np.ndarray:
        """Raw actor output: 3 numbers in [0, 1]. The env divides them by their sum."""
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def shares(self, obs: np.ndarray) -> np.ndarray:
        """PRB shares that sum to 1, exactly as the env would apply them."""
        return shares_from_action(self.action(obs))
