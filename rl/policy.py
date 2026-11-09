"""The trained policy as a plain `observation -> PRB shares` controller.

This is the small interface the RL runner (M04a) will call every step. It hides
stable-baselines3: load a saved model once, then ask for an action. It always acts
deterministically (the actor's output, no exploration noise).

Output smoothing. The trained actors jitter: they move 13-17 % of the cell's PRBs every
second, flipping between two splits, and nothing in the reward charges for that. We pass the
actor's output through an exponential filter:

    split_t = alpha * shares(actor(obs_t)) + (1 - alpha) * split_{t-1},   alpha = 0.25

It is a low-pass filter with a time constant of about 1/alpha = 4 steps (4 s). Measured on
the 100 evaluation episodes, it cut the movement to ~2 % per second and did not lower the
reward (slightly raised it). A real RAN also cannot follow a split that flips every second.
Call reset() at the start of every episode so the filter does not carry over old actions.
The filter works on PRB shares (summing to 1) and is the function sim.cell.smooth_split, the
same one the environment applies during M02b training (SlicingEnv(action_filter_alpha=...)).
Rounds 1-2 of M02 smoothed the raw actor output instead; the difference is small (the env
divides by the sum anyway) but numbers re-run after M02b can differ slightly from the record.
"""

from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG, TD3

from sim.cell import shares_from_action, smooth_split

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

    @property
    def n_inputs(self) -> int | None:
        """How many observation values the actor was trained on: 22, or 25 with the load history (M09c).
        None (all of them) for a stand-in model with no observation space, as in some tests."""
        space = getattr(self.model, "observation_space", None)
        return None if space is None else int(space.shape[0])

    def raw_action(self, obs: np.ndarray) -> np.ndarray:
        """The actor's output, unfiltered: 3 numbers in [0, 1].

        The runner builds the full observation (with the load history, M09c); a policy trained
        without the history gets only the first 22 values, the layout it was trained on (v7)."""
        action, _ = self.model.predict(np.asarray(obs)[..., : self.n_inputs], deterministic=True)
        return np.asarray(action, dtype=np.float64)

    def action(self, obs: np.ndarray) -> np.ndarray:
        """Smoothed PRB shares: 3 numbers in [0, 1] summing to 1.

        A weighted average of two splits that sum to 1 also sums to 1, so the filter never
        produces an invalid action.
        """
        self._previous = smooth_split(self._previous, shares_from_action(self.raw_action(obs)), self.alpha)
        return self._previous.astype(np.float32)

    __call__ = action  # a SlicePolicy can be passed anywhere a controller function is expected

    def shares(self, obs: np.ndarray) -> np.ndarray:
        """PRB shares that sum to 1, exactly as the env would apply them (advances the filter)."""
        return shares_from_action(self.action(obs))
