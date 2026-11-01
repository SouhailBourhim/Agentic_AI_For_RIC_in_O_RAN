"""Train the goal-conditioned DDPG (or TD3) policy on the simulator.

Run with:  uv run python -m rl.train --seed 0 [--algo td3] [--timesteps 200000]
           [--load-scale 1.2 --objective-switch-prob 0.02 --tag v2]
M02b recipes are launched by rl/sweep.py, which adds --relabel, --filter-in-loop,
--surge-max, --episode-steps, --reward, --val-every and --group.

Every episode, SlicingEnv.reset() samples a new random objective and random traffic, so the
policy learns to follow whatever objective appears in its observation (goal-conditioned).

Outputs (all outside git):
- models/m02/<algo><tag>_seed<N>.zip    the trained model (actor + critic)
- models/m02/<algo><tag>_seed<N>.json   the exact settings used
- runs/m02/<algo><tag>_seed<N>/monitor.csv   one line per episode: reward, length, time
- runs/m02/tb/                     TensorBoard logs (uv run tensorboard --logdir runs/m02/tb)
Every hyperparameter is explained in docs/learn/M02-ddpg.md.

TD3 is DDPG plus three stability fixes (twin critics, delayed actor updates, target policy
smoothing). In stable-baselines3, DDPG is literally TD3 with those three switched off, so the
two share every hyperparameter below. We added TD3 after DDPG's goal-conditioning check
failed on one seed (see the note).

M02b training options (each answers a finding of docs/learn/M02-ddpg.md, part M02b):
- --relabel F: objective relabeling of a fraction F of every replay batch (rl/relabel.py);
- --filter-in-loop: the smoothing filter is applied inside the env, as when deployed;
- --surge-max, --load-scale, --episode-steps: training traffic covering the operating envelope;
- --reward v1|v2: the reward version (v2 = uncapped log latency penalty, the default);
- --val-every N: every N steps, score the policy on the validation cases of rl/goal_test.py
  and keep the best checkpoint as <name>_best.zip (rule in ValidationCheckpoint).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from mlops import tracking
from rl.policy import SMOOTHING_ALPHA, SlicePolicy
from rl.relabel import RelabelReplayBuffer
from sim.env import SlicingEnv

DEFAULT_GROUP = "m02"  # models/<group>/ and runs/<group>/
VALIDATION_START = 50_000  # first validation check (the policy is still warming up before)
TRACK_EVERY = 25_000  # how often the training reward goes to MLflow

HYPERPARAMETERS = {
    "gamma": 0.95,  # ~20-step horizon: an allocation's effect shows within seconds
    "learning_rate": 1e-3,
    "buffer_size": 200_000,  # holds the whole run
    "batch_size": 256,
    "tau": 0.005,  # how fast the target networks follow the trained ones
    "learning_starts": 5_000,  # random actions first, to fill the replay buffer
    "train_freq": 1,  # one gradient step per environment step
    "gradient_steps": 1,
    "net_arch": [256, 256],  # same size for actor and critic
    # Gaussian exploration noise. SB3 adds it in its internal [-1, 1] action scale, so 0.1
    # there is 0.05 in share units ([0, 1]).
    "noise_sigma": 0.1,
}

ALGORITHMS = {"ddpg": DDPG, "td3": TD3}


def make_model(
    env: SlicingEnv,
    seed: int,
    tensorboard_log: str | None = None,
    algo: str = "ddpg",
    relabel: float = 0.0,
    reward_version: str = "v2",
) -> DDPG | TD3:
    """Build a DDPG or TD3 model with our hyperparameters (TD3 keeps SB3's defaults for its
    three extra settings: policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5).
    relabel > 0 swaps in the objective-relabeling replay buffer. Also used by the tests."""
    hp = HYPERPARAMETERS
    buffer: dict = {}
    if relabel > 0:
        buffer = {"replay_buffer_class": RelabelReplayBuffer,
                  "replay_buffer_kwargs": {"relabel_fraction": relabel, "reward_version": reward_version,
                                           "seed": seed}}
    n_actions = env.action_space.shape[0]
    noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=hp["noise_sigma"] * np.ones(n_actions))
    return ALGORITHMS[algo](
        "MlpPolicy",
        env,
        gamma=hp["gamma"],
        learning_rate=hp["learning_rate"],
        buffer_size=hp["buffer_size"],
        batch_size=hp["batch_size"],
        tau=hp["tau"],
        learning_starts=hp["learning_starts"],
        train_freq=hp["train_freq"],
        gradient_steps=hp["gradient_steps"],
        action_noise=noise,
        policy_kwargs={"net_arch": hp["net_arch"]},
        tensorboard_log=tensorboard_log,
        seed=seed,
        device="cpu",
        verbose=0,
        **buffer,
    )


class ValidationCheckpoint(BaseCallback):
    """Every `every` steps, score the policy on the validation cases; keep the best one.

    The policy is scored as deployed (SlicePolicy, with the smoothing filter) on the
    validation split of rl/goal_test.py, never on the held-out test. Rule, fixed before
    the M02b runs: among checkpoints whose validation reward beats the equal split, keep
    the highest T1b (captured share of the oracle's advantage), then the highest reward.
    """

    def __init__(self, every: int, path: Path, start: int = VALIDATION_START) -> None:
        super().__init__()
        from rl.goal_test import GoalTest  # imported here: building it computes the oracle once

        self.every = every
        self.path = path
        self.start = start
        self.test = GoalTest("validation")
        self.best_key: tuple[float, float] | None = None
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps < self.start or self.num_timesteps % self.every:
            return True
        m = self.test.evaluate(SlicePolicy(self.model))
        m["step"] = self.num_timesteps
        self.history.append(m)
        for key in ("t1a_share_positive", "t1b_captured", "reward"):
            self.logger.record(f"validation/{key}", m[key])
        tracking.log_metrics({f"validation/{k}": float(m[k]) for k in ("t1a_share_positive", "t1b_captured",
                                                                        "reward")}, step=self.num_timesteps)
        key = (m["t1b_captured"], m["reward"])
        if m["t2_pass"] and (self.best_key is None or key > self.best_key):
            self.best_key = key
            self.model.save(self.path.with_name(f"{self.path.stem}_best.zip"))
        self.path.with_name(f"{self.path.stem}_validation.json").write_text(json.dumps(self.history, indent=2))
        return True


class TrackingCallback(BaseCallback):
    """Every `every` steps, send the recent training reward to MLflow (no-op without a run).

    It only reads SB3's buffer of the last 100 finished episodes; it changes nothing.
    """

    def __init__(self, every: int | None = None) -> None:
        super().__init__()
        self.every = every or TRACK_EVERY

    def _on_step(self) -> bool:
        episodes = self.model.ep_info_buffer
        if self.num_timesteps % self.every == 0 and episodes:
            per_step = float(np.mean([e["r"] / e["l"] for e in episodes]))
            tracking.log_metrics({"train/reward_per_step_last100": per_step}, step=self.num_timesteps)
        return True


def train(
    seed: int,
    timesteps: int,
    threads: int,
    algo: str = "ddpg",
    load_scale: float = 1.0,
    objective_switch_prob: float = 0.0,
    tag: str = "",
    relabel: float = 0.0,
    filter_in_loop: bool = False,
    surge_max: float = 3.0,
    episode_steps: int = 200,
    reward_version: str = "v2",
    val_every: int = 0,
    group: str = DEFAULT_GROUP,
    track: bool = True,
) -> Path:
    """Train one seed and save the model; returns the model path.

    With `track`, the run is recorded in MLflow (mlops/tracking.py): settings, git commit,
    training and validation curves, and the files below as artifacts.
    """
    torch.set_num_threads(threads)  # several seeds run side by side on the 32 threads
    name = f"{algo}{tag}_seed{seed}"
    model_dir, run_dir = Path("models") / group, Path("runs") / group
    (run_dir / name).mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{name}.zip"

    env = Monitor(
        SlicingEnv(
            episode_steps=episode_steps,
            load_scale=load_scale,
            objective_switch_prob=objective_switch_prob,
            surge_range=(1.5, surge_max),
            action_filter_alpha=SMOOTHING_ALPHA if filter_in_loop else None,
            reward_version=reward_version,  # type: ignore[arg-type]
        ),
        filename=str(run_dir / name / "monitor.csv"),
    )
    model = make_model(env, seed, tensorboard_log=str(run_dir / "tb"), algo=algo,
                       relabel=relabel, reward_version=reward_version)
    settings = {"algo": algo, "seed": seed, "timesteps": timesteps, "threads": threads,
                "load_scale": load_scale, "objective_switch_prob": objective_switch_prob,
                "relabel": relabel, "filter_in_loop": filter_in_loop, "surge_max": surge_max,
                "episode_steps": episode_steps, "reward_version": reward_version, "val_every": val_every,
                **HYPERPARAMETERS}
    callbacks: list[BaseCallback] = [TrackingCallback()]
    if val_every:
        callbacks.append(ValidationCheckpoint(val_every, model_path))
    with tracking.training_run(name, group, settings, enabled=track):
        model.learn(total_timesteps=timesteps, tb_log_name=name, progress_bar=False,
                    callback=CallbackList(callbacks))
        model.save(model_path)
        model_path.with_suffix(".json").write_text(json.dumps(settings, indent=2))
        stem = model_path.with_suffix("")
        tracking.log_artifacts([model_path, model_path.with_suffix(".json"), run_dir / name / "monitor.csv",
                                Path(f"{stem}_best.zip"), Path(f"{stem}_validation.json")], subdir="files")
    print(f"saved {model_path}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default="ddpg")
    parser.add_argument("--load-scale", type=float, default=1.0, help="multiply sampled training traffic")
    parser.add_argument("--objective-switch-prob", type=float, default=0.0,
                        help="chance per step of a new random objective mid-episode")
    parser.add_argument("--tag", default="", help="added to the model name, e.g. v2 -> td3v2_seed0")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads (2 was as fast as 4 here)")
    parser.add_argument("--relabel", type=float, default=0.0, help="share of each batch relabeled (M02b)")
    parser.add_argument("--filter-in-loop", action="store_true", help="smoothing filter inside the env")
    parser.add_argument("--surge-max", type=float, default=3.0, help="max peak/base factor of sampled surges")
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--reward", choices=["v1", "v2"], default="v2")
    parser.add_argument("--val-every", type=int, default=0, help="validation checkpoint period (0 = off)")
    parser.add_argument("--group", default=DEFAULT_GROUP, help="models/<group>/ and runs/<group>/")
    parser.add_argument("--no-track", action="store_true", help="do not record the run in MLflow")
    args = parser.parse_args()
    train(args.seed, args.timesteps, args.threads, args.algo, args.load_scale, args.objective_switch_prob,
          args.tag, args.relabel, args.filter_in_loop, args.surge_max, args.episode_steps, args.reward,
          args.val_every, args.group, track=not args.no_track)


if __name__ == "__main__":
    main()
