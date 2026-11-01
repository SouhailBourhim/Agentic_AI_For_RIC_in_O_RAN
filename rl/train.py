"""Train the goal-conditioned DDPG (or TD3) policy on the simulator.

Run with:  uv run python -m rl.train --seed 0 [--algo td3] [--timesteps 200000]

Every episode, SlicingEnv.reset() samples a new random objective and random traffic, so the
policy learns to follow whatever objective appears in its observation (goal-conditioned).

Outputs (all outside git):
- models/m02/<algo>_seed<N>.zip    the trained model (actor + critic)
- models/m02/<algo>_seed<N>.json   the exact settings used
- runs/m02/<algo>_seed<N>/monitor.csv   one line per episode: reward, length, time
- runs/m02/tb/                     TensorBoard logs (uv run tensorboard --logdir runs/m02/tb)
Every hyperparameter is explained in docs/learn/M02-ddpg.md.

TD3 is DDPG plus three stability fixes (twin critics, delayed actor updates, target policy
smoothing). In stable-baselines3, DDPG is literally TD3 with those three switched off, so the
two share every hyperparameter below. We added TD3 after DDPG's goal-conditioning check
failed on one seed (see the note).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from sim.env import SlicingEnv

MODEL_DIR = Path("models/m02")
RUN_DIR = Path("runs/m02")

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
    env: SlicingEnv, seed: int, tensorboard_log: str | None = None, algo: str = "ddpg"
) -> DDPG | TD3:
    """Build a DDPG or TD3 model with our hyperparameters (TD3 keeps SB3's defaults for its
    three extra settings: policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5).
    Also used by the tests."""
    hp = HYPERPARAMETERS
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
    )


def train(seed: int, timesteps: int, threads: int, algo: str = "ddpg") -> Path:
    """Train one seed and save the model; returns the model path."""
    torch.set_num_threads(threads)  # several seeds run side by side on the 32 threads
    name = f"{algo}_seed{seed}"
    run_dir = RUN_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    env = Monitor(SlicingEnv(), filename=str(run_dir / "monitor.csv"))
    model = make_model(env, seed, tensorboard_log=str(RUN_DIR / "tb"), algo=algo)
    model.learn(total_timesteps=timesteps, tb_log_name=name, progress_bar=False)

    model_path = MODEL_DIR / f"{name}.zip"
    model.save(model_path)
    settings = {"algo": algo, "seed": seed, "timesteps": timesteps, "threads": threads, **HYPERPARAMETERS}
    model_path.with_suffix(".json").write_text(json.dumps(settings, indent=2))
    print(f"saved {model_path}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default="ddpg")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads (2 was as fast as 4 here)")
    args = parser.parse_args()
    train(args.seed, args.timesteps, args.threads, args.algo)


if __name__ == "__main__":
    main()
