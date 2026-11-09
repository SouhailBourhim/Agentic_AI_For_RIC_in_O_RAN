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

M04a option:
- --min-share F: the per-slice PRB floor (sim.cell.apply_min_share) in the env after the filter,
  as the deployed runner applies it; the validation checkpoint is scored with the same floor.

M09c option:
- --m09 RECIPE: train in the M09c setting (rl/m09.py: calibrated cell, late KPIs, filter at the served
  alpha, switches and expiries, on/off loads, 120-step episodes) with one of its recipes (rl.m09.RECIPES),
  which sets the cell, the reward and the load history. --episode-steps, --reward and --filter-in-loop are
  then set by the recipe; the checkpoint is scored on the goal test's m09_validation split.
  A v9 recipe (M09h, rl/m09.py) also sets the switching cost, the applied split in the observation, OU noise,
  gamma and the residual wrapper; its checkpoint is scored on m10_validation and must also pass T1a.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from mlops import tracking
from rl import m09
from rl.policy import SMOOTHING_ALPHA, SlicePolicy
from rl.relabel import RelabelReplayBuffer
from rl.residual import RESIDUAL_MARK, ResidualEnv, ResidualPolicy
from sim.env import SlicingEnv

DEFAULT_GROUP = "m02"  # models/<group>/ and runs/<group>/
VALIDATION_START = 50_000  # first validation check (the policy is still warming up before)
TRACK_EVERY = 25_000  # how often the training reward goes to MLflow
COLLAPSE_SPREAD = 0.01  # M09h: an actor whose action barely varies over recent states is reported as collapsed
COLLAPSE_SAMPLE = 1_000

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
    noise: str = "normal",
    gamma: float | None = None,
    switch_cost: float = 0.0,
) -> DDPG | TD3:
    """Build a DDPG or TD3 model with our hyperparameters (TD3 keeps SB3's defaults for its
    three extra settings: policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5).
    relabel > 0 swaps in the objective-relabeling replay buffer. Also used by the tests.
    M09h: noise "ou" (rl.m09's OU parameters), a gamma other than HYPERPARAMETERS', and the env's switching
    cost, which the relabeling buffer must charge too."""
    hp = HYPERPARAMETERS
    buffer: dict = {}
    if relabel > 0:
        cell = env.unwrapped.base_cfg if hasattr(env.unwrapped, "base_cfg") else None
        told = {} if cell is None else {"capacity_mbps": cell.told_capacity_mbps,
                                        "urllc_base_ms": cell.told_urllc_base_latency_ms}
        buffer = {"replay_buffer_class": RelabelReplayBuffer,
                  "replay_buffer_kwargs": {"relabel_fraction": relabel, "reward_version": reward_version,
                                           "seed": seed, "switch_cost": switch_cost, **told}}
    n_actions = env.action_space.shape[0]
    if noise == "ou":  # SB3's own defaults (theta 0.15, dt 0.01) are close to a random walk over an episode
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=m09.OU_SIGMA * np.ones(n_actions),
                                                    theta=m09.OU_THETA, dt=m09.OU_DT)
    else:
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=hp["noise_sigma"] * np.ones(n_actions))
    return ALGORITHMS[algo](
        "MlpPolicy",
        env,
        gamma=hp["gamma"] if gamma is None else gamma,
        learning_rate=hp["learning_rate"],
        buffer_size=hp["buffer_size"],
        batch_size=hp["batch_size"],
        tau=hp["tau"],
        learning_starts=hp["learning_starts"],
        train_freq=hp["train_freq"],
        gradient_steps=hp["gradient_steps"],
        action_noise=action_noise,
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
    M09h (a v9 recipe): scored on m10_validation, and a checkpoint must also pass T1a (M09c's rule ignored it, and
    all four confirm seeds then failed on T1a). Each entry also records the actor's action spread over recent
    replay states; below COLLAPSE_SPREAD the seed is reported as collapsed (seed 40 in M09c). Reported only: a
    collapsed seed is not restarted, which would be selection by another name.
    """

    def __init__(self, every: int, path: Path, start: int = VALIDATION_START, min_share: float | None = None,
                 m09_recipe: str | None = None) -> None:
        super().__init__()
        from rl.goal_test import GoalTest  # imported here: building it computes the oracle once

        self.every = every
        self.path = path
        self.start = start
        # Scored with the same floor as in training; M09c on its own validation split, in its own cell.
        self.v9 = m09_recipe in m09.V9_RECIPES
        self.residual = m09_recipe is not None and m09.option(m09_recipe, "residual")
        split = "validation" if m09_recipe is None else ("m10_validation" if self.v9 else "m09_validation")
        self.test = GoalTest(split, min_share)
        self.best_key: tuple[float, float] | None = None
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps < self.start or self.num_timesteps % self.every:
            return True
        policy = (ResidualPolicy if self.residual else SlicePolicy)(self.model, self.test.alpha)
        m = self.test.evaluate(policy)
        m["step"] = self.num_timesteps
        if self.v9:
            m["t1a_required"] = True
            m["action_spread"] = self.action_spread()
            m["collapsed"] = m["action_spread"] < COLLAPSE_SPREAD
        self.history.append(m)
        for key in ("t1a_share_positive", "t1b_captured", "reward"):
            self.logger.record(f"validation/{key}", m[key])
        tracking.log_metrics({f"validation/{k}": float(m[k]) for k in ("t1a_share_positive", "t1b_captured",
                                                                        "reward")}, step=self.num_timesteps)
        key = (m["t1b_captured"], m["reward"])
        eligible = m["t2_pass"] and (m["t1a_pass"] or not self.v9)
        if eligible and (self.best_key is None or key > self.best_key):
            self.best_key = key
            self.model.save(self.path.with_name(f"{self.path.stem}_best.zip"))
        self.path.with_name(f"{self.path.stem}_validation.json").write_text(json.dumps(self.history, indent=2))
        return True

    def action_spread(self) -> float:
        """Std of the actor's raw action over the last COLLAPSE_SAMPLE stored states, averaged over the 3 outputs."""
        buffer = self.model.replay_buffer
        end = buffer.buffer_size if buffer.full else buffer.pos
        obs = buffer.observations[max(0, end - COLLAPSE_SAMPLE):end, 0, :]
        if not len(obs):
            return float("nan")
        actions, _ = self.model.predict(obs, deterministic=True)
        return float(np.asarray(actions).std(axis=0).mean())


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
    min_share: float | None = None,
    m09_recipe: str | None = None,
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

    options: dict = {"episode_steps": episode_steps, "objective_switch_prob": objective_switch_prob,
                     "action_filter_alpha": SMOOTHING_ALPHA if filter_in_loop else None}
    v9: dict = {}  # M09h's options of the recipe (their v8 values for a v8 recipe)
    if m09_recipe is not None:  # M09c: the recipe sets the cell, the reward and the history (rl/m09.py)
        recipe = m09.RECIPES[m09_recipe]
        v9 = {key: m09.option(m09_recipe, key) for key in m09.V9_DEFAULTS}
        reward_version, episode_steps, filter_in_loop = recipe["reward"], m09.EPISODE_STEPS, True
        options = {"episode_steps": episode_steps, "action_filter_alpha": m09.FILTER_ALPHA,
                   **m09.training_options(recipe["profile"], recipe["history"], v9["applied"], v9["switch_cost"])}
        objective_switch_prob = options["objective_switch_prob"]
    cell_env: gym.Env = SlicingEnv(
        load_scale=load_scale,
        surge_range=(1.5, surge_max),
        reward_version=reward_version,  # type: ignore[arg-type]
        min_share=min_share,
        **options,
    )
    if v9.get("residual"):
        if RESIDUAL_MARK not in name:  # the runner, harness and goal test recognise a residual model by its name
            raise ValueError(f"a residual recipe's model name must contain {RESIDUAL_MARK!r} (use --tag _{m09_recipe})")
        cell_env = ResidualEnv(cell_env)  # A7: the action corrects the formula's split (rl/residual.py)
    env = Monitor(cell_env, filename=str(run_dir / name / "monitor.csv"))
    model = make_model(env, seed, tensorboard_log=str(run_dir / "tb"), algo=algo, relabel=relabel,
                       reward_version=reward_version, noise=v9.get("noise", "normal"), gamma=v9.get("gamma"),
                       switch_cost=v9.get("switch_cost", 0.0))
    settings = {"algo": algo, "seed": seed, "timesteps": timesteps, "threads": threads,
                "load_scale": load_scale, "objective_switch_prob": objective_switch_prob,
                "relabel": relabel, "filter_in_loop": filter_in_loop, "surge_max": surge_max,
                "episode_steps": episode_steps, "reward_version": reward_version, "val_every": val_every,
                "min_share": min_share, "m09_recipe": m09_recipe,
                **({} if m09_recipe is None else {**m09.RECIPES[m09_recipe], "filter_alpha": m09.FILTER_ALPHA,
                                                  "kpi_delay_range": list(m09.KPI_DELAY_RANGE),
                                                  "expiry_share": m09.EXPIRY_SHARE, "toggle_prob": m09.TOGGLE_PROB}),
                **HYPERPARAMETERS,
                **({} if m09_recipe not in m09.V9_RECIPES else
                   {**v9, "ou": [m09.OU_THETA, m09.OU_DT, m09.OU_SIGMA] if v9["noise"] == "ou" else None})}
    callbacks: list[BaseCallback] = [TrackingCallback()]
    if val_every:
        callbacks.append(ValidationCheckpoint(val_every, model_path, min_share=min_share, m09_recipe=m09_recipe))
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
    parser.add_argument("--min-share", type=float, default=None, help="per-slice PRB floor in the env (M04a: 0.05)")
    parser.add_argument("--m09", choices=sorted(m09.RECIPES), default=None, help="M09c setting and recipe (rl/m09.py)")
    args = parser.parse_args()
    train(args.seed, args.timesteps, args.threads, args.algo, args.load_scale, args.objective_switch_prob,
          args.tag, args.relabel, args.filter_in_loop, args.surge_max, args.episode_steps, args.reward,
          args.val_every, args.group, track=not args.no_track, min_share=args.min_share, m09_recipe=args.m09)


if __name__ == "__main__":
    main()
