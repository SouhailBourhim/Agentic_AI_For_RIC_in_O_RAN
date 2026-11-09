"""M09g's diagnostic D3: what seed 40's actor looks like inside (M09c's collapse), against healthy seeds.

  uv run python -m rl.diag_saturation

The actor ends in a tanh: action = (tanh(z) + 1) / 2, then the split is action / sum(action) (sim.cell.shares_from_action).
Two things follow:
- Adding the same amount to all three z (the direction (1,1,1)) changes the split little or not at all, so the critic
  gives almost no gradient along it and nothing holds the outputs back from drifting there.
- Once |z| > 3, tanh's slope is below 1 % of its maximum: the gradient that should move the actor back is gone.
  A saturated actor outputs the same action whatever the input (seed 40: [1, 1, 0] at 400k steps).
This script feeds the same observations to several final actors and reports, per actor: the share of observations with
all three |z| > 3, the mean tanh slope, the spread of the action across observations, and z split into its component
along (1,1,1) (the mean of the three, which the split can't see) and the rest (what decides the split).
Observations: the m09_validation case set (dev, already used for checkpoint selection), rolled out with seed 41's final
actor in M09c's setting. The reading rule is in docs/DECISIONS.md (2026-11-09).
"""

import csv
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from stable_baselines3 import TD3

from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from rl.goal_test import case_set, rollout, setting_of
from rl.policy import SlicePolicy
from sim.plot import SURFACE, TEXT, TEXT_MUTED

SPLIT = "m09_validation"
OBSERVER = Path("models/m09c/confirm/td3_v8_nooverserve_seed41.zip")  # a healthy actor of the same round as seed 40
COLLAPSED = Path("models/m09c/confirm/td3_v8_nooverserve_seed40.zip")
V7 = Path("var/policy_cache/td3_noreward_floor_seed20_best.zip")  # the served champion, for reference
SATURATED_Z = 3.0  # tanh'(3) = 0.0099: below 1 % of the slope at 0
OUT_CSV = Path("docs/learn/data/M09g-saturation.csv")
# The rule (DECISIONS): A3 is supported if seed 40's actor is saturated on >= 99 % of the observations with an action
# spread below 0.01, while seed 41's is saturated on < 10 % of them.
RULE_COLLAPSED_SHARE, RULE_SPREAD, RULE_HEALTHY_SHARE = 0.99, 0.01, 0.10


def finals() -> list[Path]:
    """Every final actor of M09c (dev and confirm rounds), then v7: the checkpoints are best-of, the finals show drift."""
    found = sorted(p for p in Path("models/m09c").glob("*/*_seed*.zip") if not p.stem.endswith("_best"))
    return [*found, V7]


def observations() -> np.ndarray:
    """Every observation seed 41's actor sees on the split's (scenario, objective) pairs, in M09c's setting."""
    cases, setting = case_set(SPLIT), setting_of(SPLIT)
    policy = SlicePolicy.load(OBSERVER, setting.alpha)
    seen: list[np.ndarray] = []

    def act(_env, obs: np.ndarray) -> np.ndarray:
        seen.append(np.asarray(obs, dtype=np.float32).copy())
        return policy(obs)

    for traffic in cases.scenarios.values():
        for objective in cases.objectives.values():
            rollout(act, traffic, objective, policy.reset, 0.05, setting)
    return np.stack(seen)


def pre_tanh(model: TD3, obs: np.ndarray) -> np.ndarray:
    """z, the actor's last layer before its tanh, for each observation (the actor's own input width: v7 reads 22)."""
    n = int(model.observation_space.shape[0])
    with torch.no_grad():
        z = model.actor.mu[:-1](torch.as_tensor(obs[:, :n])).numpy().astype(np.float64)
    action, _ = model.predict(obs[:, :n], deterministic=True)
    assert np.allclose((np.tanh(z) + 1) / 2, action, atol=1e-5), "the actor is not linear layers then a tanh"
    return z


def describe(name: str, z: np.ndarray) -> dict:
    along = z.mean(axis=1)  # the (1,1,1) component, per output: what the split can't see
    rest = np.linalg.norm(z - along[:, None], axis=1)  # what decides the split
    action = (np.tanh(z) + 1) / 2
    return {"model": name, "observations": len(z),
            "saturated_share": round(float(np.mean((np.abs(z) > SATURATED_Z).all(axis=1))), 4),
            "median_abs_z": round(float(np.median(np.abs(z))), 2),
            "tanh_slope": round(float(np.mean(1 - np.tanh(z) ** 2)), 4),
            "action_spread": round(float(action.std(axis=0).mean()), 4),
            "mean_action": " ".join(f"{a:.2f}" for a in action.mean(axis=0)),
            "along_111_median": round(float(np.median(along)), 2), "rest_median": round(float(np.median(rest)), 2)}


def verdict(rows: list[dict]) -> str:
    get = {r["model"]: r for r in rows}
    bad, good = get[COLLAPSED.stem], get[OBSERVER.stem]
    holds = (bad["saturated_share"] >= RULE_COLLAPSED_SHARE and bad["action_spread"] < RULE_SPREAD
             and good["saturated_share"] < RULE_HEALTHY_SHARE)
    return (f"seed 40: saturated on {bad['saturated_share']:.1%}, spread {bad['action_spread']:.4f}, slope "
            f"{bad['tanh_slope']:.4f}; seed 41: saturated on {good['saturated_share']:.1%} -> "
            f"{'SUPPORTS A3: the action regulariser enters v9' if holds else 'does not support A3: the regulariser stays out of v9'}")


def plot(zs: dict[str, np.ndarray]) -> Figure:
    """Per actor, each observation as a dot: the (1,1,1) component against the rest. Saturation is far from 0 on both."""
    names = [COLLAPSED.stem, OBSERVER.stem, V7.stem]
    fig = Figure(figsize=(11, 3.8), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, len(names), sharex=True, sharey=True)
    for ax, name in zip(axes, names):
        z = zs[name]
        along = z.mean(axis=1)
        rest = np.linalg.norm(z - along[:, None], axis=1)
        ax.scatter(along, rest, s=4, color=ACCENT if name == COLLAPSED.stem else NEUTRAL, alpha=0.35, linewidths=0)
        ax.axvspan(-SATURATED_Z, SATURATED_Z, color=TEXT_MUTED, alpha=0.06, linewidth=0)
        ax.set_xlabel("mean of z: the (1,1,1) part, unseen by the split", color=TEXT_MUTED, fontsize=8.5)
        saturated = np.mean((np.abs(z) > SATURATED_Z).all(axis=1))
        titled(ax, f"{name.replace('td3_', '')}: all |z| > 3 on {saturated:.0%}")
    axes[0].set_ylabel("|z - mean|: the part that sets the split", color=TEXT_MUTED, fontsize=8.5)
    fig.suptitle("D3: the actor's pre-tanh output on the same observations (shaded: |mean| < 3)", color=TEXT,
                 fontsize=10.5, x=0.01, ha="left")
    return fig


def main() -> None:
    obs = observations()
    print(f"{len(obs)} observations from {SPLIT}, rolled out with {OBSERVER.stem}")
    rows, zs = [], {}
    for path in finals():
        zs[path.stem] = pre_tanh(TD3.load(str(path), device="cpu"), obs)
        rows.append(describe(path.stem, zs[path.stem]))
        print(" ".join(f"{k} {v}" for k, v in rows[-1].items()))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    IMG.mkdir(parents=True, exist_ok=True)
    plot(zs).savefig(IMG / "M09g-saturation.png", dpi=150)
    print(verdict(rows))


if __name__ == "__main__":
    main()
