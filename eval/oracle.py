"""The oracle ceiling of M09b: what a controller that knows everything could achieve on an instance.

Evaluation only, never deployed. Every second it:
1. knows the active intents' stated facts from the moment they're submitted (no parsing, no verifier, no agent);
2. knows the next 60 seconds' loads exactly (it replays the backend's random generator) and every queue;
3. tries every PRB split on a 0.025 grid (rl.oracle.split_grid), each with the per-slice floor applied as the
   Runner applies it, holds it for those 60 seconds and predicts them with the cell model
   (sim.cell.step_slices_batch);
4. applies the split that meets the active intents for the most seconds of the horizon, then the one leaving the
   least demand unmet over it, then the one closest to the current split; and decides again the next second.
   No smoothing and no KPI delay: an ideal actuator.

Why a horizon (DECISIONS 2026-11-08, after the dev benchmark): the first version looked one second ahead. Once a URLLC queue had built
up, no split could bring it under the bound within a second, every split met nothing, and the tie-break served
eMBB: the queue stayed at the buffer limit and other controllers beat the "ceiling" on 253 dev intents. Holding a
split for 60 s sees that draining the queue meets the bound later. It's a receding horizon over constant splits,
not an exact optimum; eval.benchmark counts the intents where a controller still beats it.
It knows an intent only once it's submitted, as every controller does.
"Met" is eval.metrics' rule: each target slice carries min(target, its load), URLLC stays under the bound.
It runs through the harness like every controller (eval.harness, controller "oracle"), so eval.metrics scores it
like any other run and the Runner applies its splits.
"""

import copy

import numpy as np

from eval.scenarios import Scenario, ScenarioIntent
from rl.oracle import split_grid
from sim.cell import CellConfig, apply_min_share, step_slices_batch
from verifier.objective import MIN_SLICE_SHARE, SLICES

TOLERANCE = 1e-9  # as eval.metrics.THROUGHPUT_TOLERANCE
HORIZON_S = 60  # a full buffer (408 Mb) drains in 60 s at 7 Mb/s above the load (DECISIONS)


def intents_met(intents: list[ScenarioIntent], loads: np.ndarray, out: dict[str, np.ndarray]) -> np.ndarray:
    """How many of these intents each candidate split meets next second (eval.metrics.met_mask, vectorised)."""
    count = np.zeros(out["throughput_mbps"].shape[0])
    for intent in intents:
        ok = np.ones(len(count), dtype=bool)
        for s, target in intent.targets.items():
            i = SLICES.index(s)
            ok &= out["throughput_mbps"][:, i] >= min(target, loads[i]) - TOLERANCE
        if intent.max_latency_ms is not None:
            ok &= out["latency_ms"][:, SLICES.index("urllc")] <= intent.max_latency_ms
        count += ok
    return count


def choose(candidates: np.ndarray, intents: list[ScenarioIntent], loads: np.ndarray, backlog: np.ndarray,
           base: np.ndarray, cfg: CellConfig, previous: np.ndarray, t: float = 0.0) -> np.ndarray:
    """The split to apply at time t. loads: (seconds ahead, 3), the exact loads of the horizon, from t on.
    Each candidate is held over the horizon; an intent counts only until its window ends."""
    n = len(candidates)
    met, unmet = np.zeros(n), np.zeros(n)
    queue = np.broadcast_to(backlog, (n, len(SLICES))).copy()
    for k, step_loads in enumerate(loads):
        out = step_slices_batch(candidates, step_loads, queue, base, cfg)
        live = [i for i in intents if t + k * cfg.dt_s < i.window[1]]
        met += intents_met(live, step_loads, out)
        unmet += np.maximum(step_loads - out["throughput_mbps"], 0.0).sum(axis=1)
        queue = out["backlog_mb"]
    distance = np.abs(candidates - previous).sum(axis=1)
    order = np.lexsort((distance, unmet, -met))  # last key first: most intent-seconds met, then least unmet, then closest
    return candidates[order[0]]


class OraclePolicy:
    """Stands where the RL policy stands, like rl.baselines' controllers, so the Runner stays the only caller of
    apply_prb_split (hard invariant 4). Unlike them it ignores the observation and reads the simulated cell itself:
    the backend's queues and random generator, and the scenario's intents. Only possible on SimBackend."""

    def __init__(self, backend, scenario: Scenario, cfg: CellConfig) -> None:
        self.backend, self.scenario, self.cfg = backend, scenario, cfg
        self.candidates = apply_min_share(split_grid(), MIN_SLICE_SHARE)
        self.base = np.array([cfg.base_latency_ms[s] for s in SLICES])
        self.previous = np.full(len(SLICES), 1.0 / len(SLICES))

    def shares(self, _obs: np.ndarray) -> np.ndarray:
        t = self.backend._time_s  # the cell's own clock, from 0
        rng = copy.deepcopy(self.backend.rng)  # the coming loads, exactly as the backend will draw them
        dt = self.cfg.dt_s
        loads = np.array([[self.backend.traffic[s].load_at(t + k * dt, rng) for s in SLICES]  # the backend's order
                          for k in range(round(HORIZON_S / dt))])
        backlog = np.array([self.backend._last[s].backlog_mb for s in SLICES])
        active = [i for i in self.scenario.intents if i.window[0] <= t < i.window[1]]
        self.previous = choose(self.candidates, active, loads, backlog, self.base, self.cfg, self.previous, t)
        return self.previous
