"""The oracle ceiling of M09b: what a controller that knows everything could achieve on an instance.

Evaluation only, never deployed. Every second it:
1. knows the active intents' stated facts from the moment they're submitted (no parsing, no verifier, no agent);
2. knows the next second's loads exactly (it replays the backend's random generator) and every queue;
3. tries every PRB split on a 0.025 grid (rl.oracle.split_grid), each with the per-slice floor applied as the
   Runner applies it, and predicts the next second with the cell model (sim.cell.step_slices_batch);
4. applies the split that meets the most active intents, then the one leaving the least demand unmet, then the
   one closest to the current split. No smoothing and no KPI delay: an ideal actuator, so this is a ceiling.

"Met" is eval.metrics' rule: each target slice carries min(target, its load), URLLC stays under the bound. Greedy
(one second ahead), so it doesn't plan how to drain a queue over several seconds; the RL policy can beat it there.
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
           base: np.ndarray, cfg: CellConfig, previous: np.ndarray) -> np.ndarray:
    out = step_slices_batch(candidates, loads, backlog, base, cfg)
    met = intents_met(intents, loads, out)
    unmet = np.maximum(loads - out["throughput_mbps"], 0.0).sum(axis=1)
    distance = np.abs(candidates - previous).sum(axis=1)
    order = np.lexsort((distance, unmet, -met))  # last key first: most met, then least unmet, then closest
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
        rng = copy.deepcopy(self.backend.rng)  # the next second's loads, exactly as the backend will draw them
        loads = np.array([self.backend.traffic[s].load_at(t, rng) for s in SLICES])
        backlog = np.array([self.backend._last[s].backlog_mb for s in SLICES])
        active = [i for i in self.scenario.intents if i.window[0] <= t < i.window[1]]
        self.previous = choose(self.candidates, active, loads, backlog, self.base, self.cfg, self.previous)
        return self.previous
