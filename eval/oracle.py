"""The oracle ceiling of M09b: what a controller that knows everything could achieve on an instance.

Evaluation only, never deployed. Every second it:
1. knows the active intents' stated facts from the moment they're submitted (no parsing, no verifier, no agent);
2. knows the next 60 seconds' loads exactly (it replays the backend's random generator) and every queue, and on a
   cell with a radio channel (M10a) each slice's channel factor over those seconds (the backend's trace);
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

from eval.scenarios import Scenario, ScenarioIntent, facts
from rl.baselines import FormulaEstPolicy, estimated_rates
from rl.oracle import split_grid
from sim.cell import CellConfig, apply_min_share, step_slices_batch
from sim.env import KPI_FEATURES_PER_SLICE, LATENCY_SCALE_MS, OBS_SIZE, OBS_SIZE_WITH_APPLIED
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
           base: np.ndarray, cfg: CellConfig, previous: np.ndarray, t: float = 0.0,
           efficiency: np.ndarray | None = None) -> np.ndarray:
    """The split to apply at time t. loads: (seconds ahead, 3), the exact loads of the horizon, from t on;
    efficiency: the channel factors over the same seconds (M10a), None = 1.0 everywhere.
    Each candidate is held over the horizon; an intent counts only until its window ends."""
    n = len(candidates)
    met, unmet = np.zeros(n), np.zeros(n)
    queue = np.broadcast_to(backlog, (n, len(SLICES))).copy()
    for k, step_loads in enumerate(loads):
        factors = 1.0 if efficiency is None else efficiency[k]
        out = step_slices_batch(candidates, step_loads, queue, base, cfg, factors)
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
        channel = self.backend.channel
        efficiency = None if channel is None else np.array([channel.at(t + k * dt) for k in range(len(loads))])
        self.previous = choose(self.candidates, active, loads, backlog, self.base, self.cfg, self.previous, t,
                               efficiency)
        return self.previous


# ---------------------------------------------------------------- M10b: the oracle's search without the oracle's knowledge


def objective_intents(obs: np.ndarray, capacity_mbps: float, horizon_s: float) -> list[ScenarioIntent]:
    """The objective in the observation as pseudo-intents for `choose`: one per slice with a target (URLLC's carries
    the bound), one for a bound alone. One per slice, not one for all: the search then gets credit per target met,
    as the operator's intents are scored one by one. They last the whole horizon: the executor doesn't know when
    the objective expires."""
    k = KPI_FEATURES_PER_SLICE * len(SLICES)
    targets = obs[k + len(SLICES):k + 2 * len(SLICES)].astype(np.float64) * capacity_mbps
    bound = float(obs[k + 2 * len(SLICES)]) * LATENCY_SCALE_MS or None
    out = []
    for i, s in enumerate(SLICES):
        latency = bound if s == "urllc" else None
        if targets[i] > 0 or latency is not None:
            stated = {s: float(targets[i])} if targets[i] > 0 else {}
            out.append(ScenarioIntent(0.0, "", facts(horizon_s, max_latency_ms=latency, **stated), 0.0))
    return out


class MpcEstPolicy:
    """mpc_est (M10b): the oracle's search (`choose`: every split on the grid, held for 60 s, predicted with the cell
    model), fed what a controller can measure instead of the truth. It reads only the observation:
    - loads: the measured ones, held for the horizon; a bounded URLLC at its 30 s peak (the formula's rule, M09b);
    - each slice's rate per unit share: estimated_rates, the same estimate and margin as formula_est;
    - queues: from each slice's latency above its base, at the rate the slice had (latency is clipped at 1 s, so a
      queue past that is underestimated);
    - the objective in force (the verified one, with the verifier's implied targets), not the operator's intents.
    `model` is what a controller designer knows of the cell: the told capacity, the measured delay model and buffer
    (M09a), and the told URLLC base latency for every slice. Evaluation-side code, like the oracle: a non-learned
    candidate for M10c's comparator."""

    def __init__(self, model: CellConfig) -> None:
        self.model = model
        self.candidates = apply_min_share(split_grid(), MIN_SLICE_SHARE)
        self.base = np.array([model.base_latency_ms[s] for s in SLICES])
        self.previous = np.full(len(SLICES), 1.0 / len(SLICES))
        self.horizon = round(HORIZON_S / model.dt_s)

    def shares(self, obs: np.ndarray) -> np.ndarray:
        c, k = self.model.capacity_mbps, KPI_FEATURES_PER_SLICE
        loads = obs[0:k * len(SLICES):k].astype(np.float64) * c
        latency = obs[2:k * len(SLICES):k].astype(np.float64) * LATENCY_SCALE_MS
        share = obs[3:k * len(SLICES):k].astype(np.float64)
        rates = estimated_rates(obs)
        intents = objective_intents(obs, c, HORIZON_S)
        urllc = SLICES.index("urllc")
        if any(i.max_latency_ms is not None for i in intents):
            loads[urllc] = max(loads[urllc], float(obs[OBS_SIZE + urllc]) * c)  # LoadHistory's 30 s peak
        backlog = np.maximum(latency - self.base, 0.0) / 1000.0 * share * c * rates
        held_loads, held_rates = np.tile(loads, (self.horizon, 1)), np.tile(rates, (self.horizon, 1))
        self.previous = choose(self.candidates, intents, held_loads, backlog, self.base, self.model, self.previous,
                               0.0, held_rates)
        return self.previous


class FormulaToldPolicy:
    """formula_told (M10b, a reference only): formula_est's rules fed the truth of now instead of measurements,
    with no margin: the true load of the last second, the true rate of the coming one (channel x true capacity, as a
    fraction of the told one) and the true latency. What separates it from formula_est is the cost of estimating;
    what separates the oracle from it is foresight. Reads the simulated cell, like the oracle."""

    def __init__(self, backend, cfg: CellConfig) -> None:
        self.backend, self.cfg = backend, cfg
        self.inner = FormulaEstPolicy(cfg.told_urllc_base_latency_ms, estimate="latest", margin=0.0, volatility=0.0)

    def shares(self, obs: np.ndarray) -> np.ndarray:
        told, k = self.cfg.told_capacity_mbps, KPI_FEATURES_PER_SLICE
        factors = (self.backend.channel.at(self.backend._time_s) if self.backend.channel is not None
                   else np.ones(len(SLICES)))
        obs = obs.copy()
        for i, s in enumerate(SLICES):
            obs[k * i] = self.backend._loads[s] / told
            obs[k * i + 2] = min(self.backend._last[s].latency_ms / LATENCY_SCALE_MS, 10.0)
        obs[OBS_SIZE_WITH_APPLIED:OBS_SIZE_WITH_APPLIED + len(SLICES)] = factors * self.cfg.capacity_mbps / told
        return self.inner.shares(obs)

