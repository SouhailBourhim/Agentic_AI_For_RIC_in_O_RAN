"""SimBackend: the M01 simulated cell behind the Backend interface.

It advances the cell with exactly the formulas and the random-draw order of SlicingEnv.step
(a test checks both give the same KPIs for the same splits), so the policy meets the cell it
was trained on. The clock is simulated: `speed` says how fast it runs against the wall clock.
- speed = 1: real time, for the live demo (one tick per second);
- speed = 10: ten times faster;
- speed = 0: as fast as the CPU allows, for tests and evaluation runs.
"""

import time
from collections import deque

import numpy as np

from sim.cell import CellConfig, SliceStep, step_slice
from sim.scenarios import demo_traffic, s1_spike_traffic, s2_steady_traffic, urllc_spike
from sim.traffic import TrafficPattern, default_traffic
from store.kpi import CellInfo, KpiRecord
from verifier.objective import SLICES

BITS_PER_BYTE = 8


def scenario_traffic(name: str, cfg: CellConfig) -> dict[str, TrafficPattern]:
    """Traffic of a named scenario: 'steady', 'urllc_spike' (spike at 10-20 s), 'demo' (URLLC surge
    at 120-600 s, slow enough for an operator), or the M08 evaluation's 's1', 's2', 's3'."""
    if name == "steady":
        return default_traffic(cfg.capacity_mbps)
    if name == "urllc_spike":
        return urllc_spike(cfg.capacity_mbps)[0]
    if name in ("demo", "s3"):  # S3 of M08 plays the demo's surge
        return demo_traffic(cfg.capacity_mbps)
    if name == "s1":
        return s1_spike_traffic(cfg.capacity_mbps)
    if name == "s2":
        return s2_steady_traffic(cfg.capacity_mbps)
    raise ValueError(f"unknown scenario {name!r}; choose steady, urllc_spike, demo, s1, s2 or s3")


class SimBackend:
    """One simulated cell with three slices."""

    def __init__(
        self,
        cfg: CellConfig | None = None,
        traffic: dict[str, TrafficPattern] | None = None,
        seed: int = 0,
        speed: float = 0.0,
        cell_id: str = "sim-cell-1",
    ) -> None:
        self.cfg = cfg or CellConfig()
        self.traffic = traffic or default_traffic(self.cfg.capacity_mbps)
        self.rng = np.random.default_rng(seed)  # same generator type and seeding as gymnasium
        self.speed = speed
        self.cell_id = cell_id
        self._time_s = 0.0
        self._shares = np.full(len(SLICES), 1.0 / len(SLICES))
        # Before the first tick nothing has been sent, as in SlicingEnv.reset.
        self._loads = {s: self.traffic[s].mean_load_at(0.0) for s in SLICES}
        self._last = {s: SliceStep(0.0, 0.0, 0.0, 0.0, self.cfg.base_latency_ms[s]) for s in SLICES}
        self._next_wall: float | None = None  # wall-clock deadline of the next tick (speed > 0)

    def info(self) -> CellInfo:
        return CellInfo(
            cell_id=self.cell_id,
            source="sim",
            slices=SLICES,
            n_prb=self.cfg.n_prb,
            capacity_mbps=self.cfg.told_capacity_mbps,  # what the verifier and the policy are told
            urllc_base_latency_ms=self.cfg.told_urllc_base_latency_ms,
        )

    def read_kpis(self) -> list[KpiRecord]:
        return [
            KpiRecord(
                ts=self._time_s,
                source="sim",
                cell_id=self.cell_id,
                slice=s,
                dl_thr_mbps=self._last[s].throughput_mbps,
                prb_share_alloc=float(self._shares[i]),
                prb_used=self._last[s].prb_used,
                n_ue=1,
                latency_ms=self._last[s].latency_ms,
                offered_load_mbps=self._loads[s],
                buffer_bytes=self._last[s].backlog_mb * 1e6 / BITS_PER_BYTE,
            )
            for i, s in enumerate(SLICES)
        ]

    def apply_prb_split(self, shares: list[float]) -> None:
        """Refuse anything that is not 3 non-negative shares summing to 1: a backend must
        never guess what a malformed split meant."""
        values = np.asarray(shares, dtype=np.float64)
        # isfinite first: NaN fails every comparison, so "< 0" alone would let it through.
        if (values.shape != (len(SLICES),) or not np.all(np.isfinite(values)) or np.any(values < 0)
                or abs(values.sum() - 1.0) > 1e-6):
            raise ValueError(f"a PRB split is {len(SLICES)} non-negative shares summing to 1, got {shares}")
        self._shares = values

    def now(self) -> float:
        return self._time_s

    def sleep(self, seconds: float) -> None:
        """Advance the cell by seconds / dt ticks, pacing against the wall clock if speed > 0.

        Pacing uses a deadline that moves by dt / speed per tick, so the delays of slow ticks
        do not add up over a long run.
        """
        for _ in range(round(seconds / self.cfg.dt_s)):
            self._tick()
            if self.speed > 0:
                if self._next_wall is None:
                    self._next_wall = time.monotonic()
                self._next_wall += self.cfg.dt_s / self.speed
                time.sleep(max(0.0, self._next_wall - time.monotonic()))

    def _tick(self) -> None:
        """One dt of the cell: same steps, same order of random draws as SlicingEnv.step."""
        self._loads = {s: self.traffic[s].load_at(self._time_s, self.rng) for s in SLICES}
        self._last = {
            s: step_slice(
                share=float(self._shares[i]),
                load_mbps=self._loads[s],
                backlog_mb=self._last[s].backlog_mb,
                base_latency_ms=self.cfg.base_latency_ms[s],
                cfg=self.cfg,
            )
            for i, s in enumerate(SLICES)
        }
        self._time_s += self.cfg.dt_s


class DelayedBackend(SimBackend):
    """The simulated cell, but read_kpis returns the records of `delay` ticks ago: the late view the
    testbed gives the policy (M07b). A subclass, so the split still goes straight to the cell (only
    the runner calls apply_prb_split, invariant 4)."""

    def __init__(self, cfg: CellConfig, traffic: dict[str, TrafficPattern], seed: int, delay: int) -> None:
        super().__init__(cfg, traffic, seed=seed, speed=0.0)
        self.delay = delay
        self.history: deque = deque(maxlen=delay + 1)

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        self.history.append(super().read_kpis())

    def read_kpis(self) -> list[KpiRecord]:
        if not self.history:
            return super().read_kpis()
        return self.history[0]  # the oldest kept: `delay` ticks before the newest
