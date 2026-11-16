"""FaultyBackend: the simulated cell with faults at chosen seconds, for the M11a tests and figure.

It lets the runner's safe state be exercised without a testbed: stale KPIs and a dead actuator are what
FlexRICBackend.health() reports when KPM stops or the slice-control xApp dies, and the exception stands for a store
that can't be read. The cell itself is untouched: during a "stale" second the KPIs are in fact fresh, only health()
says otherwise, which is all the runner looks at.

apply_prb_split is not overridden: only rl/runner.py may call it (invariant 4, tests/test_runner_realtime.py), so
a failing apply is injected by a subclass in tests/test_safe_state.py, where the check does not look.
"""

from backends.base import BackendHealth
from backends.sim import SimBackend, scenario_traffic
from sim.cell import CellConfig
from store.kpi import KpiRecord

STALE_AGE_S = 60.0  # the report age health() gives during a stale second: far above the runner's threshold


class FaultyBackend(SimBackend):
    """Faults are sets of whole seconds of the cell's clock, checked when the call is made."""

    def __init__(self, scenario: str = "steady", seed: int = 0, stale=(), down=(), read_error=()) -> None:
        cfg = CellConfig()
        super().__init__(cfg, scenario_traffic(scenario, cfg), seed=seed, speed=0.0)
        self.stale, self.down, self.read_error = set(stale), set(down), set(read_error)

    def second(self) -> int:
        return int(round(self._time_s))

    def health(self) -> BackendHealth:
        return BackendHealth(kpi_age_s=STALE_AGE_S if self.second() in self.stale else 0.0,
                             actuator_ok=self.second() not in self.down)

    def read_kpis(self) -> list[KpiRecord]:
        if self.second() in self.read_error:
            raise ConnectionError("store unreachable")
        return super().read_kpis()
