"""The Backend interface: what the RL runner needs from a RAN, simulated or real.

Two implementations: SimBackend (backends/sim.py, the M01 simulator) and FlexRICBackend (M07b,
the OAI testbed). Everything above the runner (agent, verifier, console, monitoring) only sees
the store, never a Backend, so swapping the backend changes nothing up there (CLAUDE.md).

apply_prb_split is the only call that changes the radio. Only the RL runner may call it
(hard invariant 4); tests/test_runner_realtime.py checks no other module does.
"""

from typing import Protocol

from store.kpi import CellInfo, KpiRecord

__all__ = ["Backend", "CellInfo", "KpiRecord"]


class Backend(Protocol):
    """A typing.Protocol: any class with these methods is a Backend, no inheritance needed."""

    def info(self) -> CellInfo:
        """Fixed facts: cell id, slices, PRB budget, nominal capacity."""
        ...

    def read_kpis(self) -> list[KpiRecord]:
        """The latest record of each slice (what happened during the last tick)."""
        ...

    def apply_prb_split(self, shares: list[float]) -> None:
        """Set the PRB share of each slice (SLICES order, summing to 1) from now on."""
        ...

    def now(self) -> float:
        """The backend clock, in seconds. Simulated time for the sim, real time on the testbed."""
        ...

    def sleep(self, seconds: float) -> None:
        """Let `seconds` of backend time pass. The sim advances its cell (possibly faster than
        real time); the testbed just waits."""
        ...
