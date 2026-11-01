"""Named scenarios: a traffic set plus an objective, reused by the demo and the plots.

Keeping them here means the demo output, the figures and the learning note all describe
exactly the same situation. The evaluation scenarios of M08 will live in eval/.
"""

from sim.cell import CellConfig
from sim.traffic import TrafficPattern, default_traffic
from verifier.objective import RLObjective, SliceValues

SPIKE_START_S = 10.0
SPIKE_END_S = 20.0


def urllc_spike(
    capacity_mbps: float = CellConfig().capacity_mbps,
    spike_fraction: float = 0.40,
) -> tuple[dict[str, TrafficPattern], RLObjective]:
    """Steady eMBB (50 % of capacity) and BE (30 %); URLLC jumps from 10 % to `spike_fraction`
    of capacity between t=10 and t=20 s. With the 75 Mb/s cell: 37.5 / 22.5, URLLC 7.5 -> 30.

    The objective protects URLLC: weight 0.5, at least 10 % of capacity, latency at most 20 ms.
    """
    traffic = default_traffic(capacity_mbps)
    traffic["urllc"] = TrafficPattern(
        "spike",
        base_mbps=0.10 * capacity_mbps,
        peak_mbps=spike_fraction * capacity_mbps,
        start_s=SPIKE_START_S,
        end_s=SPIKE_END_S,
    )
    objective = RLObjective(
        weights=SliceValues(embb=0.3, urllc=0.5, be=0.2),
        min_thr_mbps=SliceValues(urllc=0.10 * capacity_mbps),
        max_latency_ms=20.0,
    )
    return traffic, objective


DEMO_SURGE_START_S = 120.0
DEMO_SURGE_END_S = 600.0


def demo_traffic(capacity_mbps: float = CellConfig().capacity_mbps) -> dict[str, TrafficPattern]:
    """The live demo (M04a): like urllc_spike, but URLLC surges from 10 % to 40 % of capacity
    between t=120 s and t=600 s, so an operator has time to submit and confirm an intent
    before and during the surge."""
    traffic, _ = urllc_spike(capacity_mbps)
    traffic["urllc"] = TrafficPattern(
        "spike",
        base_mbps=0.10 * capacity_mbps,
        peak_mbps=0.40 * capacity_mbps,
        start_s=DEMO_SURGE_START_S,
        end_s=DEMO_SURGE_END_S,
    )
    return traffic
