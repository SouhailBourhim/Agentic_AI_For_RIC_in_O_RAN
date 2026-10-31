"""Named scenarios: a traffic set plus an objective, reused by the demo and the plots.

Keeping them here means the demo output, the figures and the learning note all describe
exactly the same situation. The evaluation scenarios of M08 will live in eval/.
"""

from sim.traffic import TrafficPattern, default_traffic
from verifier.objective import RLObjective, SliceValues

SPIKE_START_S = 10.0
SPIKE_END_S = 20.0


def urllc_spike() -> tuple[dict[str, TrafficPattern], RLObjective]:
    """Steady eMBB (50) and BE (30); URLLC jumps from 10 to 40 Mb/s between t=10 and t=20 s.

    The objective protects URLLC: weight 0.5, at least 10 Mb/s, latency at most 20 ms.
    """
    traffic = default_traffic()
    traffic["urllc"] = TrafficPattern(
        "spike", base_mbps=10.0, peak_mbps=40.0, start_s=SPIKE_START_S, end_s=SPIKE_END_S
    )
    objective = RLObjective(
        weights=SliceValues(embb=0.3, urllc=0.5, be=0.2),
        min_thr_mbps=SliceValues(urllc=10.0),
        max_latency_ms=20.0,
    )
    return traffic, objective
