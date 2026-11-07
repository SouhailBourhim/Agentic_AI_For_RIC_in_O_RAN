"""The M08a scenarios on the real cell (M08d): same fractions of the capacity, in real time.

The simulator's scenarios are defined as fractions of the cell (sim/traffic.py). This cell's capacity is 79 Mb/s by KPM
(testbed/measure_capacity_m08d.py, rule in docs/DECISIONS.md), so the same situations here have other numbers, rounded
to 0.5 Mb/s: eMBB 50 % of the cell, BE 30 %, URLLC 10 % surging to 40 %; in S2 eMBB 50 %, URLLC 24 %, BE 20 %. Targets
scale too, and are whole numbers because the parser misreads decimals in French (M08c's dev round). Latency bounds
don't scale: they aren't a share of anything. M08c played these at 69 Mb/s (eval/testbed_scenarios_m08c.py).

Only intents with verified extraction are used (H7).

Each segment is (seconds, Mb/s per slice), played by iperf3 on the scenario's clock.
"""

from eval.scenarios import Scenario, ScenarioIntent, facts

TESTBED_CAPACITY_MBPS = 79.0  # also in xapps.kpm_monitor.cell_info: a test checks they agree
QUIET = {"embb": 39.5, "urllc": 8.0, "be": 23.5}
SURGE = {**QUIET, "urllc": 31.5}
STEADY = {"embb": 39.5, "urllc": 19.0, "be": 16.0}

SEGMENTS = {
    "s1": [(120, QUIET), (180, SURGE), (60, QUIET)],
    "s2": [(360, STEADY)],
    "s3": [(120, QUIET), (240, SURGE)],
}

SCENARIOS = {
    "s1": Scenario("s1", "Steady then spike", 360, (
        ScenarioIntent(60.0, "Garantis au moins 21 Mb/s au slice URLLC avec une latence sous 10 ms pendant 5 minutes.",
                       facts(300.0, urllc=21.0, max_latency_ms=10.0), reaction_from_s=120.0),)),  # updated from the log
    "s2": Scenario("s2", "New intent mid-run", 360, (
        ScenarioIntent(120.0, "URLLC must hold 19 Mb/s with latency below 5 ms for 3 minutes.",
                       facts(180.0, urllc=19.0, max_latency_ms=5.0), reaction_from_s=120.0),)),
    "s3": Scenario("s3", "Conflicting intents", 360, (
        ScenarioIntent(60.0, "Guarantee at least 23 Mb/s to eMBB and at least 23 Mb/s to best effort for 5 minutes.",
                       facts(300.0, embb=23.0, be=23.0), reaction_from_s=60.0),
        ScenarioIntent(120.0, "Garantis 23 Mb/s au slice URLLC avec une latence sous 5 ms pendant 4 minutes.",
                       facts(240.0, urllc=23.0, max_latency_ms=5.0), reaction_from_s=120.0),)),
}
