"""The scenarios as M08c played them, kept for its figures: M08d replaced them (eval/testbed_scenarios.py).

The M08a scenarios on the real cell (M08c): same fractions of the capacity, in real time.

The simulator's scenarios are defined as fractions of the cell (sim/traffic.py). The testbed carries 69 Mb/s
(M07b: 0.654 Mb/s per PRB) where the simulator carries 75, so the same situations here have smaller numbers,
rounded to 0.5 Mb/s: eMBB 50 % of the cell, BE 30 %, URLLC 10 % surging to 40 %; in S2 eMBB 50 %, URLLC 24 %, BE
20 %. Targets scale too. Latency bounds don't: they aren't a share of anything.

Keeping the simulator's own numbers would have broken S3. At 69 Mb/s a target can rise by at most 20.7 Mb/s at
once (the verifier's step check), so 22 Mb/s would be rejected for a reason unrelated to the conflict.

Only intents with verified extraction are used (H7). Two first wordings failed on the real model in the dev round and
were replaced: "18,5 Mb/s" and "18.5 Mb/s" in French came back as 18500, and "Guarantee 20 Mb/s to eMBB and 20 Mb/s to
best effort" got a question. S1's target is therefore 18 Mb/s, within rounding of 18.4.

Each segment is (seconds, Mb/s per slice), played by iperf3 as in M07b.
"""

from eval.scenarios import Scenario, ScenarioIntent, facts

TESTBED_CAPACITY_MBPS = 69.0  # what M08c gave the verifier
QUIET = {"embb": 34.5, "urllc": 7.0, "be": 20.5}
SURGE = {**QUIET, "urllc": 27.5}
STEADY = {"embb": 34.5, "urllc": 16.5, "be": 14.0}

SEGMENTS = {
    "s1": [(120, QUIET), (180, SURGE), (60, QUIET)],
    "s2": [(360, STEADY)],
    "s3": [(120, QUIET), (240, SURGE)],
}

SCENARIOS = {
    "s1": Scenario("s1", "Steady then spike", 360, (
        ScenarioIntent(60.0, "Garantis au moins 18 Mb/s au slice URLLC avec une latence sous 10 ms pendant 5 minutes.",
                       facts(300.0, urllc=18.0, max_latency_ms=10.0), reaction_from_s=120.0),)),  # updated from the log
    "s2": Scenario("s2", "New intent mid-run", 360, (
        ScenarioIntent(120.0, "URLLC must hold 16.5 Mb/s with latency below 5 ms for 3 minutes.",
                       facts(180.0, urllc=16.5, max_latency_ms=5.0), reaction_from_s=120.0),)),
    "s3": Scenario("s3", "Conflicting intents", 360, (
        ScenarioIntent(60.0, "Guarantee at least 20 Mb/s to eMBB and at least 20 Mb/s to best effort for 5 minutes.",
                       facts(300.0, embb=20.0, be=20.0), reaction_from_s=60.0),
        ScenarioIntent(120.0, "Garantis 20 Mb/s au slice URLLC avec une latence sous 5 ms pendant 4 minutes.",
                       facts(240.0, urllc=20.0, max_latency_ms=5.0), reaction_from_s=120.0),)),
}
