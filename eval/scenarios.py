"""The three scripted scenarios of the control evaluation (M08a; M08c replays them on the testbed).

A scenario is a traffic pattern (sim/scenarios.py, by name) and the operator's intents with the
facts each one states. The facts are used three times: to check the parse step read the intent
correctly (H7: only intents with verified extraction count), to score every controller against what
the operator asked for, and to see whether the agent relaxed it.

The numbers follow from the verifier's rules, not from any result (tests/test_eval_control.py):
- a target can rise by at most 22.5 Mb/s in one change, so a target above the 25 Mb/s of an equal
  split is out of reach of one intent. What an equal split cannot do here is the latency bound;
- S3's second intent needs 49 % of the cell on top of the first one's 59 %: check `conflict`.
These texts are not in the held-out split, which stays unscored until M08b.
"""

from dataclasses import dataclass

from verifier.objective import SLICES

SEEDS = range(5000, 5010)  # paired: every controller meets the same traffic noise
DEV_SEED = 4999  # dev round only, never reported


def facts(duration_s: float, embb: float | None = None, urllc: float | None = None, be: float | None = None,
          max_latency_ms: float | None = None) -> dict:
    """A ParsedIntent as a dict: what the text states, nothing else."""
    return {"understood": True, "clarification": None, "priority": None,
            "min_thr_mbps": {"embb": embb, "urllc": urllc, "be": be},
            "max_latency_ms": max_latency_ms, "duration_s": duration_s}


@dataclass(frozen=True)
class ScenarioIntent:
    at_s: float  # backend time of the submission
    text: str
    expected: dict  # the stated facts
    reaction_from_s: float  # when the clock of the reaction time starts

    @property
    def window(self) -> tuple[float, float]:
        """The operator's window: from the submission, for the stated duration. The time the agent
        takes is inside it, on purpose: the operator is waiting."""
        return self.at_s, self.at_s + self.expected["duration_s"]

    @property
    def targets(self) -> dict[str, float]:
        return {s: self.expected["min_thr_mbps"][s] for s in SLICES if self.expected["min_thr_mbps"][s] is not None}

    @property
    def max_latency_ms(self) -> float | None:
        return self.expected["max_latency_ms"]


@dataclass(frozen=True)
class Scenario:
    name: str  # also the traffic's name in backends.sim.scenario_traffic
    title: str
    seconds: int
    intents: tuple[ScenarioIntent, ...]


SCENARIOS = {
    "s1": Scenario("s1", "Steady then spike", 360, (
        ScenarioIntent(60.0, "Garantis au moins 20 Mb/s au slice URLLC avec une latence sous 10 ms pendant 5 minutes.",
                       facts(300.0, urllc=20.0, max_latency_ms=10.0), reaction_from_s=120.0),)),  # the spike
    "s2": Scenario("s2", "New intent mid-run", 360, (
        ScenarioIntent(120.0, "URLLC must hold 18 Mb/s with latency below 5 ms for 3 minutes.",
                       facts(180.0, urllc=18.0, max_latency_ms=5.0), reaction_from_s=120.0),)),
    "s3": Scenario("s3", "Conflicting intents", 360, (
        ScenarioIntent(60.0, "Guarantee 22 Mb/s to eMBB and 22 Mb/s to best effort for 5 minutes.",
                       facts(300.0, embb=22.0, be=22.0), reaction_from_s=60.0),
        ScenarioIntent(120.0, "Garantis 22 Mb/s au slice URLLC avec une latence sous 5 ms pendant 4 minutes.",
                       facts(240.0, urllc=22.0, max_latency_ms=5.0), reaction_from_s=120.0),)),
}
