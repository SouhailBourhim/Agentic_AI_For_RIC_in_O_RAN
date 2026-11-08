"""Seeded scenario instances for the broad benchmark (M09b), in five families, each aimed at one claim.

  uv run python -m eval.generator f1 0          print one instance

The families and their ranges are frozen in docs/DECISIONS.md (M09b pre-registration); this file implements them.
An instance is 360 s of piecewise-constant traffic per slice (sim.traffic "steps", as iperf3 plays it on the
testbed) and one to three intents, each with its text and the facts it states. The same seed always gives the
same instance. C below is the capacity the verifier is told (79 Mb/s).

Why the families look like this: the verifier allows at most 0.3 C of change per slice target, and an equal
split gives each slice about a third of the cell. So a single throughput target is always met by an equal
split; static can only lose to a URLLC latency bound under load (F1a, F3) or to a target raised in two steps
(F1b). That is a property of our verifier, not of the generator.
"""

import argparse
import dataclasses
import json
from dataclasses import dataclass

import numpy as np

from eval.scenarios import Scenario, ScenarioIntent, facts
from sim.traffic import TrafficPattern
from verifier.objective import SLICES

C = 79.0
SECONDS = 360
FAMILIES = ("f1", "f2", "f3", "f4", "f5")
PER_FAMILY = 40
SPLITS = {"dev": 100_000, "test": 200_000}
BOUNDS = (5, 8, 10, 15, 20)
EASY_BOUNDS = (10, 15, 20, 30, 50)
BASE = {"embb": (0.42, 0.52), "urllc": (0.06, 0.12), "be": (0.22, 0.32)}
NAMES = {"en": {"embb": "eMBB", "urllc": "URLLC", "be": "best effort"},
         "fr": {"embb": "eMBB", "urllc": "URLLC", "be": "best effort"}}


@dataclass(frozen=True)
class Instance:
    family: str
    index: int
    split: str
    seed: int
    steps: dict  # slice -> ((start_s, Mb/s), ...)
    intents: tuple[ScenarioIntent, ...]
    variant: str = ""  # f1: "a" or "b"; f4: "a" or "b"

    @property
    def name(self) -> str:
        return f"{self.family}_{self.index:02d}"

    def scenario(self) -> Scenario:
        return Scenario(self.name, f"{self.family}{self.variant}", SECONDS, self.intents)

    def traffic(self, noise_std: float) -> dict[str, TrafficPattern]:
        return {s: TrafficPattern("steps", steps=self.steps[s], noise_std=noise_std) for s in SLICES}


def seed_of(family: str, index: int, split: str) -> int:
    return SPLITS[split] + 1000 * FAMILIES.index(family) + index


def half(x: float) -> float:
    return round(x * 2) / 2


def text(rng: np.random.Generator, targets: dict[str, int], minutes: int, bound: int | None) -> str:
    """One intent's wording, English or French at random. Whole numbers only: the parser misreads French decimals."""
    lang = "en" if rng.random() < 0.5 else "fr"
    names = NAMES[lang]
    parts = [f"at least {t} Mb/s to {names[s]}" if lang == "en" else f"au moins {t} Mb/s au slice {names[s]}" for s, t in targets.items()]
    joined = " and ".join(parts) if lang == "en" else " et ".join(parts)
    latency = ""
    if bound is not None:
        latency = f" with latency below {bound} ms" if lang == "en" else f" avec une latence sous {bound} ms"
    if lang == "en":
        return f"Guarantee {joined}{latency} for {minutes} minutes."
    return f"Garantis {joined}{latency} pendant {minutes} minutes."


def intent(rng: np.random.Generator, at_s: float, targets: dict[str, int], bound: int | None,
           reaction_from_s: float | None = None) -> ScenarioIntent:
    minutes = int(min(5, (SECONDS - at_s) // 60))
    stated = facts(60.0 * minutes, max_latency_ms=float(bound) if bound is not None else None,
                   **{s: float(t) for s, t in targets.items()})
    return ScenarioIntent(float(at_s), text(rng, targets, minutes, bound), stated,
                          reaction_from_s=float(reaction_from_s if reaction_from_s is not None else at_s))


def base_loads(rng: np.random.Generator) -> dict[str, float]:
    return {s: half(rng.uniform(*BASE[s]) * C) for s in SLICES}


def constant(loads: dict[str, float]) -> dict[str, tuple]:
    return {s: ((0.0, loads[s]),) for s in SLICES}


def tens(rng: np.random.Generator, low: int, high: int) -> float:
    return float(rng.choice(np.arange(low, high + 1, 10)))


def family_f1(rng: np.random.Generator, index: int) -> tuple[str, dict, tuple]:
    loads = base_loads(rng)
    if index < PER_FAMILY // 2:  # (a, 20) URLLC surge with a latency bound
        start = tens(rng, 90, 150)
        peak = half(rng.uniform(0.40, 0.55) * C)
        if rng.random() < 0.5:
            urllc = ((0.0, loads["urllc"]), (start, peak))
        else:  # a ramp, played as 5 s steps
            ramp = float(rng.integers(30, 61))
            n = int(ramp // 5)
            urllc = ((0.0, loads["urllc"]),) + tuple((start + 5 * k, half(loads["urllc"] + (peak - loads["urllc"]) * (k + 1) / n)) for k in range(n))
        steps = {**constant(loads), "urllc": urllc}
        at = tens(rng, 40, int(start))
        return "a", steps, (intent(rng, at, {"urllc": int(rng.integers(12, 24))}, int(rng.choice(BOUNDS)), reaction_from_s=max(at, start)),)
    s = "embb" if rng.random() < 0.5 else "be"  # (b, 20) a target raised above a third, in two steps
    loads[s] = half(rng.uniform(0.42, 0.55) * C)
    at1 = tens(rng, 30, 60)
    at2 = at1 + tens(rng, 60, 120)
    t1 = int(rng.integers(15, 23))
    t2 = int(rng.integers(27, 32))
    return "b", constant(loads), (intent(rng, at1, {s: t1}, None), intent(rng, at2, {s: t2}, None))


def family_f2(rng: np.random.Generator) -> tuple[str, dict, tuple]:
    loads = base_loads(rng)
    scale = rng.uniform(0.95, 1.10) * C / sum(loads.values())
    loads = {s: half(v * scale) for s, v in loads.items()}
    s = "embb" if rng.random() < 0.5 else "be"
    target = int(rng.integers(15, int(min(23, loads[s] - 3)) + 1))
    return "", constant(loads), (intent(rng, tens(rng, 30, 120), {s: target}, None),)


def family_f3(rng: np.random.Generator) -> tuple[str, dict, tuple]:
    loads = base_loads(rng)
    urllc, t, high = [], 0.0, False
    while t < SECONDS:
        level = rng.uniform(0.40, 0.50) if high else rng.uniform(0.06, 0.12)
        urllc.append((t, half(level * C)))
        t += float(rng.integers(10, 31))
        high = not high
    steps = {**constant(loads), "urllc": tuple(urllc)}
    return "", steps, (intent(rng, tens(rng, 30, 90), {"urllc": int(rng.integers(10, 21))}, int(rng.choice(BOUNDS))),)


def family_f4(rng: np.random.Generator, index: int) -> tuple[str, dict, tuple]:
    loads = base_loads(rng)
    if index >= 30:  # (b) above the policy's range
        s = SLICES[int(rng.integers(0, 3))]
        loads[s] = max(loads[s], half(rng.uniform(0.50, 0.60) * C))
        return "b", constant(loads), (intent(rng, tens(rng, 30, 120), {s: int(rng.integers(35, 46))}, None),)
    at1 = tens(rng, 30, 60)
    at2 = at1 + tens(rng, 60, 90)
    x, y = int(rng.integers(18, 24)), int(rng.integers(18, 24))
    intents = [intent(rng, at1, {"embb": x, "be": y}, None),
               intent(rng, at2, {"urllc": int(rng.integers(15, 24))}, int(rng.choice((5, 8, 10))))]
    if index % 2 == 1 and at2 + 60 <= SECONDS - 120:
        intents.append(intent(rng, at2 + 60, {"embb": int(rng.integers(x + 4, 32))}, None))
    loads["urllc"] = max(loads["urllc"], half(0.30 * C))  # URLLC's goal must mean something
    return "a", constant(loads), tuple(intents)


def family_f5(rng: np.random.Generator) -> tuple[str, dict, tuple]:
    loads = base_loads(rng)
    scale = rng.uniform(0.70, 0.85) * C / sum(loads.values())
    loads = {s: half(v * scale) for s, v in loads.items()}
    s = SLICES[int(rng.integers(0, 3))]
    bound = None
    if s == "urllc":  # the target first, then a load that makes it mean something, still at most 0.25 C
        bound = int(rng.choice(EASY_BOUNDS))
        target = int(rng.integers(8, int(0.25 * C)))
        loads["urllc"] = min(max(loads["urllc"], target + 1.0), half(0.25 * C))
    else:
        target = int(rng.integers(8, int(min(20, loads[s] - 1)) + 1))
    return "", constant(loads), (intent(rng, tens(rng, 30, 120), {s: target}, bound),)


def generate(family: str, index: int, split: str = "dev") -> Instance:
    if split == "test":
        raise SystemExit("the test seeds belong to M09d and are refused before it (docs/DECISIONS.md)")
    return draw(family, index, split)


def draw(family: str, index: int, split: str) -> Instance:
    """No guard: for M09d's own runner and for the tests that check the families' constraints."""
    seed = seed_of(family, index, split)
    rng = np.random.default_rng(seed)
    builders = {"f2": family_f2, "f3": family_f3, "f5": family_f5}
    if family in ("f1", "f4"):
        variant, steps, intents = (family_f1 if family == "f1" else family_f4)(rng, index)
    else:
        variant, steps, intents = builders[family](rng)
    return Instance(family, index, split, seed, steps, intents, variant)


def dev_instances(families=FAMILIES, per_family: int = PER_FAMILY) -> list[Instance]:
    return [generate(f, i) for f in families for i in range(per_family)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    inst = generate(args.family, args.index)
    print(json.dumps({"name": inst.name, "variant": inst.variant, "seed": inst.seed, "steps": inst.steps,
                      "intents": [dataclasses.asdict(i) for i in inst.intents]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
