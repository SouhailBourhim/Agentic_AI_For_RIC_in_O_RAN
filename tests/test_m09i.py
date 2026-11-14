"""M09i: the sample, the paired comparison, the parse summary and the energy integral (eval/m09i.py). No network."""

import collections

import numpy as np
import pytest

from agent import llm as L
from eval import m09i as I
from eval.benchmark import dropped


def test_the_sample_is_fixed_dev_only_and_balanced() -> None:
    first, again = I.sample(), I.sample()
    assert [i.name for i in first] == [i.name for i in again]
    assert collections.Counter(i.family for i in first) == {f: I.PER_FAMILY for f in I.FAMILIES}
    assert not {i.name for i in first} & dropped()
    assert all(i.seed < 200_000 for i in first)  # dev seeds: the test seeds stay closed
    assert [i.name for i in I.sample(1)] == [i.name for i in first if i.name in {i.name for i in I.sample(1)}]
    assert collections.Counter(i.family for i in I.sample(1)) == {f: 1 for f in I.FAMILIES}


def test_the_remote_model_is_the_clients_default(monkeypatch) -> None:
    assert I.MODELS[I.REMOTE] == ("nim", L.NIM_MODEL)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    assert isinstance(I.client(I.REMOTE), L.NimClient) and isinstance(I.client(I.LOCAL), L.OllamaClient)


def row(model: str, instance: str, met: float, controller: str = "proposed") -> dict:
    return {"model": model, "controller": controller, "instance": instance, "intent_met": met}


def test_remote_minus_local_is_paired_by_instance_and_intent() -> None:
    rows = [row(I.LOCAL, "a", 0.5), row(I.LOCAL, "a", 1.0), row(I.REMOTE, "a", 1.0), row(I.REMOTE, "a", 1.0),
            row(I.LOCAL, "b", 0.2), row(I.REMOTE, "b", 0.4),
            row(I.LOCAL, "c", 1.0)]  # c has no remote run: left out, not guessed
    diff = I.paired(rows, "proposed")
    assert diff["instances"] == 2
    assert diff["mean_diff"] == pytest.approx(np.mean([0.5, 0.0, 0.2]), abs=1e-4)  # rounded to 4 decimals
    assert diff["low_95"] <= diff["mean_diff"] <= diff["high_95"]
    assert I.paired(rows, "llm_only_v2") == {}


def test_the_parse_summary_pools_the_repeats_with_a_wilson_interval() -> None:
    ex = [{"model": I.LOCAL, "split": "tune", "n": 20, "correct": c, "latency_median_s": 1.0, "calls": 20,
           "prompt_tokens": 1000, "completion_tokens": 100} for c in (20, 19)]
    [parse] = [r for r in I.summary(ex, []) if r["part"] == "parse"]
    assert parse["value"] == pytest.approx(39 / 40) and parse["low_95"] < 0.975 < parse["high_95"] <= 1.0
    assert parse["tokens_per_call"] == pytest.approx(55.0)


def test_energy_is_the_integral_of_power_over_time() -> None:
    meter = I.GpuEnergy()
    meter.samples = [(0.0, 100.0), (1800.0, 100.0), (3600.0, 300.0)]  # 100 W for 30 min, then a ramp to 300 W
    assert meter.wh() == pytest.approx(50.0 + 100.0)
    meter.samples = [(0.0, 50.0)]
    assert meter.wh() == 0.0


def event(kind: str, t: float | None, iid: str = "i0001", **data) -> dict:
    return {"event": kind, "ts_backend": t, "intent_id": iid, "data": data}


def test_the_decision_delay_runs_from_arrival_to_the_first_change() -> None:
    events = [event("received", 30.0, text="A"), event("received", 40.0, "i0002", text="B"),
              event("applied", 49.0), event("applied", 70.0), event("shares_applied", 44.0, "i0002")]
    assert I.decision_s(events, "A") == 19.0 and I.decision_s(events, "B") == 4.0
    assert I.decision_s(events, "C") == ""  # never received: blank, not 0


def test_the_applied_objective_is_compared_in_one_canonical_form() -> None:
    def act(args: dict, accepted: bool = True) -> dict:
        return event("agent_step", 32.0, node="act", tool="set_rl_objective", arguments=args,
                     verifier={"accepted": accepted})
    local = [event("received", 30.0, text="A"), act({"duration_s": 300, "min_thr_mbps": {"embb": 16}})]
    remote = [event("received", 30.0, text="A"), act({"duration_s": "999"}, accepted=False),
              act({"duration_s": "300.0", "max_latency_ms": None, "min_thr_mbps": {"embb": 16.0, "be": None}})]
    assert I.applied_objective(local, "A") == I.applied_objective(remote, "A") != ""
    rows = [{"model": m, "controller": "proposed", "instance": "x", "intent": 0, "objective": I.applied_objective(ev, "A")}
            for m, ev in ((I.LOCAL, local), (I.REMOTE, remote))]
    assert I.same_objectives(rows) == (1, 1)
    rows[1]["objective"] = I.applied_objective([event("received", 30.0, text="A"), act({"duration_s": 240})], "A")
    assert I.same_objectives(rows) == (0, 1)
