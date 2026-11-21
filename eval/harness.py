"""One evaluation run: one controller on one scenario with one seed, in one process (M08a).

The simulated clock is in charge. The harness ticks the same Runner as production and hands intents
to the same agent code (agent.intents.submit, agent.worker.process_one). Time passes in three ways:
- with nothing to do, one tick at a time;
- while the agent observes (wait_backend, as in agent/plot_trace.py);
- during every LLM call: ChargedLLM runs the cell for the seconds the call really took, on the split
  it already had, before the agent gets the reply. Without this an LLM would think for free.

So a run keeps the LLM's real latency but not the idle waiting. This is NOT the real-time path of
hard invariant 5: there the runner never waits for anything. eval/realtime_check.py runs the separate
processes at real speed and compares the two.

Nothing here calls apply_prb_split: the Runner stays the only caller (hard invariant 4).
"""

import csv
import json
import math
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from agent import tools
from agent.apply import apply_intent, backend_now, decision_dict
from agent.intents import submit
from agent.tools import AgentContext, Settings
from agent.worker import process_one
from backends.sim import DelayedBackend, SimBackend, scenario_traffic
from eval.scenarios import Scenario, ScenarioIntent
from eval.oracle import FormulaToldPolicy, MpcEstPolicy, OraclePolicy
from rl.baselines import EqualSplitPolicy, FormulaEstPolicy, FormulaPolicy, SmoothedPolicy
from rl.policy import SMOOTHING_ALPHA, SlicePolicy
from rl.monitor import TRIGGERS, FallbackMonitor
from rl.runner import Runner
from rl.shield import ShieldedPolicy
from sim.cell import CellConfig
from sim.channel import ChannelTrace
from store import db, intents, kpi
from store.events import log_event
from verifier.objective import SLICES

CONTROLLERS = ("static", "rl_only", "llm_only", "proposed")
USES_RL = ("rl_only", "proposed", "scripted")  # scripted: the over-serving side study, no LLM
# M10b: "scripted" with an executor is the LLM-free executor mode. The generator's stated facts go through the real
# verifier and apply step with the default weights (apply_scripted), and the executor turns the objective into a
# split. Only "v7" among them needs a policy file; "oracle" reads the cell itself, with no KPI delay (its caller's job).
EXECUTORS = ("static", "v7", "rev4", "formula_est", "mpc_est", "formula_told", "oracle")
# M10c': the policy trained on the m10 cell, as one more executor (its file is the run's policy_file).
LEARNED_EXECUTORS = ("m10_rl", "shielded_rl")  # M11d: the run's policy file behind the URLLC shield (rl/shield.py)
# M09b's references: "formula" is LLM + formula (the proposed agent, rl.baselines.FormulaPolicy instead of RL);
# "llm_only_v2" is the fairer LLM only (its revised prompt); "oracle" is the ceiling (eval/oracle.py), no agent.
# M09g's diagnostic: "formula_a10" is LLM + formula behind the served RL policy's filter (spec.alpha).
USES_AGENT = ("llm_only", "proposed", "formula", "llm_only_v2", "formula_a10")
AGENT_CONTROLLER = {"formula": "proposed", "llm_only_v2": "llm_only", "formula_a10": "proposed"}  # the agent's mode
PROMPT_VERSION = {"llm_only_v2": "v2"}
OBSERVE_S = 20.0  # the worker's default: LLM only's decision period, the proposed loop's observation
OTHER_IMPORTANCE = 2.0  # of the slices without a target, in the scripted runs (agent/graph.py's rule)


class RunOver(Exception):
    """The scenario's time is up. Not one of the errors process_one handles, so it stops the agent
    wherever it is. Without it LLM only would go on for its intent's whole duration, and then the next
    intent's, past the end of the run."""


class ChargedLLM:
    """An LLM client whose calls cost simulated time: what the real latency costs the cell."""

    def __init__(self, llm, advance: Callable[[int], None]) -> None:
        self.llm = llm
        self.advance = advance
        self.model = llm.model

    def chat(self, messages, tools=None, schema=None):  # same signature as LLMClient
        reply = self.llm.chat(messages, tools=tools, schema=schema)
        self.advance(math.ceil(reply.latency_s))
        return reply


@dataclass(frozen=True)
class RunSpec:
    """Everything that defines a run."""

    study: str  # headline, bridge, lever or dev
    scenario: str
    controller: str
    seed: int
    alpha: float = SMOOTHING_ALPHA
    delay_ticks: int = 0
    importance: float | None = None  # scripted runs only: the favoured slice's importance
    executor: str | None = None  # scripted runs only (M10b): who turns the objective into a split; None = the RL policy
    monitor: str | None = None  # M11b: the runtime monitor's trigger (rl.monitor.TRIGGERS); None = off, as before M11b
    # M13: the verifier's offer after a rejection. Scripted runs: True = the operator takes it. Agent runs: True = the
    # agent shows it (Settings.offers). None = as before M13 (same run_id); False = the explicit "no offer" arm.
    offers: bool | None = None
    # M13b: the verifier sizes a bounded URLLC target from its traffic (RIC_BOUND_FROM_LOAD). None = as before M13b
    # (same run_id, load-blind); False = the explicit load-blind arm.
    bound_from_load: bool | None = None

    @property
    def run_id(self) -> str:
        variant = f"_a{self.alpha:g}_d{self.delay_ticks}" if self.study == "bridge" else ""
        variant += f"_i{self.importance:g}" if self.importance is not None else ""
        variant += f"_{self.executor}" if self.executor is not None else ""
        variant += f"_mon-{self.monitor}" if self.monitor is not None else ""
        variant += "" if self.offers is None else "_offer" if self.offers else "_nooffer"
        variant += "" if self.bound_from_load is None else "_load" if self.bound_from_load else "_blind"
        return f"{self.study}/{self.scenario}/{self.controller}{variant}/seed{self.seed}"


def apply_scripted(conn: sqlite3.Connection, intent: ScenarioIntent, high: float | None, now_wall: float,
                   take_offer: bool = False) -> None:
    """The over-serving side study: the objective a plan step with this importance would propose,
    with no LLM. Verified and applied by the same functions as the agent's.
    With high=None (M10b's executor mode) no importance at all: the default weights, as the agent keeps them for a
    first objective with no stated priority (agent/graph.py keep_default_weights).
    take_offer (M13): the operator who, told the request doesn't fit, takes the verifier's offer at once. The offer is
    proposed and verified again like any objective, then applied by the same path."""
    importance = {} if high is None else {s: high if s in intent.targets else OTHER_IMPORTANCE for s in SLICES}
    record = intents.create(conn, intent.text, now_wall, backend_now(conn), state="interpreted")
    log_event("agent", "received", record.intent_id, backend_now(conn), {"text": intent.text})
    ctx = AgentContext(conn=conn, settings=Settings(mode="eval"), llm=None)
    decision, objective = tools.propose_objective(ctx, record.intent_id, None, importance, intent.targets,
                                                  intent.max_latency_ms, intent.expected["duration_s"])
    log_event("agent", "agent_step", record.intent_id, backend_now(conn),
              {"node": "act", "tool": "set_rl_objective", "verifier": decision_dict(decision)})
    if not decision.accepted and take_offer and decision.offer is not None:
        offer = decision.offer
        decision, objective = tools.propose_objective(ctx, record.intent_id, None, importance,
                                                      offer.min_thr_mbps.model_dump(), offer.max_latency_ms,
                                                      intent.expected["duration_s"])
        log_event("agent", "offer_taken", record.intent_id, backend_now(conn),
                  {"offer": offer.model_dump(), "verifier": decision_dict(decision)})
    if decision.accepted:
        intents.set_state(conn, record.intent_id, "interpreted", now_wall, objective=objective,
                          verifier=decision_dict(decision), parsed=intent.expected)
        apply_intent(conn, record.intent_id, now_wall, confirmed_by="auto-eval")


def export_kpis(conn: sqlite3.Connection, path: Path, clock_offset_s: float = 0.0) -> None:
    """The KPIs on the scenario's clock: the backend's offset (if any) is taken off."""
    rows = conn.execute("SELECT ts, slice, dl_thr_mbps, offered_load_mbps, latency_ms, prb_share_alloc, policy_id "
                        "FROM kpi ORDER BY ts, slice").fetchall()
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "slice", "dl_thr_mbps", "offered_load_mbps", "latency_ms", "prb_share_alloc", "policy_id"])
        writer.writerows((r[0] - clock_offset_s, *tuple(r)[1:]) for r in rows)


def rebase_event_file(path: Path, clock_offset_s: float) -> None:
    """Events on the scenario's clock, as eval.testbed_control.rebase_events does for the testbed."""
    if not clock_offset_s or not path.exists():
        return
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    path.write_text("".join(json.dumps({**e, "ts_backend": e["ts_backend"] - clock_offset_s} if e.get("ts_backend") is not None else e,
                                       ensure_ascii=False) + "\n" for e in events))


def needs_policy_file(spec: RunSpec) -> bool:
    return spec.controller in USES_RL and spec.executor in (None, "v7", *LEARNED_EXECUTORS)


def make_policy(spec: RunSpec, backend, scenario: Scenario, cfg: CellConfig, policy_file: Path | None):
    """What stands where the RL policy stands in the Runner, for this run. The Runner applies its splits."""
    controller = spec.executor if spec.executor is not None else spec.controller
    if controller == "shielded_rl":  # M11d: before the learned-executor branch, which would load the policy bare
        return ShieldedPolicy(SlicePolicy.load(policy_file, spec.alpha), FormulaEstPolicy(cfg.told_urllc_base_latency_ms))
    if controller in ("v7", *LEARNED_EXECUTORS) or (spec.executor is None and spec.controller in USES_RL):
        return SlicePolicy.load(policy_file, spec.alpha)
    if controller in ("formula", "rev4"):
        return FormulaPolicy(cfg.told_urllc_base_latency_ms)
    if controller == "formula_a10":
        return SmoothedPolicy(FormulaPolicy(cfg.told_urllc_base_latency_ms), spec.alpha)
    if controller == "formula_est":
        return FormulaEstPolicy(cfg.told_urllc_base_latency_ms)
    if controller == "mpc_est":  # what a designer knows of the cell: told capacity and URLLC base, measured model
        return MpcEstPolicy(CellConfig(n_prb=cfg.n_prb, capacity_mbps=cfg.told_capacity_mbps, dt_s=cfg.dt_s,
                                       base_latency_ms={s: cfg.told_urllc_base_latency_ms for s in SLICES},
                                       max_backlog_mb=cfg.max_backlog_mb, delay_model=cfg.delay_model))
    if controller == "formula_told":  # evaluation only, like the oracle: reads the simulated cell
        return FormulaToldPolicy(backend, cfg)
    if controller == "oracle":  # evaluation only: reads the simulated cell itself (eval/oracle.py)
        return OraclePolicy(backend, scenario, cfg)
    return EqualSplitPolicy()


def run_once(spec: RunSpec, scenario: Scenario, work: Path, policy_file: Path | None, llm=None,
             cfg: CellConfig | None = None, traffic: dict | None = None, clock_offset_s: float = 0.0,
             channel: ChannelTrace | None = None) -> Path:
    """Play the scenario. Writes kpi.csv, events.jsonl and, last of all, run.json: a run directory
    without run.json did not finish and is played again. `cfg` and `traffic` default to the M01 cell and the
    scenario's own traffic; M09a passes the testbed profile and the testbed's segments instead, and a clock offset
    so the agent reads epoch-like times as on the testbed (the outputs are on the scenario's clock either way).
    `channel` gives each slice's UE a radio channel (M10a, eval.sim_profiles.channel_of); None = as before M10."""
    if needs_policy_file(spec) and policy_file is None:
        raise ValueError(f"{spec.controller} needs a policy file")
    if spec.executor is not None and (spec.controller != "scripted" or spec.executor not in EXECUTORS + LEARNED_EXECUTORS):
        raise ValueError(f"an executor runs under the scripted controller, one of {EXECUTORS + LEARNED_EXECUTORS}; "
                         f"got {spec}")
    if spec.controller in USES_AGENT and llm is None:
        raise ValueError(f"{spec.controller} needs an LLM")
    work.mkdir(parents=True, exist_ok=True)
    for old in ("events.jsonl", "ric.db", "ric.db-wal", "ric.db-shm", "kpi.csv"):
        (work / old).unlink(missing_ok=True)
    saved = {k: os.environ.get(k) for k in ("RIC_DB", "RIC_EVENTS", "RIC_BOUND_FROM_LOAD")}
    os.environ["RIC_DB"], os.environ["RIC_EVENTS"] = str(work / "ric.db"), str(work / "events.jsonl")
    os.environ["RIC_BOUND_FROM_LOAD"] = "1" if spec.bound_from_load else "0"  # explicit: earlier runs replay blind
    started = time.monotonic()
    try:
        conn = db.connect(work / "ric.db")
        cfg = cfg or CellConfig()
        traffic = traffic or scenario_traffic(scenario.name, cfg)
        backend = (DelayedBackend(cfg, traffic, spec.seed, spec.delay_ticks, clock_offset_s, channel) if spec.delay_ticks
                   else SimBackend(cfg, traffic, seed=spec.seed, speed=0.0, clock_offset_s=clock_offset_s, channel=channel))
        elapsed = lambda: backend.now() - clock_offset_s  # noqa: E731  the scenario's clock
        policy = make_policy(spec, backend, scenario, cfg, policy_file)
        kpi.write_backend_info(conn, backend.info(), spec.run_id)
        # on_error="raise": a bug ends the evaluation run instead of quietly playing the safe split (M11a)
        monitor = None if spec.monitor is None else FallbackMonitor(
            FormulaEstPolicy(cfg.told_urllc_base_latency_ms), cfg.n_prb, cfg.dt_s, TRIGGERS[spec.monitor])
        runner = Runner(backend, policy, db.connect(work / "ric.db", timeout_s=0.0), cfg.dt_s, on_error="raise",
                        monitor=monitor)
        waiting = list(scenario.intents)

        def advance(ticks: int) -> None:
            for _ in range(ticks):
                if elapsed() >= scenario.seconds:
                    raise RunOver
                runner.tick()
                while waiting and elapsed() >= waiting[0].at_s:
                    intent = waiting.pop(0)
                    if spec.controller in USES_AGENT:
                        submit(conn, intent.text, time.time())
                    elif spec.controller == "scripted":
                        apply_scripted(conn, intent, spec.importance, time.time(), take_offer=bool(spec.offers))
                    # static and RL only ignore intents; they are still scored against them

        ctx = None
        if spec.controller in USES_AGENT:
            settings = Settings(mode="eval", controller=AGENT_CONTROLLER.get(spec.controller, spec.controller), observe_s=OBSERVE_S,
                                llm_only_prompt=PROMPT_VERSION.get(spec.controller, "v1"), llm_only_joins=spec.controller == "llm_only_v2",
                                offers=bool(spec.offers))
            ctx = AgentContext(conn=conn, settings=settings, llm=ChargedLLM(llm, advance),
                               wait_backend=lambda s: advance(int(s)))
        try:
            while elapsed() < scenario.seconds:
                if ctx is None or process_one(ctx) is None:
                    advance(1)
        except RunOver:
            log_event("harness", "run_over", ts_backend=backend.now())
        export_kpis(conn, work / "kpi.csv", clock_offset_s)
        conn.close()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    rebase_event_file(work / "events.jsonl", clock_offset_s)
    for leftover in ("ric.db", "ric.db-wal", "ric.db-shm"):  # the csv and the events are the record
        (work / leftover).unlink(missing_ok=True)
    (work / "run.json").write_text(json.dumps({**asdict(spec), "run_id": spec.run_id,
                                               "wall_s": round(time.monotonic() - started, 1)}, indent=1))
    return work
