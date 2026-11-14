"""M09i: the local LLM (qwen3:8b on the laptop's GPU) against a remote one (NVIDIA NIM), on the same intents.

  uv run python -m eval.m09i --check                     one plain call, one tool call, one parse, on the remote model
  uv run python -m eval.m09i --extraction                the parse step on the 40 annotated intents, 5 repeats per model
  uv run python -m eval.m09i --loop [--limit N]          the whole agent on 25 dev instances, LLM + RL and LLM only v2
  uv run python -m eval.m09i --table                     the tables, from what is on disk

The proposal (§ "Pour le modèle") plans this comparison: gain in accuracy, latency and cost. Descriptive, no pass
rule; the sample, the measures and how they are read are in docs/DECISIONS.md (M09i) before the first remote run.
- The parse step is M08b's (`eval.extraction.evaluate`: the frozen prompt, the same scoring), on both splits of
  eval/intents/annotated.jsonl, reported apart: the tune split tuned the prompt for qwen3:8b, the held-out one was
  scored once for it in M08b. Nothing is tuned or chosen on either here.
- The closed loop is M09b's player (`eval.benchmark.play`) on dev instances (the test seeds stay closed). Its harness
  charges every LLM call's real latency to the simulated clock, so a slower model pays in reaction and intent met.
- Calls and tokens come from each client's own counters (agent.llm.Usage); GPU energy from nvidia-smi, once a
  second, during the local model's runs.
Raw results: eval/results/m09i/ (git-ignored). Tables: docs/learn/data/M09i-*.csv.
"""

import argparse
import csv
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from agent import tools
from agent.graph import llm_parse
from agent.llm import NIM_MODEL, LLMError, NimClient, OllamaClient
from agent.prompts import PLAN_SYSTEM
from eval.benchmark import dropped, outcome_of, play
from eval.extraction import FIELDS, evaluate
from eval.generator import FAMILIES, Instance, dev_instances
from eval.interface_metrics import wilson
from eval.m09d import bootstrap_means, interval
from eval.metrics import load_run, measure_run
from eval.validate_sim import served_policy

RESULTS = Path("eval/results/m09i")
DATA = Path("docs/learn/data")
EXTRACTION_CSV, RUNS_CSV, STEPS_CSV, SUMMARY_CSV = (DATA / f"M09i-{n}.csv" for n in ("extraction", "runs", "steps", "summary"))
MODELS = {"qwen3_8b": ("ollama", "qwen3:8b"), "nemotron_3_ultra": ("nim", NIM_MODEL)}  # name: (provider, model id)
# Both answer without a reasoning trace: qwen3:8b with Ollama's think=false (agent.llm), Nemotron with this switch
# (checked 2026-11-14: same parse, 1.1 s instead of 3.7 s; left on, it wrote ~1,750 tokens per LLM-only call).
NIM_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}
LOCAL, REMOTE = "qwen3_8b", "nemotron_3_ultra"
SPLITS = ("tune", "heldout")
REPEATS = 5  # as M08b
CONTROLLERS = ("proposed", "llm_only_v2")
PER_FAMILY = 5
SAMPLE_SEED = 20261114


def client(name: str):
    provider, model = MODELS[name]
    return OllamaClient(model=model) if provider == "ollama" else NimClient(model=model, extra=NIM_EXTRA)


def sample(keep: int = PER_FAMILY) -> list[Instance]:
    """5 dev instances per family, drawn with a fixed seed, without the ones M09b's parse check dropped;
    `keep` < 5 keeps the first ones of each family (the dry run)."""
    rng = np.random.default_rng(SAMPLE_SEED)
    skip = dropped()
    out = []
    for family in FAMILIES:
        pool = [i for i in dev_instances((family,)) if i.name not in skip]
        out += [pool[k] for k in sorted(rng.choice(len(pool), PER_FAMILY, replace=False))][:keep]
    return out


class GpuEnergy:
    """The GPU's power draw, once a second, while the local model works (nvidia-smi; the whole board)."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, check=False).stdout.strip()
            try:
                self.samples.append((time.monotonic(), float(out.splitlines()[0])))
            except (ValueError, IndexError):
                pass
            self._stop.wait(1.0)

    def __enter__(self) -> "GpuEnergy":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def wh(self) -> float:
        t = np.array([s[0] for s in self.samples])
        w = np.array([s[1] for s in self.samples])
        return float(np.sum((t[1:] - t[:-1]) * (w[1:] + w[:-1]) / 2) / 3600) if len(t) > 1 else 0.0


def idle_watts(seconds: float = 10.0) -> float:
    with GpuEnergy() as meter:
        time.sleep(seconds)
    return float(np.mean([w for _, w in meter.samples])) if meter.samples else 0.0


# ---------------------------------------------------------------- the check


def check() -> None:
    """Can the key reach the remote model, and does it call tools and follow a JSON schema? Stops on the first no."""
    remote = client(REMOTE)
    try:
        models = remote.list_models()
        print(f"{len(models)} models reachable; {NIM_MODEL} {'listed' if NIM_MODEL in models else 'NOT listed'}")
    except LLMError as err:
        print(f"model list unavailable: {err}")
    plain = remote.chat([{"role": "user", "content": "Reply with OK."}])
    print(f"plain: {plain.content[:60]!r}, {plain.latency_s:.1f} s, tokens {plain.prompt_tokens}+{plain.completion_tokens}")
    user = ("Operator request: Guarantee 10 Mb/s to the URLLC slice for 10 minutes.\nExtracted facts: "
            + json.dumps({"understood": True, "clarification": None, "priority": None,
                          "min_thr_mbps": {"embb": None, "urllc": 10, "be": None}, "max_latency_ms": None,
                          "duration_s": 600}) + "\nBackend time: 0 s\nChoose exactly one tool.")
    tool = remote.chat([{"role": "system", "content": PLAN_SYSTEM["proposed"]}, {"role": "user", "content": user}],
                       tools=tools.definitions("proposed"))
    print(f"tool call: {[(c.name, c.arguments) for c in tool.tool_calls]}, {tool.latency_s:.1f} s")
    facts, latency, attempts = llm_parse(remote, "Assure au moins 12 Mb/s au slice best effort pendant 40 minutes.")  # not an annotated intent
    print(f"parse: {facts.model_dump() if facts else None}, {latency:.1f} s, {attempts} attempt(s)")
    if not tool.tool_calls or facts is None:
        raise SystemExit(f"{NIM_MODEL} can't serve the agent here: "
                         + ("no tool call. " if not tool.tool_calls else "") + ("no valid parse." if facts is None else ""))
    print("check passed:", remote.usage.snapshot())


# ---------------------------------------------------------------- the parse step


def extraction(names=tuple(MODELS)) -> None:
    for name in names:
        llm = client(name)
        for split in SPLITS:
            for repeat in range(1, REPEATS + 1):
                out = RESULTS / "extraction" / name / f"{split}_r{repeat}.json"
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                before = llm.usage.snapshot()
                with GpuEnergy() as meter:
                    result = evaluate(MODELS[name][1], split, llm=llm)
                after = llm.usage.snapshot()
                result["usage"] = {k: round(after[k] - before[k], 3) for k in after}
                result["gpu_wh"] = round(meter.wh(), 4) if MODELS[name][0] == "ollama" else None
                out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
                print(f"{name} {split} r{repeat}: {result['correct']}/{result['n']}, "
                      f"median {result['latency_median_s']:.1f} s, {result['usage']['calls']} calls", flush=True)


# ---------------------------------------------------------------- the closed loop


def work_dir(inst: Instance, controller: str, name: str) -> Path:
    return RESULTS / f"m09i_{name}" / inst.name / controller / f"seed{inst.seed}"


def loop(names=tuple(MODELS), limit: int | None = None) -> None:
    instances = sample(limit or PER_FAMILY)
    policy = served_policy()
    for name in names:
        llm = client(name)
        idle = idle_watts() if MODELS[name][0] == "ollama" else None
        for n, (inst, controller) in enumerate(((i, c) for i in instances for c in CONTROLLERS), 1):
            if (work_dir(inst, controller, name) / "m09i.json").exists():
                continue
            before, began = llm.usage.snapshot(), time.monotonic()
            with GpuEnergy() as meter:
                work = play(inst, controller, policy, llm, study=f"m09i_{name}", root=RESULTS)
            after = llm.usage.snapshot()
            usage = {k: round(after[k] - before[k], 3) for k in after}
            (work / "m09i.json").write_text(json.dumps({
                "model": MODELS[name][1], "provider": MODELS[name][0], "usage": usage, "wall_s": round(time.monotonic() - began, 1),
                "gpu_wh": round(meter.wh(), 4) if idle is not None else None, "idle_w": idle}))
            print(f"[{name} {n}/{len(instances) * len(CONTROLLERS)}] {inst.name} {controller}: "
                  f"{usage['calls']} calls, {usage['prompt_tokens']}+{usage['completion_tokens']} tokens", flush=True)


# ---------------------------------------------------------------- the tables


def extraction_rows() -> list[dict]:
    rows = []
    for path in sorted((RESULTS / "extraction").glob("*/*_r*.json")):
        result = json.loads(path.read_text())
        name, (split, repeat) = path.parent.name, path.stem.split("_r")
        rows.append({"model": name, "split": split, "repeat": int(repeat), "n": result["n"], "correct": result["correct"],
                     "accuracy": round(result["accuracy"], 4), "json_valid": result["json_valid"],
                     **{f"field_{f}": round(result["per_field"][f], 4) for f in FIELDS},
                     "latency_median_s": round(result["latency_median_s"], 3), "latency_p95_s": round(result["latency_p95_s"], 3),
                     "calls": result["usage"]["calls"], "prompt_tokens": result["usage"]["prompt_tokens"],
                     "completion_tokens": result["usage"]["completion_tokens"], "gpu_wh": result.get("gpu_wh")})
    return rows


def intent_ids(events: list[dict], text: str) -> list[str]:
    return [e["intent_id"] for e in events if e["event"] == "received" and e["data"].get("text") == text]


def decision_s(events: list[dict], text: str) -> float | str:
    """Simulated seconds from the intent's arrival to the first change applied for it (an objective, or LLM only's split).

    The harness charges each LLM call's latency to the simulated clock, so this is where a slow model shows.
    """
    ids = intent_ids(events, text)
    arrived = [e["ts_backend"] for e in events if e["event"] == "received" and e["intent_id"] in ids]
    applied = [e["ts_backend"] for e in events if e["event"] in ("applied", "shares_applied") and e["intent_id"] in ids]
    return round(applied[0] - arrived[0], 1) if arrived and applied else ""


def applied_objective(events: list[dict], text: str) -> str:
    """The first objective the verifier accepted for this intent, numbers as floats (LLM + RL only; '' otherwise).

    Written in one canonical form so two models' choices compare as text: Nemotron sends some numbers as strings
    ("300.0") and leaves out empty slices, which the schema reads the same way.
    """
    ids = intent_ids(events, text)
    for e in events:
        d = e["data"]
        if (e["event"] == "agent_step" and e["intent_id"] in ids and d.get("node") == "act"
                and d.get("tool") == "set_rl_objective" and d.get("verifier", {}).get("accepted")):
            args = d["arguments"]

            def num(v):
                return None if v is None else float(v)
            canon = {"duration_s": num(args.get("duration_s")), "max_latency_ms": num(args.get("max_latency_ms")),
                     "min_thr_mbps": {s: float(v) for s, v in sorted((args.get("min_thr_mbps") or {}).items()) if v is not None}}
            return json.dumps(canon, sort_keys=True)
    return ""


def step_rows() -> list[dict]:
    """Every timed LLM call of the closed loop, with its latency (retry waits included: the agent waited for them)."""
    rows = []
    for name in MODELS:
        for inst in sample():
            for controller in CONTROLLERS:
                work = work_dir(inst, controller, name)
                if not (work / "m09i.json").exists():
                    continue
                for e in load_run(work)[1]:
                    if e["event"] == "agent_step" and e["data"].get("latency_s"):
                        rows.append({"model": name, "controller": controller, "instance": inst.name,
                                     "node": e["data"].get("node", ""), "latency_s": e["data"]["latency_s"]})
    return rows


def same_objectives(loop_rows: list[dict]) -> tuple[int, int]:
    """LLM + RL: on how many intents both models had the same objective applied, of those both applied."""
    chosen = {(r["model"], r["instance"], r["intent"]): r["objective"] for r in loop_rows if r["controller"] == "proposed"}
    pairs = [(chosen[(LOCAL, i, k)], o) for (m, i, k), o in chosen.items()
             if m == REMOTE and (LOCAL, i, k) in chosen and o and chosen[(LOCAL, i, k)]]
    return sum(a == b for a, b in pairs), len(pairs)


def run_rows() -> list[dict]:
    """One row per run and intent, measured as every simulator run is (eval.metrics), with the run's LLM usage."""
    rows = []
    for name in MODELS:
        for inst in sample():
            for controller in CONTROLLERS:
                work = work_dir(inst, controller, name)
                if not (work / "m09i.json").exists():
                    continue
                side = json.loads((work / "m09i.json").read_text())
                measured = measure_run(work, inst.scenario())
                events = load_run(work)[1]
                steps = [e["data"] for e in events if e["event"] == "agent_step" and e["data"].get("latency_s")]
                net_wh = (side["gpu_wh"] - side["idle_w"] * side["wall_s"] / 3600) if side["gpu_wh"] is not None else ""
                for k, (intent, row) in enumerate(zip(inst.intents, measured["intents"], strict=True)):
                    rows.append({"model": name, "family": inst.family, "instance": inst.name, "controller": controller,
                                 "intent": k, "intent_met": row["intent_met"], "reaction_s": row.get("reaction_s", ""),
                                 "decision_s": decision_s(events, intent.text), "objective": applied_objective(events, intent.text),
                                 "outcome": outcome_of(events, intent.text), "intents_in_run": len(inst.intents),
                                 "calls": side["usage"]["calls"], "prompt_tokens": side["usage"]["prompt_tokens"],
                                 "completion_tokens": side["usage"]["completion_tokens"], "llm_s": side["usage"]["seconds"],
                                 "step_latency_median_s": float(np.median([s["latency_s"] for s in steps])) if steps else "",
                                 "gpu_net_wh": round(net_wh, 4) if net_wh != "" else "", "wall_s": side["wall_s"]})
    return rows


def paired(rows: list[dict], controller: str, key: str = "intent_met") -> dict:
    """Remote minus local, intent by intent, with M09d's cluster bootstrap by instance."""
    def per_instance(model: str) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for r in rows:
            if r["model"] == model and r["controller"] == controller:
                out.setdefault(r["instance"], []).append(float(r[key]))  # intents in the instance's order
        return out

    local, remote = per_instance(LOCAL), per_instance(REMOTE)
    groups = [np.array(remote[inst]) - np.array(local[inst]) for inst in local
              if inst in remote and len(remote[inst]) == len(local[inst])]
    if not groups:
        return {}
    low, high = interval(bootstrap_means(groups), 0.95)
    return {"mean_diff": round(float(np.concatenate(groups).mean()), 4), "low_95": round(low, 4), "high_95": round(high, 4),
            "instances": len(groups)}


def summary(ex_rows: list[dict], loop_rows: list[dict]) -> list[dict]:
    out = []
    for name in MODELS:
        for split in SPLITS:
            mine = [r for r in ex_rows if r["model"] == name and r["split"] == split]
            if not mine:
                continue
            correct, n = sum(r["correct"] for r in mine), sum(r["n"] for r in mine)
            low, high = wilson(correct, n)
            out.append({"part": "parse", "model": name, "split_or_controller": split, "value": round(correct / n, 4),
                        "low_95": round(low, 4), "high_95": round(high, 4), "what": f"intents right, {len(mine)} repeats",
                        "latency_median_s": round(float(np.median([r["latency_median_s"] for r in mine])), 3),
                        "tokens_per_call": round(sum(r["prompt_tokens"] + r["completion_tokens"] for r in mine)
                                                 / max(1, sum(r["calls"] for r in mine)), 1)})
        for controller in CONTROLLERS:
            mine = [r for r in loop_rows if r["model"] == name and r["controller"] == controller]
            if not mine:
                continue
            runs = {r["instance"]: r for r in mine}
            out.append({"part": "loop", "model": name, "split_or_controller": controller,
                        "value": round(float(np.mean([r["intent_met"] for r in mine])), 4), "low_95": "", "high_95": "",
                        "what": f"mean intent met, {len(mine)} intents in {len(runs)} runs",
                        "latency_median_s": round(float(np.median([r["step_latency_median_s"] for r in runs.values()
                                                                   if r["step_latency_median_s"] != ""] or [0])), 3),
                        "tokens_per_call": round(sum(r["prompt_tokens"] + r["completion_tokens"] for r in runs.values())
                                                 / max(1, sum(r["calls"] for r in runs.values())), 1),
                        "calls_per_run": round(float(np.mean([r["calls"] for r in runs.values()])), 1),
                        "decision_median_s": median_or_blank([r["decision_s"] for r in mine]),
                        "decision_max_s": max([r["decision_s"] for r in mine if r["decision_s"] != ""], default="")})
    for controller in CONTROLLERS:
        diff = paired(loop_rows, controller)
        if diff:
            out.append({"part": "loop, remote minus local", "model": f"{REMOTE} - {LOCAL}", "split_or_controller": controller,
                        "value": diff["mean_diff"], "low_95": diff["low_95"], "high_95": diff["high_95"],
                        "what": f"paired intent met, cluster bootstrap over {diff['instances']} instances"})
    same, both = same_objectives(loop_rows)
    if both:
        out.append({"part": "loop, same objective applied", "model": f"{REMOTE} = {LOCAL}", "split_or_controller": "proposed",
                    "value": round(same / both, 4), "what": f"{same} of {both} intents: the first accepted objective, compared"})
    return out


def median_or_blank(values: list) -> float | str:
    kept = [v for v in values if v != ""]
    return round(float(np.median(kept)), 1) if kept else ""


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def table() -> list[dict]:
    ex, runs, steps = extraction_rows(), run_rows(), step_rows()
    rows = summary(ex, runs)
    for path, data in ((EXTRACTION_CSV, ex), (RUNS_CSV, runs), (STEPS_CSV, steps), (SUMMARY_CSV, rows)):
        if data:
            write_csv(path, data)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--extraction", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--limit", type=int, default=None, help="--loop: only the first N sampled instances of each family")
    args = parser.parse_args()
    if args.check:
        check()
    if args.extraction:
        extraction(args.models)
    if args.loop:
        loop(args.models, args.limit)
    if args.table:
        for row in table():
            print(row)


if __name__ == "__main__":
    main()
