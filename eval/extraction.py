"""Extraction score: how well a parser turns operator text into the right facts (H8, M08b).

  uv run python -m eval.extraction --parser rules --split tune
  uv run python -m eval.extraction --parser qwen3:8b --split tune
  uv run python -m eval.extraction --parser <model> --split heldout --final     (M08b, once)

The parser is the agent's real parse step (agent/graph.py llm_parse with prompt
agent/prompts.py, or the regex twin agent/parser.py), scored against eval/intents/annotated.jsonl.

Scoring rules (eval/intents/README.md):
- an intent is correct when every field matches: understood, priority, the three minimum
  throughputs, the latency bound and the duration (numbers within 1e-6, null only matches null);
- when the annotation says understood=false, only `understood` is scored (the wording of the
  clarification question is free);
- a reply that is not valid JSON for the schema counts as wrong.
"""

import argparse
import json
import statistics
from pathlib import Path

from agent.graph import llm_parse
from agent.llm import OllamaClient
from agent.parser import parse_facts
from agent.prompts import PARSE_VERSION
from agent.schema import ParsedIntent

INTENTS = Path("eval/intents/annotated.jsonl")
RESULT_DIR = Path("runs/m04b")
HELD_OUT_SCORED = Path("eval/results/m08b/heldout/extraction/done.json")
FIELDS = ("priority", "min_thr_embb", "min_thr_urllc", "min_thr_be", "max_latency_ms", "duration_s")


def load_intents(split: str, path: Path = INTENTS) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["split"] == split]


def flat(facts: ParsedIntent | dict) -> dict:
    """The scored fields of a ParsedIntent, flattened."""
    d = facts.model_dump() if isinstance(facts, ParsedIntent) else facts
    return {"understood": d["understood"], "priority": d["priority"],
            "min_thr_embb": d["min_thr_mbps"]["embb"], "min_thr_urllc": d["min_thr_mbps"]["urllc"],
            "min_thr_be": d["min_thr_mbps"]["be"], "max_latency_ms": d["max_latency_ms"],
            "duration_s": d["duration_s"]}


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-6
    return a == b


def score_one(expected: dict, got: ParsedIntent | None) -> dict:
    """Per-field verdicts for one intent, and the overall one."""
    want = flat(expected)
    if got is None:
        return {"correct": False, "understood_ok": False, "json_valid": False, "fields": {}}
    have = flat(got)
    understood_ok = want["understood"] == have["understood"]
    if not want["understood"]:
        return {"correct": understood_ok, "understood_ok": understood_ok, "json_valid": True, "fields": {}}
    fields = {f: same(want[f], have[f]) for f in FIELDS} if have["understood"] else {f: False for f in FIELDS}
    return {"correct": understood_ok and all(fields.values()), "understood_ok": understood_ok,
            "json_valid": True, "fields": fields}


def evaluate(parser: str, split: str, intents: list[dict] | None = None, llm=None) -> dict:
    """Run the parser on every intent of the split and summarise."""
    rows = intents if intents is not None else load_intents(split)
    if parser != "rules" and llm is None:
        llm = OllamaClient(model=parser)
    items, latencies = [], []
    for row in rows:
        if parser == "rules":
            got, latency = parse_facts(row["text"]), 0.0
        else:
            got, latency, _ = llm_parse(llm, row["text"])
        latencies.append(latency)
        verdict = score_one(row["expected"], got)
        items.append({"id": row["id"], "text": row["text"], **verdict,
                      "got": got.model_dump() if got is not None else None})
    understood_rows = [i for i, r in zip(items, rows) if r["expected"]["understood"]]
    per_field = {f: sum(i["fields"].get(f, False) for i in understood_rows) / max(1, len(understood_rows))
                 for f in FIELDS}
    return {
        "parser": parser, "prompt": PARSE_VERSION, "split": split, "n": len(items),
        "correct": sum(i["correct"] for i in items),
        "accuracy": sum(i["correct"] for i in items) / max(1, len(items)),
        "understood_agreement": sum(i["understood_ok"] for i in items) / max(1, len(items)),
        "json_valid": sum(i["json_valid"] for i in items) / max(1, len(items)),
        "per_field": per_field,
        "latency_median_s": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_s": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)] if latencies else 0.0,
        "items": items,
    }


def print_report(result: dict) -> None:
    print(f"{result['parser']} on {result['split']} (prompt {result['prompt']}): "
          f"{result['correct']}/{result['n']} correct ({result['accuracy']:.0%}), "
          f"understood agreement {result['understood_agreement']:.0%}, valid JSON {result['json_valid']:.0%}, "
          f"latency median {result['latency_median_s']:.1f} s, p95 {result['latency_p95_s']:.1f} s")
    print("  per field (intents meant to be understood): " +
          ", ".join(f"{f} {v:.0%}" for f, v in result["per_field"].items()))
    for item in result["items"]:
        if not item["correct"]:
            print(f"  WRONG {item['id']}: {item['text']}\n        got {json.dumps(item['got'], ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parser", default="rules", help="'rules' or an Ollama model tag")
    parser.add_argument("--split", choices=["tune", "heldout"], default="tune")
    parser.add_argument("--final", action="store_true", help="required for the held-out split (M08b, once)")
    args = parser.parse_args()
    if args.split == "heldout" and not args.final:
        raise SystemExit("the held-out split is scored once, in M08b: add --final")
    if args.split == "heldout" and HELD_OUT_SCORED.exists():
        raise SystemExit("the held-out split has already been scored (eval/results/m08b); its results stand")
    result = evaluate(args.parser, args.split)
    print_report(result)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f"extraction_{args.parser.replace(':', '_').replace('/', '_')}_{args.split}_{PARSE_VERSION}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
