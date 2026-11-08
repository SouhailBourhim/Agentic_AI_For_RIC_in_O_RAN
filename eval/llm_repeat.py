"""Does the same prompt give the same decision? (M09a's diagnosis; M09b measures it per instance.)

  uv run python -m eval.llm_repeat eval/results/m09a2/m08d/s1/llm_only/seed9000 [--times 5]

Rebuilds LLM only's second plan prompt from a run's own log (the operator's text, the parsed facts, the
get_kpis result and the backend time, exactly as agent/graph.py builds it) and sends it to the model several
times: once after an unrelated prompt, then repeated. At temperature 0 the answers should be identical; with
qwen3:8b through Ollama they weren't always (the M09a note). Prints one line per call.
"""

import argparse
import json
from pathlib import Path

from agent import tools
from agent.llm import OllamaClient
from agent.prompts import PLAN_SYSTEM
from store.events import read_events


def second_plan_prompt(work: Path) -> list[dict[str, str]]:
    events = read_events(work / "events.jsonl")
    steps = [e["data"] for e in events if e["event"] == "agent_step"]
    text = next(e["data"]["text"] for e in events if e["event"] == "received")
    facts = next(d["facts"] for d in steps if d.get("node") == "parse")
    kpis = next(d for d in steps if d.get("node") == "act" and d.get("tool") == "get_kpis")
    second = [d for d in steps if d.get("node") == "plan"][1]
    user = (f"Operator request: {text}\nExtracted facts: {json.dumps(facts)}\nBackend time: {second['backend_t']:g} s\n"
            f"get_kpis -> {json.dumps(kpis['result'])}" + "\nChoose exactly one tool.")
    return [{"role": "system", "content": PLAN_SYSTEM["llm_only"]}, {"role": "user", "content": user}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--times", type=int, default=5)
    args = parser.parse_args()
    llm = OllamaClient()
    messages = second_plan_prompt(args.run)
    llm.chat([{"role": "user", "content": "Say hello in one word."}])  # start from an unrelated prompt
    for i in range(args.times):
        reply = llm.chat(messages, tools=tools.definitions("llm_only"))
        call = reply.tool_calls[0] if reply.tool_calls else None
        print(f"call {i + 1}: {call.name if call else None} {call.arguments if call else reply.content[:80]}")


if __name__ == "__main__":
    main()
