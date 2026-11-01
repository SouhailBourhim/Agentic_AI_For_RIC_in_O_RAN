"""Choose the agent's model by the rule pre-registered in docs/DECISIONS.md (M04b).

  uv run python -m eval.benchmark_models
  uv run python -m eval.benchmark_models --models qwen3:8b llama3.1:8b

For each candidate: skip it if Ollama does not have it; unload every other model; score the
parse step on the TUNE split (eval/extraction.py, prompt v1, temperature 0, num_ctx 4096);
then read `ollama ps`: a model not entirely in GPU memory is excluded (it spills to the CPU and
an agent loop becomes very slow). Among the models that fit, highest full-intent accuracy;
within one intent, lower median latency. Writes runs/m04b/model_benchmark.json.
"""

import argparse
import json
import urllib.request
from pathlib import Path

from agent.llm import DEFAULT_URL, OllamaClient
from agent.prompts import PARSE_VERSION
from eval.extraction import evaluate, load_intents

CANDIDATES = ["qwen3.5:9b", "ministral-3:8b", "granite4.2:8b", "qwen3:8b", "llama3.1:8b"]
OUT = Path("runs/m04b/model_benchmark.json")


def ollama(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{DEFAULT_URL}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def installed() -> set[str]:
    return {m["name"] for m in ollama("/api/tags")["models"]}


def unload_all() -> None:
    """Free the GPU: every loaded model gets keep_alive 0 (Ollama unloads it)."""
    for m in ollama("/api/ps")["models"]:
        ollama("/api/generate", {"model": m["name"], "keep_alive": 0})


def gpu_share(model: str) -> float:
    """Fraction of the loaded model that sits in GPU memory (1.0 = entirely)."""
    for m in ollama("/api/ps")["models"]:
        if m["name"] == model:
            return m["size_vram"] / m["size"] if m["size"] else 0.0
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="*", default=CANDIDATES)
    args = parser.parse_args()
    available = installed()
    tune = load_intents("tune")
    rows = []
    for model in args.models:
        if model not in available:
            print(f"{model:18s} not installed: skipped")
            rows.append({"model": model, "status": "not installed"})
            continue
        unload_all()
        result = evaluate(model, "tune", tune, llm=OllamaClient(model=model))
        share = gpu_share(model)
        fits = share >= 0.999
        rows.append({"model": model, "status": "ok" if fits else "excluded: not entirely on GPU",
                     "gpu_share": round(share, 3), "correct": result["correct"], "n": result["n"],
                     "accuracy": result["accuracy"], "latency_median_s": result["latency_median_s"],
                     "json_valid": result["json_valid"], "per_field": result["per_field"],
                     "wrong": [i["id"] for i in result["items"] if not i["correct"]]})
        print(f"{model:18s} {result['correct']:2d}/{result['n']} correct, median {result['latency_median_s']:.1f} s, "
              f"GPU {share:.0%}{'' if fits else '  -> EXCLUDED'}")
    eligible = [r for r in rows if r["status"] == "ok"]
    chosen = None
    if eligible:
        best = max(r["correct"] for r in eligible)
        close = [r for r in eligible if r["correct"] >= best - 1]  # ties within one intent
        chosen = min(close, key=lambda r: r["latency_median_s"])["model"]
    print(f"\nchosen (rule fixed in DECISIONS.md): {chosen or 'NONE'}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"prompt": PARSE_VERSION, "split": "tune", "rows": rows, "chosen": chosen}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
