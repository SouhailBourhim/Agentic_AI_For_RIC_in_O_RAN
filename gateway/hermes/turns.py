"""Send operator texts to the isolated Hermes container, one fresh one-shot session per text (M05b).

  uv run python gateway/hermes/turns.py --out FILE "text" ["text" ...]
  uv run python gateway/hermes/turns.py --out FILE --from-file TEXTS   (one text per line)

The text goes through stdin (--query-file -), so the shell interprets nothing and the bytes reach
Hermes exactly as written. One session per text because the real window is only 4096 tokens
(docs/learn/M05b-hermes.md section 6). Appends one JSON line per turn to FILE: the text, Hermes's
raw stream-json events (tokens merged, tool_use and tool_result kept) and the wall-clock bounds.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

COMPOSE = ["docker", "compose", "-f", "gateway/hermes/docker-compose.yml"]
HERMES = ["hermes", "chat", "--oneshot", "--query-file", "-", "--format", "stream-json"]


def run_turn(text: str, timeout_s: float = 240.0) -> dict:
    """One Hermes session for one text. The events are as Hermes printed them, except that the many
    one-token `text` events are merged into one, with the time of the first token."""
    start = time.time()
    proc = subprocess.run([*COMPOSE, "exec", "-T", "--workdir", "/tmp", "hermes", *HERMES], input=text,
                          capture_output=True, text=True, timeout=timeout_s)
    events: list[dict] = []
    for line in proc.stdout.splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        if event.get("type") == "text" and events and events[-1].get("type") == "text":
            events[-1]["text"] += event.get("text", "")
            events[-1]["last_timestamp"] = event.get("timestamp")
        else:
            events.append(event)
    return {"text": text, "start_s": start, "end_s": time.time(), "exit_code": proc.returncode,
            "events": events, "stderr": proc.stderr[-500:]}


def summary(turn: dict) -> str:
    calls = [e.get("name") or e.get("tool") for e in turn["events"] if e.get("type") == "tool_use"]
    result = next((e for e in turn["events"] if e.get("type") == "result"), {})
    return (f"{result.get('duration_ms', 0) / 1000:5.1f} s, tools {calls}, "
            f"tokens in {result.get('tokens', {}).get('input')}\n  {str(result.get('text'))[:700]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("texts", nargs="*")
    parser.add_argument("--from-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    texts = args.texts + (args.from_file.read_text(encoding="utf-8").splitlines() if args.from_file else [])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for text in texts:
        turn = run_turn(text)
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
        print(f"> {text}\n{summary(turn)}\n")


if __name__ == "__main__":
    main()
