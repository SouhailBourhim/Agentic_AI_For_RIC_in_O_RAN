"""A small command-line MCP client for the gateway: the operator's side when Hermes is not used.

It speaks MCP like Hermes would: it lists the tools the gateway offers and calls them by name.
It has one command per tool and nothing else, so it can do no more than Hermes could.

  uv run python -m gateway.client tools                     the tools the gateway exposes
  uv run python -m gateway.client submit "Garantis 10 Mb/s au slice URLLC pendant 10 minutes."
  uv run python -m gateway.client status [i0001]
  uv run python -m gateway.client kpis [--slice urllc] [--window 30]
  uv run python -m gateway.client watch i0001               get_status every 2 s until a final state

By default it starts the gateway itself over stdio (`python -m gateway`, same environment).
--url http://127.0.0.1:8100/mcp connects to a gateway already serving HTTP instead.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp import Client, StdioServerParameters

REPO = Path(__file__).resolve().parent.parent
# States after which an intent does not move on its own any more ('applied' can still be
# cancelled from the console, but nothing reachable from here changes it).
FINAL_STATES = {"rejected", "applied", "expired", "cancelled"}
WATCH_PERIOD_S = 2.0


def gateway_server(url: str | None) -> StdioServerParameters | str:
    """What Client connects to: a URL (HTTP gateway) or the command that starts a stdio gateway.

    The SDK passes only a few safe variables (PATH, HOME...) to the subprocess; the RIC_* settings
    are passed on explicitly so the gateway logs and calls where this shell would.
    """
    if url is not None:
        return url
    settings = {key: value for key, value in os.environ.items() if key.startswith("RIC_")}
    return StdioServerParameters(command=sys.executable, args=["-m", "gateway"], env=settings, cwd=REPO)


def show(tool: str, result) -> dict | None:
    """Print one tool result with a wall-clock timestamp; return the JSON body (None on error)."""
    stamp = time.strftime("%H:%M:%S")
    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
    if result.is_error:
        print(f"[{stamp}] {tool}: ERROR {text}")
        return None
    body = json.loads(text)
    print(f"[{stamp}] {tool}:\n{json.dumps(body, indent=2, ensure_ascii=False)}")
    return body


async def run(args: argparse.Namespace) -> int:
    async with Client(gateway_server(args.url)) as client:
        if args.command == "tools":
            for tool in (await client.list_tools()).tools:
                parameters = ", ".join(tool.input_schema.get("properties", {}))
                print(f"{tool.name}({parameters})\n    {' '.join((tool.description or '').split())}\n")
            return 0
        if args.command == "submit":
            return 0 if show("submit_intent", await client.call_tool("submit_intent", {"text": args.text})) else 1
        if args.command == "kpis":
            arguments = {"window_s": args.window, **({"slice": args.slice} if args.slice else {})}
            return 0 if show("get_kpis", await client.call_tool("get_kpis", arguments)) else 1
        if args.command == "status":
            arguments = {"intent_id": args.intent_id} if args.intent_id else {}
            return 0 if show("get_status", await client.call_tool("get_status", arguments)) else 1
        # watch: print only when the state changes, stop at a final state
        last = None
        while True:
            result = await client.call_tool("get_status", {"intent_id": args.intent_id})
            if result.is_error:
                show("get_status", result)
                return 1
            body = json.loads(result.content[0].text)
            if body["state"] != last:
                show("get_status", result)
                last = body["state"]
            if last in FINAL_STATES:
                return 0
            await asyncio.sleep(WATCH_PERIOD_S)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m gateway.client", description="MCP client of the gateway")
    parser.add_argument("--url", help="HTTP gateway URL (default: start a stdio gateway)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("tools", help="list the tools the gateway exposes")
    submit = commands.add_parser("submit", help="submit_intent")
    submit.add_argument("text")
    status = commands.add_parser("status", help="get_status")
    status.add_argument("intent_id", nargs="?")
    kpis = commands.add_parser("kpis", help="get_kpis")
    kpis.add_argument("--slice")
    kpis.add_argument("--window", type=float, default=10.0)
    watch = commands.add_parser("watch", help="get_status until a final state")
    watch.add_argument("intent_id")
    sys.exit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
