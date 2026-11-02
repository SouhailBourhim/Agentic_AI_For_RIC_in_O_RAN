"""Run the MCP gateway.

  uv run python -m gateway          stdio: the MCP client starts this process and talks over its
                                    stdin/stdout (gateway/client.py does this)
  uv run python -m gateway --http   streamable HTTP on http://127.0.0.1:8100/mcp (Hermes, M05b)

In stdio mode stdout is the protocol channel: nothing else may print to it. The gateway logs to
var/events.jsonl; the SDK's own logs go to stderr.
Settings (environment): RIC_AGENT_URL (default http://127.0.0.1:8000), RIC_EVENTS.
"""

import argparse

from gateway.server import mcp


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m gateway", description=__doc__.splitlines()[0])
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    if args.http:
        mcp.run("streamable-http", host=args.host, port=args.port)
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
