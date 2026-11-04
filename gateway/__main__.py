"""Run the MCP gateway.

  uv run python -m gateway          stdio: the MCP client starts this process and talks over its
                                    stdin/stdout (gateway/client.py does this)
  uv run python -m gateway --http   streamable HTTP on http://127.0.0.1:8100/mcp

Hermes (M05b) reaches it through a relay, whose name arrives as the HTTP Host header:
  uv run python -m gateway --http --host 0.0.0.0 --allowed-host gateway-relay:8100

In stdio mode stdout is the protocol channel: nothing else may print to it. The gateway logs to
var/events.jsonl; the SDK's own logs go to stderr.
Settings (environment): RIC_AGENT_URL (default http://127.0.0.1:8000), RIC_EVENTS.
"""

import argparse

from mcp.server.transport_security import TransportSecuritySettings

from gateway.server import mcp

LOCAL_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]


def host_check(extra_hosts: list[str]) -> TransportSecuritySettings:
    """Refuse any HTTP Host header but these. The SDK only does this by itself for its default
    hosts on a localhost bind, and Hermes's relay arrives under its own name."""
    return TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                     allowed_hosts=[*LOCAL_HOSTS, *extra_hosts], allowed_origins=[])


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m gateway", description=__doc__.splitlines()[0])
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--allowed-host", action="append", default=[],
                        help="an extra accepted Host header, e.g. gateway-relay:8100 (repeatable)")
    args = parser.parse_args()
    if args.http:
        mcp.run("streamable-http", host=args.host, port=args.port, transport_security=host_check(args.allowed_host))
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
