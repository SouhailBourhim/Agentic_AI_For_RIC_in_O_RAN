"""The operator confirmation console (hard invariant 6, hypothesis H4).

A separate process with its own path to var/ric.db, not reachable from Hermes or the MCP
gateway: the agent HTTP API has no confirm endpoint. Confirming re-verifies the intent against
what is applied now, then writes the a1_policy timetable through agent/apply.py.

  uv run python -m console list            pending intents, applied intents, the timetable
  uv run python -m console confirm i0003   apply it (re-verified first)
  uv run python -m console refuse i0003
  uv run python -m console cancel i0002    withdraw an applied intent
"""

import argparse
import sys
import time

from agent import intents as machine
from agent.apply import ApplyError, apply_intent, backend_now, cancel_intent
from store import a1_policy, db, intents


def show_list(conn) -> None:
    now_wall = time.time()
    machine.expire_pending(conn, now_wall)
    now = backend_now(conn)
    print(f"backend clock: {now:g} s\n")
    pending = intents.in_states(conn, ("awaiting_confirmation",))
    print("PENDING (confirm or refuse):" if pending else "No intent awaiting confirmation.")
    for r in pending:
        obj = r.objective
        print(f"  {r.intent_id}  \"{r.text}\"  ({r.confirm_deadline - now_wall:.0f} s left)")
        print(f"         weights {obj.weights.as_list()}  min Mb/s {obj.min_thr_mbps.as_list()}"
              f"  latency bound {obj.max_latency_ms} ms  until t={obj.valid_until:g} s")
        print(f"         verifier: {r.reason}")
    applied = intents.in_states(conn, ("applied",))
    if applied:
        print("\nAPPLIED:")
        for r in applied:
            print(f"  {r.intent_id}  \"{r.text}\"  until t={r.objective.valid_until:g} s")
    schedule = a1_policy.current_schedule(conn)
    if schedule:
        print("\nTIMETABLE (a1_policy, newest schedule):")
        for row in schedule:
            mark = "  <- in force" if row.valid_from <= now < (row.expires_at or float("inf")) else ""
            print(f"  policy {row.policy_id}: t={row.valid_from:g}..{row.expires_at:g} s"
                  f"  intents {', '.join(row.intent_ids)}{mark}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="console", description="Operator confirmation console")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for name in ("confirm", "refuse", "cancel"):
        sub.add_parser(name).add_argument("intent_id")
    args = parser.parse_args()

    conn = db.connect()
    try:
        if args.command == "list":
            show_list(conn)
            return
        if args.command == "confirm":
            record = apply_intent(conn, args.intent_id, time.time(), confirmed_by="console")
        elif args.command == "refuse":
            record = machine.refuse(conn, args.intent_id, time.time())
        else:
            record = cancel_intent(conn, args.intent_id, time.time())
        print(f"{record.intent_id}: {record.state}" + (f" ({record.reason})" if record.reason else ""))
    except (ApplyError, ValueError) as err:
        print(f"refused: {err}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
