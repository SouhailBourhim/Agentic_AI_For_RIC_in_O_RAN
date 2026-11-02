"""Figure for M05a: the demo through the gateway, read back from the event log.

Run with:  uv run python -m gateway.plot_demo [--events var/events.jsonl] [--out docs/learn/img/M05a-demo.png]

It only reads var/events.jsonl (the latest run: events after the last runner 'run_start'), so it
draws what happened, not what should have happened. Wall clock, seconds since the first gateway call.
Top panel: every tool call the gateway received, one row per tool, marker by outcome.
Bottom panel: every intent submitted in that run, one row each, with its state changes; the last
marker is the final state (applied / rejected / expired).

This file imports only matplotlib and store.events, like the rest of gateway/ (tests/test_gateway.py).
"""

import argparse
import textwrap
from pathlib import Path

from matplotlib.figure import Figure

from store.events import read_events

OUT = Path("docs/learn/img/M05a-demo.png")
TOOLS = ["submit_intent", "get_status", "get_kpis"]
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e2e1dc"  # as in sim/plot.py
# Outcomes are states, so they use the status colours (dataviz reference palette), each with its
# own marker shape so that colour never carries the meaning alone.
OUTCOME_STYLE = {
    "ok": ("#0ca30c", "o", "ok"),
    "invalid_args": ("#fab219", "s", "refused: invalid arguments"),
    "error": ("#d03b3b", "X", "error: agent service"),
}
FINAL_STYLE = {"applied": ("#0ca30c", "o"), "rejected": ("#d03b3b", "X"), "expired": ("#fab219", "s"),
               "cancelled": ("#fab219", "s")}
INTENT_EVENTS = {("agent", "received"): "received", ("agent", "awaiting_confirmation"): "awaiting confirmation",
                 ("console", "refused"): "refused at console", ("agent", "applied"): "applied",
                 ("agent", "expired"): "expired", ("console", "cancelled"): "cancelled"}


def latest_run(events: list[dict]) -> list[dict]:
    starts = [i for i, e in enumerate(events) if e["source"] == "runner" and e["event"] == "run_start"]
    return events[starts[-1]:] if starts else events


def outcome_kind(outcome: str) -> str:
    return outcome if outcome in ("ok", "invalid_args") else "error"


def _style(ax, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=9)


def plot(events: list[dict], out: Path) -> Path:
    calls = [e for e in events if e["source"] == "gateway" and e["event"] == "tool_call"]
    if not calls:
        raise SystemExit("no gateway tool_call event in this run: run the demo first")
    t0 = calls[0]["ts_wall"]
    submitted = []  # intent ids in submission order, as the gateway returned them
    for e in calls:
        if e["data"]["tool"] == "submit_intent" and e["intent_id"] and e["intent_id"] not in submitted:
            submitted.append(e["intent_id"])

    fig = Figure(figsize=(10, 2.2 + 0.45 * (len(TOOLS) + len(submitted))), layout="constrained")
    fig.set_facecolor(SURFACE)
    ax_calls, ax_intents = fig.subplots(2, 1, sharex=True,
                                        height_ratios=[len(TOOLS), max(len(submitted), 1)])

    for kind, (colour, marker, label) in OUTCOME_STYLE.items():
        xs = [e["ts_wall"] - t0 for e in calls if outcome_kind(e["data"]["outcome"]) == kind]
        ys = [TOOLS.index(e["data"]["tool"]) for e in calls if outcome_kind(e["data"]["outcome"]) == kind]
        if xs:
            ax_calls.scatter(xs, ys, s=40, color=colour, marker=marker, edgecolors=SURFACE,
                             linewidths=1.5, label=label, zorder=3)
    ax_calls.set_yticks(range(len(TOOLS)), TOOLS)
    ax_calls.set_ylim(len(TOOLS) - 0.5, -0.5)
    _style(ax_calls, "tool calls\nreceived")
    ax_calls.set_title("The only three tools the gateway exposes, and every call it received",
                       loc="left", color=TEXT, fontsize=10)
    ax_calls.legend(loc="upper right", frameon=False, fontsize=8)  # the right side is kept empty (xlim)

    for row, intent_id in enumerate(submitted):
        mine = [e for e in events if e["intent_id"] == intent_id and (e["source"], e["event"]) in INTENT_EVENTS]
        done = [e for e in events if e["intent_id"] == intent_id and e["event"] == "agent_done"]
        times = [e["ts_wall"] - t0 for e in mine]
        ax_intents.plot(times, [row] * len(times), color=GRID, linewidth=1.6, zorder=1)
        for k, e in enumerate(mine):  # labels alternate above and below, so close steps stay readable
            ax_intents.scatter(e["ts_wall"] - t0, row, s=28, color=TEXT_MUTED, zorder=2)
            ax_intents.annotate(INTENT_EVENTS[(e["source"], e["event"])], (e["ts_wall"] - t0, row),
                                xytext=(0, 8 if k % 2 == 0 else -13), textcoords="offset points", fontsize=7,
                                color=TEXT_MUTED, ha="center")
        final = next((e["data"]["outcome"] for e in done if e["data"]["outcome"] in FINAL_STYLE), None)
        if final == "rejected":  # the rejection is not a separate event: it is in agent_done
            e = done[-1]
            reason = textwrap.shorten(e["data"]["reason"], 100, placeholder=" ...")
            ax_intents.annotate("rejected: " + "\n".join(textwrap.wrap(reason, 55)), (e["ts_wall"] - t0, row),
                                xytext=(9, 0), textcoords="offset points", fontsize=7, color=TEXT, va="center")
            mine.append(e)
        if final is not None and mine:
            colour, marker = FINAL_STYLE[final]
            ax_intents.scatter(mine[-1]["ts_wall"] - t0, row, s=60, color=colour, marker=marker,
                               edgecolors=SURFACE, linewidths=1.5, zorder=3)
    last = max(e["ts_wall"] for e in events if e["source"] == "gateway" or e["intent_id"] in submitted) - t0
    ax_intents.set_xlim(-0.05 * last, 1.45 * last)  # room on the right for the rejection reasons
    ax_intents.set_yticks(range(len(submitted)), submitted)
    ax_intents.set_ylim(len(submitted) - 0.4, -0.6)
    _style(ax_intents, "intents")
    ax_intents.set_xlabel("wall clock (s since the first gateway call)", color=TEXT_MUTED, fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="M05a figure: gateway calls and intents of the latest run")
    parser.add_argument("--events", type=Path, default=None, help="event log (default: RIC_EVENTS or var/events.jsonl)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(f"wrote {plot(latest_run(read_events(args.events)), args.out)}")


if __name__ == "__main__":
    main()
