"""Figures for docs/learn/M05b-hermes.md, from the files in testbed/results/m05b/.

  uv run python gateway/hermes/plot.py

1. M05b-turns.png: where the time goes in each demo turn through Hermes.
2. M05b-fidelity.png: per tune-split intent, text forwarded exactly or not, and how long it took.
3. M05b-context.png: what a larger context window costs on this GPU.
4. M05b-session.png: the real prompt of one growing session against the 4096-token window.
5. M05b-probe.png: what the Hermes container can reach, against the same probe on the testbed network.
"""

import csv
import json
import sys
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `import sim`
from sim.plot import SURFACE, TEXT, TEXT_MUTED, _style  # noqa: E402

RESULTS = Path("testbed/results/m05b")
IMG = Path("docs/learn/img")
# Categories, fixed colours (the project's three slice hues mean slices elsewhere, so not reused here).
START, LLM, TOOL, REPLY = "#b9b8b3", "#6250d6", "#0b0b0b", "#a99ee8"
EXACT, PUNCT, NOT_SENT = "#0b0b0b", "#b9860b", "#c0392b"


def load_turns(name: str) -> list[dict]:
    return [json.loads(line) for line in (RESULTS / name).read_text(encoding="utf-8").splitlines()]


def result_of(turn: dict) -> dict:
    return next(e for e in turn["events"] if e.get("type") == "result")


def plot_turns() -> Figure:
    """Stacked segments per turn: Hermes starting, LLM choosing, the gateway call, LLM replying."""
    turns = load_turns("demo.jsonl")
    fig = Figure(figsize=(12, 4.6), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    for row, turn in enumerate(reversed(turns)):
        ev = turn["events"]
        init = next(e["timestamp"] for e in ev if e.get("type") == "system") / 1000
        use = [e["timestamp"] / 1000 for e in ev if e.get("type") == "tool_use"]
        res = [e["timestamp"] / 1000 for e in ev if e.get("type") == "tool_result"]
        end = result_of(turn)["timestamp"] / 1000
        segments = [(turn["start_s"], init, START), (init, use[0], LLM), (use[0], res[0], TOOL), (res[0], end, REPLY)]
        for a, b, colour in segments:
            ax.barh(row, b - a, left=a - turn["start_s"], color=colour, edgecolor=SURFACE, linewidth=1.5, height=0.7)
        ax.annotate(f"{end - turn['start_s']:.1f} s", (end - turn["start_s"], row), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(range(len(turns)), [t["text"][:58] for t in reversed(turns)], fontsize=8)
    for colour, label in ((START, "Hermes starts (client, daemon, MCP connection)"), (LLM, "LLM chooses the tool"),
                          (TOOL, "gateway call"), (REPLY, "LLM writes the reply")):
        ax.barh(0, 0, color=colour, label=label)
    ax.set_xlabel("seconds since the operator's text was sent (one fresh session per text)", color=TEXT_MUTED, fontsize=8)
    _style(ax, "")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT)
    ax.set_title("The demo through Hermes: the gateway call itself is too short to see", color=TEXT, fontsize=11, loc="left")
    return fig


def fidelity_rows(name: str, texts: list[str]) -> list[tuple[str, float]]:
    """(status, seconds) per text. Status compares what reached submit_intent with what was typed."""
    rows = []
    for text, turn in zip(texts, load_turns(name), strict=True):
        sent = [e["input"].get("text") for e in turn["events"]
                if e.get("type") == "tool_use" and "submit_intent" in e.get("name", "")]
        status = "exact" if sent == [text] else "punct" if sent == [text.rstrip(".")] else "not_sent"
        rows.append((status, result_of(turn)["duration_ms"] / 1000))
    return rows


def plot_fidelity() -> Figure:
    texts = (RESULTS / "fidelity_texts.txt").read_text(encoding="utf-8").splitlines()
    fig = Figure(figsize=(12, 6.2), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 2, sharey=True, sharex=True)
    style = {"exact": ("o", EXACT, "forwarded character for character"),
             "punct": ("s", PUNCT, "only the final period lost"),
             "not_sent": ("X", NOT_SENT, "not submitted (no reply or declined)")}
    for ax, name, title in ((axes[0], "fidelity.jsonl", "First SOUL.md"), (axes[1], "fidelity_v2.jsonl", "After one tightening")):
        rows = fidelity_rows(name, texts)
        for i, (status, seconds) in enumerate(rows):
            marker, colour, _ = style[status]
            ax.scatter(seconds, len(texts) - 1 - i, marker=marker, color=colour, s=48, zorder=3)
        exact = sum(s == "exact" for s, _ in rows)
        ax.set_title(f"{title}: {exact} / {len(rows)} exact", color=TEXT, fontsize=10, loc="left")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("seconds for the whole turn (log)", color=TEXT_MUTED, fontsize=8)
        _style(ax, "")
    axes[0].set_yticks(range(len(texts)), [t[:52] for t in reversed(texts)], fontsize=7.5)
    for marker, colour, label in style.values():
        axes[1].scatter([], [], marker=marker, color=colour, label=label)
    axes[1].legend(frameon=False, fontsize=8, loc="center right", labelcolor=TEXT)
    fig.suptitle("The 20 tune-split intents through Hermes: is the operator's text what reaches the gateway?",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    return fig


def plot_context() -> Figure:
    rows = list(csv.DictReader((RESULTS / "context_vs_gpu.csv").open()))
    labels = [f"{int(r['num_ctx']):,}" for r in rows]
    fig = Figure(figsize=(10, 3.6), facecolor=SURFACE, layout="constrained")
    ax_gpu, ax_speed = fig.subplots(1, 2)
    for ax, key, title, unit in ((ax_gpu, "gpu_pct", "Share of the model in GPU memory", "%"),
                                 (ax_speed, "tok_per_s", "Generation speed", " tok/s")):
        values = [float(r[key]) for r in rows]
        bars = ax.bar(labels, values, color=LLM, width=0.6)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v:g}{unit}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=TEXT)
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("num_ctx requested (65,536 is capped by Ollama to the model's 40,960)", color=TEXT_MUTED, fontsize=8)
        ax.set_ylim(0, max(values) * 1.15)
        _style(ax, "")
    fig.suptitle("qwen3:8b on this 8 GB GPU: Hermes's 64K floor cannot really be met", color=TEXT, fontsize=12, x=0.01, ha="left")
    return fig


def plot_session() -> Figure:
    rows = list(csv.DictReader((RESULTS / "session_growth.csv").open()))
    x = [int(r["request"]) for r in rows]
    fig = Figure(figsize=(9, 3.8), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    ax.plot(x, [int(r["chars_sent"]) / 4 for r in rows], color=TEXT_MUTED, linewidth=1.5, linestyle="--",
            label="messages Hermes sends, growing (characters / 4, without the tool schemas)")
    ax.plot(x, [int(r["prompt_tokens"]) for r in rows], color=LLM, linewidth=2, marker="o", markersize=4,
            label="what the model really read (Ollama's prompt_tokens)")
    ax.axhline(4096, color=NOT_SENT, linewidth=1.2)
    ax.annotate("real window: 4096", (1, 4096), xytext=(0, 4), textcoords="offset points", fontsize=8, color=TEXT)
    ax.set_xlabel("LLM request number in one resumed session", color=TEXT_MUTED, fontsize=8)
    _style(ax, "tokens")
    ax.set_ylim(0, 10500)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TEXT, bbox_to_anchor=(0, 0.9))
    ax.set_title("A long session silently loses its beginning (and the prompt cache)", color=TEXT, fontsize=11, loc="left")
    return fig


def probe_words(row: dict) -> str:
    """The Python error of the connection attempt, in plain words."""
    if row["reached"]:
        return "connected"
    kind = row["detail"].split(":")[0]
    return {"ConnectionRefusedError": "refused, port closed", "OSError": "no route",
            "gaierror": "name does not resolve", "TimeoutError": "timed out"}.get(kind, kind)


def plot_probe() -> Figure:
    """Two columns, same targets: only the network differs between them."""
    hermes = json.loads((RESULTS / "network_probe.json").read_text())
    control = json.loads((RESULTS / "network_probe_control.json").read_text())
    fig = Figure(figsize=(11, 5.4), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    for x, rows in ((0, hermes), (1, control)):
        for i, r in enumerate(reversed(rows)):
            ax.scatter(x, i, marker="o" if r["reached"] else "X", s=70, color=EXACT if r["reached"] else NOT_SENT)
            ax.annotate(probe_words(r), (x + 0.03, i), va="center", fontsize=8, color=TEXT)
    ax.set_yticks(range(len(hermes)), [f"{r['label']}  ({r['host']}:{r['port']})" for r in reversed(hermes)], fontsize=8)
    ax.set_xticks([0, 1], ["Hermes container", "Same image on the testbed network (control)"], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_xlim(-0.04, 1.42)
    _style(ax, "")
    ax.grid(False)
    ax.set_title("Testbed up: the addresses are alive, and only the two relay ports answer Hermes", color=TEXT, fontsize=11, loc="left")
    return fig


def main() -> None:
    for name, plot in (("M05b-turns.png", plot_turns), ("M05b-fidelity.png", plot_fidelity),
                       ("M05b-context.png", plot_context), ("M05b-session.png", plot_session),
                       ("M05b-probe.png", plot_probe)):
        plot().savefig(IMG / name, dpi=120)
        print(f"wrote {IMG / name}")


if __name__ == "__main__":
    main()
