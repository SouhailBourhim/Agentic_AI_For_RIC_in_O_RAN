"""The LLM behind the agent, behind one small interface so it can be swapped.

- LLMClient: the interface (a typing.Protocol). The graph only ever calls chat().
- OllamaClient: the local model on Windows (localhost:11434), over plain HTTP with the standard
  library: no SDK, so the whole exchange is readable here.
- FakeLLM: scripted replies for the tests, so they run without a GPU (and in CI).
A cloud model can be added later as one more class with the same chat() method.

The LLM is never on the real-time path (hard invariant 5): only the agent worker calls it, and
the RL runner never waits for the worker.
"""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "qwen3:8b"  # replaced by the benchmark's choice (eval/benchmark_models.py)
DEFAULT_URL = "http://localhost:11434"


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMReply:
    content: str  # the text (or JSON, when a schema was given)
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_s: float = 0.0


class LLMClient(Protocol):
    model: str

    def chat(self, messages: list[dict[str, str]], tools: list[dict] | None = None,
             schema: dict | None = None) -> LLMReply:
        """One request. `tools`: tool definitions the model may call (OpenAI-style JSON).
        `schema`: a JSON schema the reply must follow (structured output)."""
        ...


class LLMError(Exception):
    """The model could not be reached or answered with an error."""


class OllamaClient:
    """A model served by Ollama. Temperature 0 so the same prompt gives the same answer."""

    def __init__(self, model: str | None = None, url: str | None = None, num_ctx: int = 4096,
                 timeout_s: float = 120.0) -> None:
        self.model = model or os.environ.get("RIC_LLM_MODEL", DEFAULT_MODEL)
        self.url = (url or os.environ.get("RIC_OLLAMA_URL", DEFAULT_URL)).rstrip("/")
        self.num_ctx = num_ctx  # above ~4k the KV cache spills out of the 8 GB GPU
        self.timeout_s = timeout_s
        self.send_think = True  # "think": false turns off the reasoning trace of thinking models

    def chat(self, messages: list[dict[str, str]], tools: list[dict] | None = None,
             schema: dict | None = None) -> LLMReply:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False,
                                "options": {"temperature": 0, "num_ctx": self.num_ctx}}
        if tools:
            body["tools"] = tools
        if schema is not None:
            body["format"] = schema
        if self.send_think:
            body["think"] = False
        start = time.monotonic()
        try:
            reply = self._post(body)
        except LLMError as err:
            if self.send_think and "think" in str(err).lower():  # model without a thinking mode
                self.send_think = False
                body.pop("think")
                reply = self._post(body)
            else:
                raise
        message = reply.get("message", {})
        calls = [ToolCall(c["function"]["name"], c["function"].get("arguments") or {})
                 for c in message.get("tool_calls") or []]
        return LLMReply(message.get("content", ""), calls, time.monotonic() - start)

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as err:
            raise LLMError(f"Ollama {err.code}: {err.read().decode(errors='replace')[:300]}") from err
        except (urllib.error.URLError, TimeoutError) as err:
            raise LLMError(f"cannot reach Ollama at {self.url}: {err}") from err


class FakeLLM:
    """Replies from a script, in order; records every request. For tests only."""

    def __init__(self, replies: list[LLMReply | str], model: str = "fake") -> None:
        self.model = model
        self.replies = list(replies)
        self.requests: list[dict] = []

    def chat(self, messages: list[dict[str, str]], tools: list[dict] | None = None,
             schema: dict | None = None) -> LLMReply:
        self.requests.append({"messages": messages, "tools": tools, "schema": schema})
        if not self.replies:
            raise LLMError("FakeLLM: no scripted reply left")
        reply = self.replies.pop(0)
        return LLMReply(reply) if isinstance(reply, str) else reply


def tool_call(name: str, **arguments: Any) -> LLMReply:
    """A reply that calls one tool (for FakeLLM scripts)."""
    return LLMReply("", [ToolCall(name, arguments)])
