"""The LLM behind the agent, behind one small interface so it can be swapped.

- LLMClient: the interface (a typing.Protocol). The graph only ever calls chat().
- OllamaClient: the local model on Windows (localhost:11434), over plain HTTP with the standard
  library: no SDK, so the whole exchange is readable here.
- NimClient: a remote model on NVIDIA NIM (OpenAI-compatible API), the same way (M09i).
- FakeLLM: scripted replies for the tests, so they run without a GPU (and in CI).
- client_from_env(): which one the worker and the evaluations use (RIC_LLM_PROVIDER, default ollama).
Every client counts its calls and tokens, so an evaluation can read what a run cost.

The LLM is never on the real-time path (hard invariant 5): only the agent worker calls it, and
the RL runner never waits for the worker.
"""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

DEFAULT_MODEL = "qwen3:8b"  # replaced by the benchmark's choice (eval/benchmark_models.py)
DEFAULT_URL = "http://localhost:11434"
NIM_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"  # M09i (2026-11-14): best of 10 probed on texts outside the scored set;
# Kimi K3.1 Pro (the first pick) isn't served to this key, Mistral Large 4 reasons for up to 3 min per call
NIM_KEY_FILE = Path.home() / ".config" / "agentic-ric" / "nim_api_key"  # chmod 600, never in the repo
RETRY_STATUS = (429, 500, 502, 503, 504)  # rate limit (about 40 requests/min) and passing server errors
RETRY_WAITS_S = (2.0, 4.0, 8.0, 16.0, 30.0, 30.0)
NIM_MAX_REQUESTS = 1500  # per client, retries included: a hard ceiling on remote use (M09i needs about 1,300 in all)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMReply:
    content: str  # the text (or JSON, when a schema was given)
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_s: float = 0.0
    prompt_tokens: int = 0  # as the server counts them (Ollama: prompt_eval_count; NIM: usage)
    completion_tokens: int = 0


class Usage:
    """Running totals of one client: an evaluation reads the difference before and after an item or a run."""

    def __init__(self) -> None:
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self.seconds = 0.0

    def add(self, reply: "LLMReply") -> "LLMReply":
        self.calls += 1
        self.prompt_tokens += reply.prompt_tokens
        self.completion_tokens += reply.completion_tokens
        self.seconds += reply.latency_s
        return reply

    def snapshot(self) -> dict[str, float]:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "seconds": round(self.seconds, 3)}


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
        self.usage = Usage()

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
        return self.usage.add(LLMReply(message.get("content", ""), calls, time.monotonic() - start,
                                       int(reply.get("prompt_eval_count") or 0), int(reply.get("eval_count") or 0)))

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


def nim_key() -> str:
    """The NIM key: NVIDIA_API_KEY, else the key file. Never logged, never written by this code."""
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key and NIM_KEY_FILE.exists():
        key = NIM_KEY_FILE.read_text().strip()
    if not key:
        raise LLMError(f"no NIM key: set NVIDIA_API_KEY or write it to {NIM_KEY_FILE} (chmod 600)")
    return key


class NimClient:
    """A remote model on NVIDIA NIM, through its OpenAI-compatible chat API. Temperature 0, like OllamaClient.

    Differences from Ollama handled here: tool-call arguments arrive as a JSON string; a JSON schema is asked
    for with `response_format` (or NIM's `nvext.guided_json`, schema_mode="guided_json"); a rate limit (429)
    or a passing server error is retried after a wait (Retry-After when the server gives one). `extra` is
    merged into every request body (RIC_NIM_EXTRA, JSON), for model switches such as turning thinking off.
    """

    def __init__(self, model: str | None = None, url: str | None = None, timeout_s: float = 180.0,
                 schema_mode: str | None = None, extra: dict | None = None, sleep=time.sleep,
                 max_requests: int | None = None) -> None:
        self.model = model or os.environ.get("RIC_LLM_MODEL", NIM_MODEL)
        self.url = (url or os.environ.get("RIC_NIM_URL", NIM_URL)).rstrip("/")
        self.timeout_s = timeout_s
        self.schema_mode = schema_mode or os.environ.get("RIC_NIM_SCHEMA_MODE", "response_format")
        self.extra = extra if extra is not None else json.loads(os.environ.get("RIC_NIM_EXTRA", "{}"))
        self.sleep = sleep
        self.usage = Usage()
        self.max_requests = max_requests or int(os.environ.get("RIC_NIM_MAX_REQUESTS", NIM_MAX_REQUESTS))
        self.requests = 0  # HTTP requests sent, retries included
        self._key = nim_key()

    def chat(self, messages: list[dict[str, str]], tools: list[dict] | None = None,
             schema: dict | None = None) -> LLMReply:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0, **self.extra}
        if tools:
            body["tools"] = tools
        if schema is not None:
            if self.schema_mode == "guided_json":
                body["nvext"] = {"guided_json": schema}
            else:
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": "reply", "schema": schema, "strict": True}}
        start = time.monotonic()
        reply = self._post_with_retry(body)
        message = (reply.get("choices") or [{}])[0].get("message") or {}
        calls = []
        for c in message.get("tool_calls") or []:
            raw = c.get("function", {}).get("arguments") or "{}"
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except ValueError:
                continue  # not a usable call: the agent asks again, as when no tool was chosen
            calls.append(ToolCall(c["function"]["name"], arguments if isinstance(arguments, dict) else {}))
        usage = reply.get("usage") or {}
        return self.usage.add(LLMReply(message.get("content") or "", calls, time.monotonic() - start,
                                       int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)))

    def list_models(self) -> list[str]:
        request = urllib.request.Request(f"{self.url}/models", headers={"Authorization": f"Bearer {self._key}"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return sorted(m["id"] for m in json.loads(response.read()).get("data", []))
        except (urllib.error.URLError, TimeoutError) as err:
            raise LLMError(self._redact(f"cannot list NIM models: {err}")) from None

    def _post_with_retry(self, body: dict) -> dict:
        for wait in (*RETRY_WAITS_S, None):
            try:
                return self._post(body)
            except _Retryable as err:
                if wait is None:
                    raise LLMError(self._redact(f"NIM {err.status} after {len(RETRY_WAITS_S)} retries: {err.text}")) from None
                self.sleep(err.retry_after if err.retry_after is not None else wait)
        raise AssertionError("unreachable")

    def _post(self, body: dict) -> dict:
        if self.requests >= self.max_requests:  # refuse before sending: the ceiling is never crossed
            raise LLMError(f"NIM request ceiling reached ({self.max_requests}, RIC_NIM_MAX_REQUESTS): nothing more is sent")
        self.requests += 1
        request = urllib.request.Request(f"{self.url}/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Accept": "application/json",
                                                  "Authorization": f"Bearer {self._key}"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as err:
            text = err.read().decode(errors="replace")[:300]
            if err.code in RETRY_STATUS:
                after = err.headers.get("Retry-After") if err.headers else None
                raise _Retryable(err.code, text, float(after) if after and after.isdigit() else None) from None
            raise LLMError(self._redact(f"NIM {err.code}: {text}")) from None
        except (urllib.error.URLError, TimeoutError) as err:
            raise LLMError(self._redact(f"cannot reach NIM at {self.url}: {err}")) from None

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text


class _Retryable(Exception):
    def __init__(self, status: int, text: str, retry_after: float | None) -> None:
        super().__init__(status)
        self.status, self.text, self.retry_after = status, text, retry_after


def client_from_env() -> "LLMClient":
    """The LLM the worker and the evaluations use: RIC_LLM_PROVIDER=ollama (default) or nim, model RIC_LLM_MODEL."""
    provider = os.environ.get("RIC_LLM_PROVIDER", "ollama")
    if provider == "nim":
        return NimClient()
    if provider == "ollama":
        return OllamaClient()
    raise LLMError(f"unknown RIC_LLM_PROVIDER {provider!r}: ollama or nim")


class FakeLLM:
    """Replies from a script, in order; records every request. For tests only."""

    def __init__(self, replies: list[LLMReply | str], model: str = "fake") -> None:
        self.model = model
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.usage = Usage()

    def chat(self, messages: list[dict[str, str]], tools: list[dict] | None = None,
             schema: dict | None = None) -> LLMReply:
        self.requests.append({"messages": messages, "tools": tools, "schema": schema})
        if not self.replies:
            raise LLMError("FakeLLM: no scripted reply left")
        reply = self.replies.pop(0)
        return self.usage.add(LLMReply(reply) if isinstance(reply, str) else reply)


def tool_call(name: str, **arguments: Any) -> LLMReply:
    """A reply that calls one tool (for FakeLLM scripts)."""
    return LLMReply("", [ToolCall(name, arguments)])
