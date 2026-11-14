"""The remote LLM client (M09i): request shape, reply decoding, retries, the key's safety. No network."""

import io
import json
import urllib.error

import pytest

from agent import llm as L

KEY = "nvapi-test-SECRET-123"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", KEY)
    monkeypatch.delenv("RIC_NIM_EXTRA", raising=False)
    monkeypatch.delenv("RIC_NIM_SCHEMA_MODE", raising=False)
    monkeypatch.delenv("RIC_LLM_MODEL", raising=False)
    waits = []
    c = L.NimClient(sleep=waits.append)
    c.waits = waits
    return c


def answer(content="", tool_calls=None, prompt=12, completion=5) -> dict:
    return {"choices": [{"message": {"content": content, "tool_calls": tool_calls}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def test_a_tool_call_is_decoded_and_counted(client, monkeypatch) -> None:
    sent = []
    call = {"function": {"name": "set_rl_objective", "arguments": json.dumps({"min_thr_mbps": {"urllc": 10}})}}
    monkeypatch.setattr(client, "_post", lambda body: sent.append(body) or answer(tool_calls=[call]))
    tools = [{"type": "function", "function": {"name": "set_rl_objective", "parameters": {"type": "object"}}}]
    reply = client.chat([{"role": "user", "content": "hi"}], tools=tools)
    assert reply.tool_calls == [L.ToolCall("set_rl_objective", {"min_thr_mbps": {"urllc": 10}})]
    assert (reply.prompt_tokens, reply.completion_tokens) == (12, 5)
    assert sent[0]["model"] == L.NIM_MODEL and sent[0]["temperature"] == 0 and sent[0]["tools"] == tools
    assert client.usage.snapshot()["calls"] == 1 and client.usage.prompt_tokens == 12


def test_a_broken_tool_call_is_dropped_not_guessed(client, monkeypatch) -> None:
    call = {"function": {"name": "set_rl_objective", "arguments": "{not json"}}
    monkeypatch.setattr(client, "_post", lambda body: answer(tool_calls=[call]))
    assert client.chat([{"role": "user", "content": "hi"}]).tool_calls == []


@pytest.mark.parametrize("mode, key", [("response_format", "response_format"), ("guided_json", "nvext")])
def test_a_schema_is_asked_for_in_either_dialect(client, monkeypatch, mode, key) -> None:
    sent = []
    client.schema_mode = mode
    monkeypatch.setattr(client, "_post", lambda body: sent.append(body) or answer('{"understood": true}'))
    reply = client.chat([{"role": "user", "content": "x"}], schema={"type": "object"})
    assert key in sent[0] and reply.content == '{"understood": true}'


def http_error(code: int, headers: dict | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("u", code, "err", headers or {}, io.BytesIO(f"error {code} {KEY}".encode()))


def test_rate_limits_are_waited_out_then_given_up(client, monkeypatch) -> None:
    replies = iter([http_error(429, {"Retry-After": "7"}), http_error(503), answer("ok")])

    def fake_urlopen(request, timeout):
        item = next(replies)
        if isinstance(item, Exception):
            raise item
        return io.BytesIO(json.dumps(item).encode())

    monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)
    assert client.chat([{"role": "user", "content": "x"}]).content == "ok"
    assert client.waits == [7.0, L.RETRY_WAITS_S[1]]
    monkeypatch.setattr(L.urllib.request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(http_error(429)))
    with pytest.raises(L.LLMError, match="429 after"):
        client.chat([{"role": "user", "content": "x"}])


def test_the_key_never_appears_in_an_error(client, monkeypatch) -> None:
    monkeypatch.setattr(L.urllib.request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(http_error(401)))
    with pytest.raises(L.LLMError) as caught:
        client.chat([{"role": "user", "content": "x"}])
    assert KEY not in str(caught.value) and "***" in str(caught.value)


def test_the_key_is_sent_only_as_a_bearer_header(client, monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(L.urllib.request, "urlopen",
                        lambda request, timeout: seen.append(request) or io.BytesIO(json.dumps(answer("ok")).encode()))
    client.chat([{"role": "user", "content": "x"}])
    assert seen[0].get_header("Authorization") == f"Bearer {KEY}" and KEY not in seen[0].data.decode()


def test_no_key_is_a_clear_error(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(L, "NIM_KEY_FILE", tmp_path / "missing")
    with pytest.raises(L.LLMError, match="no NIM key"):
        L.NimClient()


def test_the_key_file_is_read_when_the_variable_is_absent(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    (tmp_path / "key").write_text(KEY + "\n")
    monkeypatch.setattr(L, "NIM_KEY_FILE", tmp_path / "key")
    assert L.nim_key() == KEY


def test_the_factory_defaults_to_the_local_model(monkeypatch) -> None:
    monkeypatch.delenv("RIC_LLM_PROVIDER", raising=False)
    assert isinstance(L.client_from_env(), L.OllamaClient)
    monkeypatch.setenv("RIC_LLM_PROVIDER", "nim")
    monkeypatch.setenv("NVIDIA_API_KEY", KEY)
    assert isinstance(L.client_from_env(), L.NimClient)
    monkeypatch.setenv("RIC_LLM_PROVIDER", "cloud")
    with pytest.raises(L.LLMError, match="unknown"):
        L.client_from_env()


def test_ollama_counts_its_tokens_too(monkeypatch) -> None:
    c = L.OllamaClient(model="qwen3:8b")
    monkeypatch.setattr(c, "_post", lambda body: {"message": {"content": "ok"}, "prompt_eval_count": 30, "eval_count": 4})
    reply = c.chat([{"role": "user", "content": "x"}])
    assert (reply.prompt_tokens, reply.completion_tokens) == (30, 4) and c.usage.calls == 1


def test_the_request_ceiling_stops_before_sending(client, monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(L.urllib.request, "urlopen",
                        lambda request, timeout: sent.append(request) or io.BytesIO(json.dumps(answer("ok")).encode()))
    client.max_requests = 2
    client.chat([{"role": "user", "content": "x"}])
    client.chat([{"role": "user", "content": "x"}])
    with pytest.raises(L.LLMError, match="ceiling"):
        client.chat([{"role": "user", "content": "x"}])
    assert len(sent) == 2 and client.requests == 2


def test_retries_count_against_the_ceiling(client, monkeypatch) -> None:
    monkeypatch.setattr(L.urllib.request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(http_error(429)))
    client.max_requests = 3
    with pytest.raises(L.LLMError, match="ceiling"):
        client.chat([{"role": "user", "content": "x"}])
    assert client.requests == 3
