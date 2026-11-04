"""M05b: hard invariants 1 and 2 for Hermes, checked on its configuration files.

CI cannot run Hermes, but what limits it is written in two files: which tools it may use
(config.yaml) and what it can reach (docker-compose.yml). The live checks are in
docs/learn/M05b-hermes.md section 5; these tests stop a later edit from widening either one.
"""

from pathlib import Path

import pytest
import yaml

HERMES = Path(__file__).resolve().parent.parent / "gateway" / "hermes"
THE_THREE_TOOLS = {"submit_intent", "get_kpis", "get_status"}


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((HERMES / "config.yaml").read_text())


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((HERMES / "docker-compose.yml").read_text())


def test_one_mcp_server_with_exactly_the_three_tools(config) -> None:
    assert list(config["mcp_servers"]) == ["ric"]
    server = config["mcp_servers"]["ric"]
    assert set(server["tools"]["include"]) == THE_THREE_TOOLS
    assert "exclude" not in server["tools"]
    assert server["tools"]["resources"] is False and server["tools"]["prompts"] is False
    assert server["sampling"]["enabled"] is False and server["elicitation"]["enabled"] is False
    assert "command" not in server  # HTTP only: Hermes starts no process for it


def test_no_hermes_general_tool_is_enabled(config) -> None:
    assert config["toolsets"] == ["mcp-ric"]
    assert config["platform_toolsets"] == {"cli": ["mcp-ric"]}
    assert config["tools"]["tool_search"]["enabled"] == "off"  # no tool_call bridge next to the three
    assert config["memory"] == {"memory_enabled": False, "user_profile_enabled": False}


def test_hermes_uses_the_agent_model_and_no_outside_address(config, compose) -> None:
    from agent.llm import DEFAULT_MODEL

    assert config["model"]["default"] == DEFAULT_MODEL
    assert config["model"]["base_url"] == "http://ollama.internal:11434/v1"
    assert config["mcp_servers"]["ric"]["url"] == "http://gateway-relay:8100/mcp"
    aliases = compose["services"]["ollama-relay"]["networks"]["hermes-only"]["aliases"]
    assert "ollama.internal" in aliases


def test_hermes_container_has_no_way_out_but_the_internal_network(compose) -> None:
    hermes = compose["services"]["hermes"]
    assert hermes["networks"] == ["hermes-only"]
    assert compose["networks"]["hermes-only"]["internal"] is True
    assert "ports" not in hermes and "network_mode" not in hermes and "extra_hosts" not in hermes
    assert hermes["cap_drop"] == ["ALL"] and hermes["user"] == "hermes"
    mounted = " ".join(hermes["volumes"])
    assert "docker.sock" not in mounted


def test_each_relay_forwards_one_port_to_one_address(compose) -> None:
    targets = {"gateway-relay": ("8100", "host.docker.internal:8100"),
               "ollama-relay": ("11434", "host.docker.internal:11434")}
    assert set(compose["services"]) == {"hermes", *targets}
    for name, (port, target) in targets.items():
        relay = compose["services"][name]
        assert relay["command"] == [f"TCP-LISTEN:{port},fork,reuseaddr", f"TCP:{target}"]
        assert "ports" not in relay


def test_nothing_joins_the_testbed_network(compose) -> None:
    assert set(compose["networks"]) == {"hermes-only", "outside"}
    assert "external" not in compose["networks"]["outside"]
    assert "oai-cn5g-public-net" not in (HERMES / "docker-compose.yml").read_text()


def test_gateway_accepts_only_its_own_names_and_the_relay() -> None:
    from starlette.testclient import TestClient

    from gateway.__main__ import host_check
    from gateway.server import mcp

    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    app = mcp.streamable_http_app(transport_security=host_check(["gateway-relay:8100"]))
    with TestClient(app, base_url="http://gateway-relay:8100") as client:
        assert client.post("/mcp", headers=headers, content="{}").status_code != 421
        assert client.post("/mcp", headers={**headers, "Host": "evil:8100"}, content="{}").status_code == 421
