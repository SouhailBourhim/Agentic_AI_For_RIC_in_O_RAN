"""M04a monitoring: the exporter turns var/ric.db into Prometheus metrics, read-only."""

from pathlib import Path

from prometheus_client import CollectorRegistry, generate_latest

from conftest import run_agent, start_cell
from monitoring.exporter import RicCollector
from store.db import db_path


def scrape(path: Path) -> str:
    registry = CollectorRegistry()
    registry.register(RicCollector(path))
    return generate_latest(registry).decode()


def test_exporter_exposes_slices_policy_and_intents(ric_db, tiny_policy_file) -> None:
    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(10)
    run_agent(ric_db, "garantir 10 Mb/s au slice URLLC pendant 10 minutes", "eval")
    run_agent(ric_db, "make the network better", "eval")
    runner.run(2)
    text = scrape(db_path())
    for line in [
        "ric_backend_clock_seconds 12.0",
        'ric_policy_min_thr_mbps{slice="urllc"} 10.0',
        "ric_policy_in_force 1.0",
        'ric_served_model_info{version="tiny test policy"} 1.0',
        'ric_intents{state="applied"} 1.0',
        'ric_intents{state="rejected"} 1.0',
        'ric_verifier_decisions{check="ok"} 1.0',
    ]:
        assert line in text, line
    for metric in ["ric_slice_throughput_mbps", "ric_slice_latency_ms", "ric_slice_prb_share", "ric_slice_prb_used"]:
        assert sum(line.startswith(metric + "{") for line in text.splitlines()) == 3, metric


def test_exporter_is_read_only_and_quiet_without_a_run(tmp_path: Path) -> None:
    missing = tmp_path / "nothing.db"
    assert "ric_" not in scrape(missing)
    assert not missing.exists()  # it did not create the database


def test_exporter_serves_last_snapshot_when_the_database_is_busy(ric_db, tiny_policy_file, monkeypatch) -> None:
    """A failed read must not leave a hole in Prometheus: the previous values are served, counted."""
    import sqlite3

    start_cell(ric_db, tiny_policy_file).run(5)
    registry = CollectorRegistry()
    registry.register(RicCollector(db_path()))
    first = generate_latest(registry).decode()
    assert "ric_backend_clock_seconds 5.0" in first and "ric_exporter_db_errors_total 0.0" in first

    def busy(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(sqlite3, "connect", busy)
    second = generate_latest(registry).decode()
    assert "ric_backend_clock_seconds 5.0" in second  # the last good snapshot, not an empty scrape
    assert "ric_exporter_db_errors_total 1.0" in second


def test_exporter_counts_gateway_tool_calls(ric_db, tiny_policy_file) -> None:
    """M05a: the gateway's tool_call events become ric_gateway_tool_calls_total{tool,outcome}."""
    from store.events import log_event

    start_cell(ric_db, tiny_policy_file).run(2)
    for tool, outcome in [("submit_intent", "ok"), ("get_status", "ok"), ("get_status", "ok"),
                          ("get_kpis", "invalid_args"), ("get_kpis", "http_503"), ("get_kpis", "unreachable")]:
        log_event("gateway", "tool_call", data={"tool": tool, "outcome": outcome})
    log_event("agent", "received", "i0001")  # other sources are not counted
    text = scrape(db_path())
    for line in ['ric_gateway_tool_calls_total{outcome="ok",tool="submit_intent"} 1.0',
                 'ric_gateway_tool_calls_total{outcome="ok",tool="get_status"} 2.0',
                 'ric_gateway_tool_calls_total{outcome="invalid_args",tool="get_kpis"} 1.0',
                 'ric_gateway_tool_calls_total{outcome="error",tool="get_kpis"} 2.0',
                 # every series exists from the start, so Prometheus' increase() sees the first call
                 'ric_gateway_tool_calls_total{outcome="error",tool="submit_intent"} 0.0']:
        assert line in text, line
