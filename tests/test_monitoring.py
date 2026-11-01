"""M04a monitoring: the exporter turns var/ric.db into Prometheus metrics, read-only."""

from pathlib import Path

from prometheus_client import CollectorRegistry, generate_latest

from agent import intents as machine
from conftest import start_cell
from monitoring.exporter import RicCollector
from store.db import db_path


def scrape(path: Path) -> str:
    registry = CollectorRegistry()
    registry.register(RicCollector(path))
    return generate_latest(registry).decode()


def test_exporter_exposes_slices_policy_and_intents(ric_db, tiny_policy_file) -> None:
    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(10)
    machine.submit(ric_db, "garantir 10 Mb/s au slice URLLC pendant 10 minutes", machine.Settings(mode="eval"), 0.0)
    machine.submit(ric_db, "make the network better", machine.Settings(mode="eval"), 0.0)
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
