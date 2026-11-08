"""The simulator's testbed profile (M09a): the cell as the OAI testbed showed it, from docs/learn/data/M09a-calibration.csv.

What changes against the M01 cell (sim/cell.py's defaults, kept so M01-M08a reproduce):
- delay: flat at the measured base until the slice saturates, then the queue's drain time (no M/M/1 term);
- buffer: about 400 Mb per slice instead of 50, as the testbed's standing queues showed;
- capacity: drawn per seed in the range measured for the study, since the testbed's rate varies with the split;
- load noise: iperf3's constant rate, almost none.
The KPIs reach the policy 2 ticks late: that is the backend (backends.sim.DelayedBackend), not the cell.
"""

import csv
from pathlib import Path

import numpy as np

from sim.cell import CellConfig
from verifier.objective import SLICES

CALIBRATION = Path("docs/learn/data/M09a-calibration.csv")
KPI_DELAY_TICKS = 2  # M07b
# True capacity, uniform per seed: M08c's backlogged seconds (10-90 %), M08d's measurement by split.
TRUE_CAPACITY_MBPS = {"m08c": (78.2, 84.7), "m08d": (73.5, 80.2), "m09": (73.5, 80.2)}
# What each study told the verifier and the policy: capacity, and the URLLC base latency the verifier sized with.
# M09b on: 79 Mb/s and the calibrated base latency (None = the cell's own), as decided after M09a.
TOLD = {"m08c": (69.0, 2.0), "m08d": (79.0, 2.0), "m09": (79.0, None)}
CAPACITY_STREAM = 9  # a separate random stream, so drawing the capacity doesn't shift the traffic's noise


def calibration(path: Path = CALIBRATION) -> dict[str, float]:
    with path.open() as f:
        return {row["parameter"]: float(row["value"]) for row in csv.DictReader(f)}


def testbed_cell(study: str, seed: int, path: Path = CALIBRATION) -> CellConfig:
    """The calibrated cell for replaying one testbed study, with this seed's capacity."""
    values = calibration(path)
    low, high = TRUE_CAPACITY_MBPS[study]
    told_capacity, told_urllc_ms = TOLD[study]
    return CellConfig(
        capacity_mbps=float(np.random.default_rng([seed, CAPACITY_STREAM]).uniform(low, high)),
        base_latency_ms={s: values[f"base_latency_ms_{s}"] for s in SLICES},
        max_backlog_mb=values["max_backlog_mb"],
        delay_model="flat",
        reported_capacity_mbps=told_capacity,
        reported_urllc_base_latency_ms=told_urllc_ms,
    )


def noise_std(path: Path = CALIBRATION) -> float:
    return calibration(path)["noise_std"]
