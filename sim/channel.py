"""The radio channel of each slice's UE (M10a): how many Mb/s one PRB carries, second by second.

Before M10 every PRB of the cell carried the same rate (sim/cell.py: a slice's capacity is share x C). A real
cell doesn't work like that: each UE sees its own signal quality, which drifts as it moves, so the same share
gives a different rate per slice and per second. That is what makes "target / C" the wrong share, and what a
controller has to estimate from what it measures.

The model, one UE per slice as on the testbed:
1. Each slice's UE has a mean SINR (its distance to the gNB): uniform in MEAN_SINR_DB.
2. Around that mean, shadowing (buildings, as the UE moves): a Gaussian in dB with SHADOWING_SIGMA_DB, correlated
   in time as an AR(1) process. The correlation over one second is exp(-v dt / DECORRELATION_M): a UE at 3 km/h
   keeps its shadowing for about a minute, one at 30 km/h for about 6 s. Values from 3GPP TR 38.901, Table 7.5-6
   (UMa NLOS: sigma 6 dB, decorrelation distance 50 m).
3. SINR -> spectral efficiency (b/s/Hz) by the attenuated, truncated Shannon bound of 3GPP TR 36.942, Annex A.2
   (downlink: alpha 0.6, nothing under -10 dB, at most 4.4 b/s/Hz).
4. Divided by MEAN_EFFICIENCY_BPS_HZ, the mean of step 3 over every UE and every second (all mean SINRs, the
   whole shadowing distribution). So the factor is 1 on average over the ensemble, and the cell still carries the
   capacity it is told (79 Mb/s) on average, while one slice's UE can sit at 0.7 and another at 1.25 for a whole
   run. It is not divided by each trace's own mean: that would use the future, and pull every slow UE back to 1.

A trace is drawn in full before the run, from its own random stream, so the load draws, the oracle's look-ahead
and every earlier result stay exactly as they were. Formulas and choices: docs/learn/M10a-varying-channel.md.
"""

from dataclasses import dataclass

import numpy as np

from verifier.objective import SLICES

MEAN_SINR_DB = (8.0, 18.0)  # each UE's mean, uniform: cell edge to well inside the cell
SHADOWING_SIGMA_DB = 6.0  # TR 38.901 Table 7.5-6, UMa NLOS
DECORRELATION_M = 50.0  # TR 38.901 Table 7.5-6, UMa NLOS
SPEEDS_KMH = (3.0, 30.0)  # pedestrian or vehicle, one each with probability 1/2
# TR 36.942 Annex A.2, downlink: throughput = alpha log2(1 + SINR) between SINR_MIN_DB and the cap.
SHANNON_ALPHA = 0.6
SINR_MIN_DB = -10.0
MAX_EFFICIENCY_BPS_HZ = 4.4
# Mean of spectral_efficiency over MEAN_SINR_DB x N(0, SHADOWING_SIGMA_DB^2), by quadrature (tests/test_m10a.py
# recomputes it). Change it whenever one of the four values above changes.
MEAN_EFFICIENCY_BPS_HZ = 2.6427


def spectral_efficiency(sinr_db: np.ndarray) -> np.ndarray:
    """TR 36.942 A.2: b/s/Hz a link can carry at this SINR. Under SINR_MIN_DB the link carries nothing."""
    sinr_db = np.asarray(sinr_db, dtype=np.float64)
    shannon = SHANNON_ALPHA * np.log2(1.0 + 10.0 ** (sinr_db / 10.0))
    return np.where(sinr_db < SINR_MIN_DB, 0.0, np.minimum(shannon, MAX_EFFICIENCY_BPS_HZ))


@dataclass(frozen=True)
class ChannelTrace:
    """Each slice's factor on the cell's Mb/s per PRB, one row per second. Also kept: the SINR it comes from, and
    each UE's mean SINR and speed, for the plots and the note."""

    factors: np.ndarray  # (seconds, 3): multiply share x C by it
    sinr_db: np.ndarray  # (seconds, 3)
    mean_sinr_db: np.ndarray  # (3,)
    speed_kmh: np.ndarray  # (3,)
    dt_s: float = 1.0

    def at(self, t_s: float) -> np.ndarray:
        """The 3 factors during the tick that starts at t_s. Past the end of the trace is an error, not a guess."""
        return self.factors[round(t_s / self.dt_s)]


def sample_channel(rng: np.random.Generator, seconds: int, dt_s: float = 1.0) -> ChannelTrace:
    """Draw one trace. Order of draws: the 3 mean SINRs, the 3 speeds, the 3 starting shadowings, then
    (seconds - 1) x 3 innovations. The start is drawn from the stationary distribution, so the factor has the
    same distribution at every second, from the first one on."""
    n_slices = len(SLICES)
    mean_sinr_db = rng.uniform(*MEAN_SINR_DB, size=n_slices)
    speed_kmh = rng.choice(SPEEDS_KMH, size=n_slices)
    rho = np.exp(-(speed_kmh / 3.6) * dt_s / DECORRELATION_M)  # correlation of the shadowing over one tick
    shadowing = np.empty((seconds, n_slices))
    shadowing[0] = rng.normal(0.0, SHADOWING_SIGMA_DB, size=n_slices)
    innovations = rng.standard_normal((max(seconds - 1, 0), n_slices))
    for k in range(1, seconds):  # AR(1): keeps the variance at sigma^2
        shadowing[k] = rho * shadowing[k - 1] + np.sqrt(1.0 - rho**2) * SHADOWING_SIGMA_DB * innovations[k - 1]
    sinr_db = mean_sinr_db + shadowing
    factors = spectral_efficiency(sinr_db) / MEAN_EFFICIENCY_BPS_HZ
    return ChannelTrace(factors=factors, sinr_db=sinr_db, mean_sinr_db=mean_sinr_db, speed_kmh=speed_kmh, dt_s=dt_s)
