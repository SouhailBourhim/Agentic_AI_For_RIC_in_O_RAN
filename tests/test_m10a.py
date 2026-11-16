"""M10a: a radio channel per slice (sim/channel.py), the measured efficiency features, and the "m10" profile.

Two kinds of tests. First, nothing earlier moves: with no channel every function gives exactly what it gave, and
static, RL only and the oracle replay the M09b runs byte for byte: compared by sha256 with the original files, whose
hashes are committed (tests/data/m09b_replay_sha256.json). static and the oracle replay everywhere, CI included; RL
only needs the served weights, which are not in git, so it replays locally only. Second, the channel
is what the note says it is, and training (SlicingEnv) and serving (backend + runner) see the same cell and build
the same observation from it.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from backends.sim import SimBackend
from eval import benchmark
from eval.generator import FAMILIES, dev_instances
from eval.sim_profiles import CHANNEL_SECONDS, channel_of
from eval.sim_profiles import testbed_cell as cell_of  # not test_*: pytest would collect it
from rl import m09
from rl.runner import observation_from_records, seen_loads, update_efficiency
from sim import channel as ch
from sim.cell import CellConfig, step_slice, step_slices_batch
from sim.env import OBS_SIZE_WITH_APPLIED, OBS_SIZE_WITH_EFFICIENCY, EfficiencyTracker, LoadHistory, SlicingEnv
from sim.traffic import TrafficPattern
from verifier.objective import DEFAULT_OBJECTIVE, SLICES

M09B_DEV = Path("eval/results/m09b/m09b_dev")
M09B_REPLAY = json.loads((Path(__file__).parent / "data" / "m09b_replay_sha256.json").read_text())["kpi_csv_sha256"]
SERVED = Path("var/policy_cache/served.json")


def _trace(seed: int = 0, seconds: int = 200) -> ch.ChannelTrace:
    return ch.sample_channel(np.random.default_rng(seed), seconds)


def _traffic() -> dict[str, TrafficPattern]:
    return {"embb": TrafficPattern("steady", 25.0, 25.0, noise_std=0.05),
            "urllc": TrafficPattern("spike", 8.0, 30.0, 20, 50, noise_std=0.05),
            "be": TrafficPattern("steady", 20.0, 20.0, noise_std=0.05)}


# ---------------------------------------------------------------- nothing earlier moves


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(controller: str, root: Path, policy: Path | None) -> None:
    """Instance 00 of each family, played now: its kpi.csv must hash to the original M09b run's."""
    for inst in dev_instances(FAMILIES, 1):
        work = benchmark.play(inst, controller, policy, None, root=root)
        assert sha256(work / "kpi.csv") == M09B_REPLAY[f"{inst.name}/{controller}"], f"{inst.name} {controller}"


@pytest.mark.parametrize("controller", ["static", "oracle"])
def test_m09b_runs_replay_byte_for_byte(controller: str, tmp_path: Path) -> None:
    """No policy file needed: these two replay in CI too."""
    replay(controller, tmp_path, None)


@pytest.mark.skipif(not SERVED.exists(), reason="needs the served policy (var/policy_cache, not in git): local only")
def test_m09b_rl_only_replays_byte_for_byte(tmp_path: Path) -> None:
    from eval.validate_sim import served_policy

    replay("rl_only", tmp_path, served_policy())


@pytest.mark.skipif(not (M09B_DEV / "f1_00").exists(), reason="needs the M09b dev runs on disk (local only)")
def test_the_committed_hashes_are_the_original_runs() -> None:
    """The references in tests/data are what M09b wrote, not a later run's output."""
    for inst in dev_instances(FAMILIES, 1):
        for controller in ("static", "rl_only", "oracle"):
            path = benchmark.work_of(inst, controller) / "kpi.csv"
            assert sha256(path) == M09B_REPLAY[f"{inst.name}/{controller}"], path


def test_an_efficiency_of_one_changes_nothing() -> None:
    cfg = cell_of("m09", 7)
    for share, load, backlog in [(0.3, 20.0, 0.0), (0.05, 30.0, 12.0), (0.6, 0.0, 0.0)]:
        assert step_slice(share, load, backlog, 0.6, cfg) == step_slice(share, load, backlog, 0.6, cfg, efficiency=1.0)
    shares = np.random.default_rng(0).dirichlet(np.ones(3), size=50)
    loads, backlog, base = np.array([30.0, 12.0, 25.0]), np.array([0.0, 5.0, 1.0]), np.array([0.6, 0.6, 0.6])
    old, new = step_slices_batch(shares, loads, backlog, base, cfg), step_slices_batch(shares, loads, backlog, base, cfg, 1.0)
    assert all(np.array_equal(old[k], new[k]) for k in old)


def test_batch_and_single_agree_with_a_channel() -> None:
    cfg = cell_of("m10", 3)
    factors = np.array([0.6, 1.3, 0.9])
    shares = np.array([[0.5, 0.2, 0.3]])
    loads, backlog = np.array([30.0, 12.0, 25.0]), np.array([0.0, 5.0, 1.0])
    out = step_slices_batch(shares, loads, backlog, np.array([0.6, 0.6, 0.6]), cfg, factors)
    for i in range(3):
        one = step_slice(shares[0, i], loads[i], backlog[i], 0.6, cfg, efficiency=factors[i])
        assert out["throughput_mbps"][0, i] == pytest.approx(one.throughput_mbps)
        assert out["latency_ms"][0, i] == pytest.approx(one.latency_ms)


def test_the_options_off_give_the_same_episode() -> None:
    """The M09h env with the M10a options off is the M09h env: same observations, same rewards."""
    def play(env: SlicingEnv) -> list:
        obs, _ = env.reset(seed=4)
        out, rng = [obs], np.random.default_rng(0)
        for _ in range(40):
            obs, reward, *_ = env.step(rng.uniform(0, 1, 3).astype(np.float32))
            out += [obs, reward]
        return out

    options = m09.training_options("m09", "on", applied=True)
    a = play(SlicingEnv(episode_steps=40, **options))
    b = play(SlicingEnv(episode_steps=40, channel=False, efficiency_features=False, **options))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_a_channel_leaves_the_load_draws_alone() -> None:
    """Spawned off the episode's generator, the channel draws nothing from it: same loads with or without."""
    def loads(env: SlicingEnv) -> list:
        env.reset(seed=11)
        return [env.step(np.array([0.4, 0.3, 0.3], dtype=np.float32))[4]["load_mbps"] for _ in range(60)]

    options = m09.training_options("m09", "on", applied=True)
    with_channel = SlicingEnv(episode_steps=60, channel=True, **options)
    assert loads(SlicingEnv(episode_steps=60, **options)) == loads(with_channel)
    assert with_channel.channel_trace is not None and len(with_channel.channel_trace.factors) == 60


def test_an_unknown_profile_is_refused() -> None:
    """It used to fall back to the M01 cell silently."""
    with pytest.raises(ValueError):
        m09.cell("m10x")
    with pytest.raises(ValueError):
        m09.env_options("calibrated", "on")
    assert m09.cell("m01") == CellConfig()


def test_the_m10_profile_is_the_m09_cell_plus_a_channel() -> None:
    for seed in (100000, 101005, 7):
        assert cell_of("m10", seed) == cell_of("m09", seed)  # same capacity draw: runs pair by seed
        assert channel_of("m09", seed) is None
        trace = channel_of("m10", seed)
        assert trace.factors.shape == (CHANNEL_SECONDS, 3)
        assert np.array_equal(trace.factors, channel_of("m10", seed).factors)  # a seed always gives its channel
    assert not np.array_equal(channel_of("m10", 1).factors, channel_of("m10", 2).factors)


def test_a_new_profile_needs_its_own_results_root() -> None:
    inst = dev_instances(["f5"], 1)[0]
    with pytest.raises(ValueError):
        benchmark.play(inst, "static", Path("unused.zip"), None, profile="m10")


# ---------------------------------------------------------------- the channel itself


def test_the_mean_efficiency_constant_is_the_ensemble_mean() -> None:
    """MEAN_EFFICIENCY_BPS_HZ recomputed on a grid: mean SINR uniform, shadowing Gaussian."""
    means = np.linspace(*ch.MEAN_SINR_DB, 401)
    shadow = np.linspace(-8 * ch.SHADOWING_SIGMA_DB, 8 * ch.SHADOWING_SIGMA_DB, 4001)
    weight = np.exp(-0.5 * (shadow / ch.SHADOWING_SIGMA_DB) ** 2)
    per_mean = [np.sum(weight * ch.spectral_efficiency(m + shadow)) / weight.sum() for m in means]
    assert np.trapezoid(per_mean, means) / (means[-1] - means[0]) == pytest.approx(ch.MEAN_EFFICIENCY_BPS_HZ, rel=1e-4)


def test_the_shannon_mapping_follows_tr_36_942() -> None:
    eff = ch.spectral_efficiency(np.array([-12.0, -10.0, 0.0, 10.0, 30.0]))
    assert eff[0] == 0.0  # under -10 dB nothing goes through
    assert eff[2] == pytest.approx(0.6)  # 0.6 log2(1 + 1)
    assert eff[3] == pytest.approx(0.6 * np.log2(11.0))
    assert eff[4] == ch.MAX_EFFICIENCY_BPS_HZ
    assert np.all(np.diff(eff) >= 0)


def test_the_factor_is_one_on_average_and_the_same_at_every_second() -> None:
    factors = np.array([_trace(seed, 60).factors for seed in range(2000)])  # (seeds, seconds, slices)
    assert factors.mean() == pytest.approx(1.0, abs=0.02)
    assert factors[:, 0].mean() == pytest.approx(factors[:, -1].mean(), abs=0.03)  # stationary from the start
    assert factors.min() >= 0.0 and factors.max() <= ch.MAX_EFFICIENCY_BPS_HZ / ch.MEAN_EFFICIENCY_BPS_HZ + 1e-12


@pytest.mark.parametrize("speed_kmh", ch.SPEEDS_KMH)
def test_the_shadowing_decorrelates_at_the_speed_of_the_ue(speed_kmh: float) -> None:
    seconds = 20000
    for seed in range(50):
        trace = _trace(seed, seconds)
        if speed_kmh in trace.speed_kmh:
            break
    i = int(np.flatnonzero(trace.speed_kmh == speed_kmh)[0])
    x = trace.sinr_db[:, i] - trace.mean_sinr_db[i]
    expected = np.exp(-(speed_kmh / 3.6) / ch.DECORRELATION_M)
    assert np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(expected, abs=0.01)
    assert x.std() == pytest.approx(ch.SHADOWING_SIGMA_DB, rel=0.15 if speed_kmh < 10 else 0.05)


def test_a_trace_ends_where_it_ends() -> None:
    with pytest.raises(IndexError):
        _trace(0, 10).at(10.0)


# ---------------------------------------------------------------- training and serving see the same cell


def test_the_backend_and_the_env_give_the_same_cell_and_observation_with_a_channel() -> None:
    """Twin of tests/test_store_backend.py with a channel and every feature: SlicingEnv (training) against
    SimBackend + the runner's observation (serving), same seed, same splits, same trace."""
    cfg, trace, traffic = cell_of("m10", 5), _trace(5), _traffic()
    env = SlicingEnv(cfg, episode_steps=100, history="on", applied_split=True, efficiency_features=True)
    env_obs, _ = env.reset(seed=3, options={"traffic": traffic, "objective": DEFAULT_OBJECTIVE, "channel": trace})
    backend = SimBackend(cfg, traffic, seed=3, channel=trace)
    told = CellConfig(n_prb=cfg.n_prb, capacity_mbps=cfg.told_capacity_mbps)
    history, tracker, applied = LoadHistory(), EfficiencyTracker(), np.full(3, 1 / 3)
    for split in np.random.default_rng(0).dirichlet(np.ones(3), size=80):
        records = backend.read_kpis()
        history.update(seen_loads(records))
        update_efficiency(tracker, records, told.capacity_mbps)
        obs = observation_from_records(records, DEFAULT_OBJECTIVE, told, history, applied, tracker)
        assert obs.shape == (OBS_SIZE_WITH_EFFICIENCY,) and obs == pytest.approx(env_obs, abs=1e-6)
        env_obs, _, _, _, info = env.step(split)
        backend.apply_prb_split(list(split))
        backend.sleep(1.0)
        applied = env.shares
        by_slice = {r.slice: r for r in backend.read_kpis()}
        for s in SLICES:
            assert by_slice[s].dl_thr_mbps == pytest.approx(info["throughput_mbps"][s])


def test_the_tracker_reads_the_true_factor_while_a_slice_sends() -> None:
    """In the simulator a sending slice's throughput / (share x PRB use) is exactly its rate per unit share."""
    cfg, trace = cell_of("m10", 9), _trace(9)
    backend = SimBackend(cfg, {s: TrafficPattern("steady", 20.0, 20.0, noise_std=0.0) for s in SLICES}, seed=0,
                         channel=trace)
    tracker = EfficiencyTracker()
    for t in range(50):
        backend.apply_prb_split([0.4, 0.3, 0.3])
        backend.sleep(1.0)
        update_efficiency(tracker, backend.read_kpis(), cfg.told_capacity_mbps)
        expected = trace.at(t) * cfg.capacity_mbps / cfg.told_capacity_mbps
        sending = trace.at(t) > 0
        assert tracker.latest[sending] == pytest.approx(expected[sending], rel=1e-9)


def test_an_idle_slice_keeps_its_estimate() -> None:
    tracker = EfficiencyTracker()
    values = lambda x: dict(zip(SLICES, x))  # noqa: E731
    tracker.update(values([30.0, 0.0, 10.0]), values([0.5, 0.3, 0.2]), values([1.0, 0.0, 0.8]), 79.0)
    assert tracker.latest[1] == 1.0 and tracker.average[1] == 1.0  # URLLC sent nothing: still the told value
    assert tracker.latest[0] == pytest.approx(30.0 / (0.5 * 79.0))
    assert tracker.average[0] == pytest.approx(0.2 * tracker.latest[0] + 0.8)
    assert tracker.features().shape == (6,)


def test_the_runner_appends_the_efficiency_after_the_applied_split() -> None:
    cfg = cell_of("m10", 1)
    backend = SimBackend(cfg, _traffic(), seed=0, channel=_trace(1))
    backend.sleep(1.0)
    records, history, tracker = backend.read_kpis(), LoadHistory(), EfficiencyTracker()
    history.update(seen_loads(records))
    update_efficiency(tracker, records, 79.0)
    applied = np.array([0.5, 0.3, 0.2])
    old = observation_from_records(records, DEFAULT_OBJECTIVE, cfg, history, applied)
    new = observation_from_records(records, DEFAULT_OBJECTIVE, cfg, history, applied, tracker)
    assert np.array_equal(new[:OBS_SIZE_WITH_APPLIED], old)  # every earlier policy reads exactly what it read
    assert np.array_equal(new[OBS_SIZE_WITH_APPLIED:], tracker.features())
    with pytest.raises(ValueError):
        observation_from_records(records, DEFAULT_OBJECTIVE, cfg, history, None, tracker)
    with pytest.raises(ValueError):
        SlicingEnv(history="on", efficiency_features=True)  # needs the applied split before it


def test_the_oracle_predicts_the_cell_with_its_channel() -> None:
    """What eval/oracle.py predicts for a held split (the copied generator, the trace) is what the cell then does."""
    cfg, trace = cell_of("m10", 2), _trace(2)
    backend = SimBackend(cfg, _traffic(), seed=4, channel=trace)
    split = np.array([0.45, 0.35, 0.2])
    rng = copy.deepcopy(backend.rng)
    loads = np.array([[backend.traffic[s].load_at(float(k), rng) for s in SLICES] for k in range(40)])
    queue, base = np.zeros(3), np.array([cfg.base_latency_ms[s] for s in SLICES])
    for k in range(40):
        predicted = step_slices_batch(split[None, :], loads[k], queue, base, cfg, trace.at(float(k)))
        queue = predicted["backlog_mb"][0]
        backend.apply_prb_split(list(split))
        backend.sleep(1.0)
        actual = np.array([r.dl_thr_mbps for r in backend.read_kpis()])
        assert predicted["throughput_mbps"][0] == pytest.approx(actual)
