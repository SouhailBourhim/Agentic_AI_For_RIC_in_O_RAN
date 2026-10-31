"""Watch the simulator: equal PRB split, URLLC traffic spike from t=10 s to t=20 s.

Run with:  uv run python -m sim.demo

With an equal split each slice gets about 33 Mb/s. URLLC normally offers 10 Mb/s; during
the spike it offers 40 Mb/s, so its queue builds up and its latency climbs. After the
spike the queue drains and latency comes back down.
"""

import numpy as np

from sim.env import SlicingEnv
from sim.scenarios import urllc_spike
from verifier.objective import SLICES


def main() -> None:
    traffic, objective = urllc_spike()
    env = SlicingEnv(episode_steps=30)
    env.reset(seed=0, options={"objective": objective, "traffic": traffic})
    equal_split = np.ones(len(SLICES), dtype=np.float32)

    print("   t | load Mb/s (eMBB URLLC BE) | thr Mb/s (eMBB URLLC BE) | latency ms (eMBB URLLC BE) | reward")
    for _ in range(30):
        _, reward, _, truncated, info = env.step(equal_split)
        load = " ".join(f"{info['load_mbps'][s]:5.1f}" for s in SLICES)
        thr = " ".join(f"{info['throughput_mbps'][s]:5.1f}" for s in SLICES)
        lat = " ".join(f"{info['latency_ms'][s]:7.1f}" for s in SLICES)
        print(f"{info['time_s']:4.0f} | {load:>25} | {thr:>24} | {lat:>26} | {reward:6.3f}")
        if truncated:
            break


if __name__ == "__main__":
    main()
