"""FlexRICBackend: the OAI testbed behind the Backend interface (M07b).

The same RL runner, policy, verifier and agent run on it unchanged. What is different here:

- apply_prb_split turns the shares into whole PRBs and writes one line ("64 32 10") to the
  slice-control xApp (xapps/slice_ctrl), a child process that sends it to the gNB as an E2 SLICE
  control. It never waits for the answer, so the runner's tick cannot hang on E2 (invariant 5):
  the xApp's replies are read on a thread and logged.
- read_kpis reads the newest KPM values per slice from the store (kpm_latest, written by
  `xapps.kpm_monitor --for-runner`) and adds what only this class knows: the split in force
  while that report was measured. That gives prb_share_alloc and prb_used (the share of the
  slice's OWN PRBs it used).
- the clock is the wall clock: one tick is one real second.

If the xApp dies, the radio keeps the last split (the gNB holds it), the runner keeps running and
the failure is logged. If KPM reports stop, the last values are held and that is logged too.
"""

import json
import subprocess
import threading
import time
from collections import deque

from store import testbed
from store.events import log_event
from store.kpi import CellInfo, KpiRecord
from verifier.objective import SLICES

N_PRB = 106  # gNB carrier PRBs (testbed/rfsim-flexric/conf/gnb.conf); the gNB checks the ranges again
MIN_PRB = 5  # the gNB refuses fewer (its PF scheduler allocates at least 5 PRBs); 5 % of 106 = 5.3
REPORT_PERIOD_S = 1.0  # KPM granularity period (xapps/kpm/xapp_kpm_slices.c)
STALE_S = 5.0  # no new KPM values for this long: hold the last ones and log it
# The default command runs the xApp binary inside the RL xApp container (xapps/rl/Dockerfile).
XAPP_CMD = ["stdbuf", "-o0", "/usr/local/flexric/xApp/c/slice/xapp_slice_ctrl"]
# RRU.PrbTotDl of a fully used cell, as a share: it averages over EVERY slot, uplink ones included.
# The TDD pattern (gnb.conf) repeats every 10 slots: 7 downlink, 1 mixed with 6 of 14 symbols
# downlink, 2 uplink. A full downlink therefore reads (7 + 6/14) / 10 = 0.743; measured 0.73-0.75
# in M07a (stock) and M07b (each slice in its own range). Dividing by it gives the simulator's
# meaning of prb_used: 1.0 = every downlink PRB the slice may use is used.
PRB_FULL_CELL = (7 + 6 / 14) / 10


def shares_to_prbs(shares: list[float], n_prb: int = N_PRB, min_prb: int = MIN_PRB) -> list[int]:
    """Whole PRBs per slice, summing to n_prb, each at least min_prb.

    Largest remainder: floor every slice's exact count, then give the PRBs left over to the slices
    with the largest fractional parts (ties: SLICES order). The runner has already applied the
    5 % floor, so the min_prb top-up only moves a PRB when rounding lands just under it.
    """
    exact = [round(s * n_prb, 9) for s in shares]  # 0.6 * 106 = 63.599999999999994 otherwise
    prbs = [int(x) for x in exact]
    by_remainder = sorted(range(len(shares)), key=lambda i: (-(exact[i] - prbs[i]), i))
    for i in by_remainder[: n_prb - sum(prbs)]:
        prbs[i] += 1
    for i in range(len(prbs)):
        while prbs[i] < min_prb:
            richest = max(range(len(prbs)), key=lambda j: prbs[j])
            prbs[richest] -= 1
            prbs[i] += 1
    return prbs


class FlexRICBackend:
    """The real cell, through the RIC. One instance per runner process."""

    def __init__(self, conn, cell: CellInfo, xapp_cmd: list[str] | None = None) -> None:
        self.conn = conn
        self.cell = cell
        self.xapp = subprocess.Popen(xapp_cmd or XAPP_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        self.xapp_alive = True
        self.last_prbs: list[int] | None = None
        # (wall time sent, shares actually sent): to know the split in force when a report was measured
        self.history: deque[tuple[float, tuple[float, ...]]] = deque(maxlen=120)
        self.stale_logged = False
        threading.Thread(target=self._read_replies, daemon=True).start()

    def _read_replies(self) -> None:
        """Log each JSON line of the xApp (acks with their round-trip time, errors)."""
        assert self.xapp.stdout is not None
        for line in self.xapp.stdout:
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                continue  # FlexRIC's own log lines share stdout
            log_event("slice_ctrl_xapp", f"xapp_{reply.pop('event', 'unknown')}", data=reply)
        self.xapp_alive = False
        log_event("slice_ctrl_xapp", "xapp_exited", data={"returncode": self.xapp.wait()})

    def info(self) -> CellInfo:
        return self.cell

    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def apply_prb_split(self, shares: list[float]) -> None:
        prbs = shares_to_prbs(shares)
        if prbs == self.last_prbs:
            return  # the gNB keeps the last ranges; nothing to send
        if not self.xapp_alive or self.xapp.stdin is None:
            return  # logged once by the reader thread; the gNB keeps the last split it got
        try:
            self.xapp.stdin.write(" ".join(map(str, prbs)) + "\n")
            self.xapp.stdin.flush()
        except (BrokenPipeError, OSError):
            self.xapp_alive = False
            return
        self.last_prbs = prbs
        self.history.append((time.time(), tuple(p / N_PRB for p in prbs)))

    def share_in_force(self, wall: float) -> tuple[float, ...]:
        """The split the gNB was using at `wall`. Before the first command the gNB runs its stock
        scheduler over the whole cell, so every slice may use all of it: 1.0 each."""
        current: tuple[float, ...] = (1.0,) * len(SLICES)
        for sent, shares in self.history:
            if sent > wall:
                break
            current = shares
        return current

    def read_kpis(self) -> list[KpiRecord]:
        now = time.time()
        reports = testbed.read_latest(self.conn)
        targets = testbed.read_traffic_targets(self.conn)
        newest = max((r.report_wall for r in reports.values()), default=None)
        stale = newest is None or now - newest > STALE_S
        if stale != self.stale_logged:
            log_event("flexric_backend", "kpm_stale" if stale else "kpm_fresh",
                      data={"last_report_age_s": None if newest is None else round(now - newest, 2)})
            self.stale_logged = stale
        records = []
        for i, s in enumerate(SLICES):
            r = reports.get(s)
            if r is None:  # no report yet for this slice: nothing measured, say nothing was used
                r = testbed.SliceReport(s, now, 0.0, 0.0, None, 0)
            # The report covers the second before report_wall: use the split at its start.
            alloc = self.share_in_force(r.report_wall - REPORT_PERIOD_S)[i]  # >= 5/106, never 0
            records.append(KpiRecord(
                ts=now,
                source="flexric",
                cell_id=self.cell.cell_id,
                slice=s,
                dl_thr_mbps=r.dl_thr_mbps,
                prb_share_alloc=alloc,
                prb_used=r.prb_cell / PRB_FULL_CELL / alloc,
                n_ue=r.n_ue,
                latency_ms=r.latency_ms,
                offered_load_mbps=targets.get(s),
            ))
        return records

    def close(self) -> None:
        """Close the xApp's stdin: it finishes the command in flight and exits."""
        if self.xapp.stdin is not None:
            try:
                self.xapp.stdin.close()
            except OSError:
                pass
        try:
            self.xapp.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.xapp.kill()
