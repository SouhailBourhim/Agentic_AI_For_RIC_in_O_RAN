# Perspectives

Directions we might take later. Each entry says where the idea came from, what it would change, what it would cost, and how we would decide. **Nothing here is decided or started.** A perspective becomes work only through a checkpoint plan, approved like any other, and a line in [DECISIONS.md](DECISIONS.md).

## P-1. OCUDU (formerly the srsRAN Project) as the RAN (instead of, or next to, OAI)

**Origin.** Supervisor meeting, 2026-11-17: Pr. Benjillali suggested srsRAN as a better RAN than OpenAirInterface. We might still keep OAI.

**Which project to target: OCUDU.** The Linux Foundation took over the srsRAN Project code and launched it as **OCUDU** (the OCUDU Ecosystem Foundation, MWC, 2026-03-01). According to the sources below, the srsRAN Project itself is now deprecated, so we would build on OCUDU, the maintained code.
- **What it is:** an open-source 5G gNB with the full CU-CP, CU-UP and DU stack, on standard x86 or ARM machines. BSD-3-Clause. The name comes from the CU and the DU of a split gNB.
- **Who's behind it:** SRS (the original srsRAN team) and DeepSig wrote the initial software with US FutureG funding; large companies have joined since (Red Hat among them).
- **Releases:** "YY.MM" numbering. The latest found on 2026-11-17 is 26.10, which adds 3GPP Release 17 non-terrestrial (satellite) support and native 5G positioning.
- **Maturity:** one source says the release notes still list O-RU and end-to-end tests as outstanding.
- **Already used in labs:** the COSMOS testbed has tutorials with the OCUDU gNB, Open5GS and srsUE over ZMQ, single-node and cross-vendor. That's close to the spike below.
- **Same code lineage as srsRAN:** the srsRAN findings below (E2 control, multi-UE over ZMQ) are the best evidence we have, but each has to be re-checked on OCUDU.

These facts come mostly from industry news, not the foundation's own pages. Check ocudu.org and the OCUDU docs before relying on them.

**Status.** Open. Not decided, nothing started. First question for the supervisor: what "better" means here.
- **Standard E2 control** (E2SM-RC) instead of our custom gNB patch: a real gain, and worth the switch.
- **Code quality or real-time performance:** they matter less while the radio is simulated.
- **A later move to real SDR hardware** (USRP, see P-2): OCUDU, like srsRAN before it, is a good choice for that.
- **Maintenance and backing:** a Linux Foundation project with industry members and a release cycle.

**Today (OAI).** OAI gNB v3.0.0 (our build with two patches) + 3 OAI nrUEs over the RF simulator (TCP sockets) + OAI CN5G v2.2.1 + FlexRIC v3.0.0. Details in [M06](learn/M06-testbed.md), [M07a](learn/M07a-kpm.md), [M07b](learn/M07b-slice-control.md), [M10d](learn/M10d-pdcp-demand.md).

### What would not change

Everything above the `Backend` interface ([backends/base.py](../backends/base.py)): simulator, RL policy and training, verifier, LangGraph agent, MCP gateway, Hermes, console, store, MLOps, and almost all tests. FlexRIC can stay as the RIC. OAI CN5G can probably stay as the core, since N2 and N3 are standard interfaces. This is the design's own promise: swapping the backend needs no change in the upper layers.

### What would have to be redone

| Part | OAI today | With OCUDU | Unknowns |
|---|---|---|---|
| gNB + UEs | OAI gNB + 3 nrUEs over rfsim | OCUDU gNB + 3 srsUEs over ZeroMQ | Several UEs over ZMQ need a GNU Radio broker and network namespaces; srsUE's 5G SA support is a prototype |
| Per-slice KPM | Our patch for OAI's per-slice counter bug ([PR #1124](https://github.com/duranta-project/openairinterface5g/pull/1124)) | OCUDU's own E2SM-KPM | Whether the per-slice values are right: check against iperf3, as in M07a |
| PRB-split control | Our `slice-control.patch` + FlexRIC's SLICE service model: contiguous PRB ranges, hard cap | Standard O-RAN **E2SM-RC, "slice-level PRB quota"** (min/max/dedicated ratios) | A FlexRIC RC control decode problem was reported with srsRAN ([srsRAN_Project #468](https://www.github.com/srsran/srsRAN_Project/issues/468)), to re-check on OCUDU; ratio quotas aren't the same actuator as our ranges, so the simulator's cell model may need a variant |
| Measured demand (M10d) | Our PDCP volume patch (`DRB.PdcpSduVolumeDL`, remainder carried) | ? | Whether OCUDU reports it |
| Calibration | Simulator calibrated on OAI: 75 Mb/s, 0.654 Mb/s per PRB ([M09a](learn/M09a-sim-calibration.md)) | Measure again, new simulator profile | Observations are normalised by capacity, so the policy may need no retraining; to be checked |
| Evidence | C16, C20–C24 and the M09a calibration were measured on OAI ([CLAIMS](CLAIMS.md)) | Re-run to make the same claims on OCUDU | |

### Cost (estimate, 2026-11-17)

- **RAN side:** on OAI, going from the M06 spike to working PRB control over E2 (M07b) took 5 days (2026-11-01 to 11-05). Expect about **1–2 weeks** on OCUDU; the multi-UE ZMQ setup and the RC control path are the unknowns.
- **Testbed evidence:** about **1–2 more weeks** to re-run calibration, the sim-to-testbed check and the S1 runs, since testbed runs happen in real time.

### How we would decide

1. **A timeboxed spike, 2–3 days, like M06-spike.** Set up the OCUDU gNB (latest release) + 3 srsUEs over ZMQ + OAI CN5G + FlexRIC. Write the go/stop rules first:
   - per-slice KPM throughput matches iperf3 (the M07a check);
   - one E2SM-RC slice PRB quota command measurably changes the slices' throughput, and the gNB actually applies it, not just acknowledges it.
2. **Go:** add an OCUDU backend **next to** OAI rather than replacing it. The same agent, verifier and policy on two RAN stacks, with no change above the interface, would remove one "not claimed" item in CLAIMS (generality beyond one stack). Moving from a custom SLICE patch to the standard E2SM-RC action is also an improvement worth stating.
3. **Stop:** stay on OAI and record the reason in DECISIONS.md.

### Sources

OCUDU:
- [OCUDU docs](https://ocudu-docs-604e90.gitlab.io/)
- [Red Hat joins the OCUDU Ecosystem Foundation](https://www.redhat.com/en/blog/scaling-future-open-ran-red-hat-joins-ocudu-ecosystem-foundation)
- [The Mobile Network: OCUDU gets major vendor backing](https://the-mobile-network.com/?p=24604)
- [Runtime Wire: OCUDU 26.10 release](https://runtimewire.com/article/ocudu-26-10-satellite-5g-srs-stack)
- [DeepSig on OCUDU](https://www.deepsig.ai/ocudu-powering-wireless-innovation-from-nextg-prototypes-to-carrier-grade-deployment-through-open-source/)
- [Linux Foundation launches OCUDU](https://briefglance.com/companies/the-linux-foundation/pulses/47619)
- [RCR Wireless on OCUDU](https://www.rcrwireless.com/?p=429632)
- [SUSE: OCUDU reference configuration](https://documentation.suse.com/trd/telco/html/rc_suse-telco_ocudu/index.html)
- [COSMOS: OCUDU 5G SA cross-vendor tutorial](https://www.cosmos-lab.org/wiki/public/tutorials/cellular-4g5g-oran/ocudu-5g-sa-cross-vendor)

srsRAN (the same code lineage) and related work:
- [srsRAN gNB with srsUE (srsRAN docs)](https://docs.srsran.com/projects/project/en/latest/tutorials/source/srsUE/source/index.html)
- [srsRAN_Project issue #468: FlexRIC RC control with srsRAN](https://www.github.com/srsran/srsRAN_Project/issues/468)
- [OAIC: multiple UEs over ZMQ](https://openaicellular.github.io/oaic/MultipleUEs.html)
- [xDevSM (arXiv 2602.03821): KPM monitoring and RC slice PRB control across RAN stacks](https://arxiv.org/pdf/2602.03821)
- [arXiv 2609.18857: upstream OAI's E2 slice service model is an emulator](https://arxiv.org/pdf/2609.18857) (we replaced it with our own patch in M07b)

## P-2. Real radio with a USRP (instead of the RF simulator)

**Origin.** Discussion on 2026-11-17, after the supervisor meeting: if we get an SDR, can the testbed use a real radio instead of OAI's RF simulator?

**Status.** Open. Not decided, nothing started, and no SDR in hand yet. Questions to settle first: which USRP model (printed on the case, or shown by `uhd_find_devices`), which UEs and SIMs, and whether a native Linux PC is available.

**Today.** OAI's RF simulator: the gNB and the 3 nrUEs exchange time-domain IQ samples over TCP sockets (gNB at 192.168.70.151). The full PHY runs in software, and the channel is ideal: the `channelmod` section in the UE configs is not enabled. Details in [M06](learn/M06-testbed.md).

### What a USRP is

A family of software-defined radios from Ettus Research (NI), driven through the UHD library. It is the standard SDR for OAI and srsRAN. Models seen in 5G labs:

| Model | Link | Bandwidth | Fit for this project |
|---|---|---|---|
| B210 | USB 3 | up to 56 MHz, 2×2 | Usual lab choice, cheapest. Fine for one gNB at 20 MHz; 40 MHz (our 106 PRB) is at its limit. Needs native Linux (see below). |
| B200 / B200mini | USB 3 | up to 56 MHz, 1×1 | Smaller; enough for a single-antenna cell. |
| X310 | 10 GbE (or 1 GbE) | up to 160 MHz per channel | Much more headroom; Ethernet is easier to reach from a PC. Expensive. |
| N310 / N300 | 10 GbE | up to 100 MHz, 4×4 / 2×2 | Onboard CPU, built for 5G research. Most expensive. |

The gNB needs one USRP. The UEs need none if they are 5G modules or phones; only OAI's software nrUE would need one USRP per UE.

### What would not change

Everything above the gNB's radio: our two gNB patches (`kpm-per-ue.patch`, `slice-control.patch`) live in the MAC and the E2 agent, so they still apply. FlexRIC, the KPM and RL xApps, `FlexRICBackend`, the core, and every upper layer stay the same.

### What would change

| Part | rfsim today | With a USRP | Unknowns |
|---|---|---|---|
| gNB radio | `--rfsim` | UHD: drop `--rfsim`, set the SDR address and clock source in `gnb.conf` | Sustained real-time sample rate on our PC |
| UEs | 3 OAI nrUEs | 5G modules (e.g. Quectel RM500Q / RM520N) or n78 phones, each with a programmable SIM (e.g. sysmocom SJA2 / SJA5) written with our IMSIs and keys | Selecting DNN and S-NSSAI per UE: AT commands on modules, less control on phones |
| Bandwidth | 106 PRB, 30 kHz (40 MHz) | Start at 20 MHz (51 PRB) on a B210 | Policy inputs are normalised by capacity, so it may need no retraining; to be checked |
| Channel | Ideal | Real: HARQ, link adaptation, varying MCS | Removes the "ideal channel" item from what CLAIMS doesn't claim |
| Calibration | 75 Mb/s, 0.654 Mb/s per PRB ([M09a](learn/M09a-sim-calibration.md)) | Measure again; capacity will vary with the channel, closer to the M10 simulator profile | |
| KPM timing | One tick late (M07b) | Probably the same or worse | Re-check the smoothing alpha (0.10 today) |
| Evidence | C16, C20–C24 measured on rfsim | Re-run the testbed checks on the radio cell | |

### Constraints

1. **Spectrum licensing.** Transmitting over the air in n78 needs an authorisation (in Morocco, from the ANRT). Use a **cabled setup**: SDR → attenuators → UE, or a shielded box. It is also the usual lab practice and gives repeatable measurements.
2. **Not under WSL2.** A USB USRP would have to pass through `usbipd` into WSL2, which will very likely drop samples at the rates the gNB needs. OAI recommends native Ubuntu (22.04 for v3.0.0), a low-latency kernel and the CPU on the performance governor. In practice: dual-boot the laptop, or a separate Linux PC. An Ethernet USRP (X310, N310) is easier to reach.
3. **UEs and SIMs.** One UE per slice means three modules or phones and three programmable SIMs. Start with one UE on one slice.

### How we would decide

1. **A timeboxed spike, 2–3 days, with go/stop rules written first.** One cabled UE on one slice, on native Linux. Check two things:
   - the KPM throughput for that slice matches iperf3 (the M07a check);
   - one PRB-split command over E2 measurably changes the throughput (the M07b check).
2. **Go:** add the other two UEs, recalibrate, and add a radio testbed next to the rfsim one. Keep rfsim for CI and quick runs.
3. **Stop:** stay on rfsim and record the reason in [DECISIONS.md](DECISIONS.md).

**Link with P-1.** OCUDU, like srsRAN before it, is well documented with USRPs and commercial UEs. If we get a USRP and decide to try OCUDU, run the P-1 spike on the real radio from the start, so the two questions are answered with one setup.

## P-3. Realistic channels with Sionna (in the simulator)

**Origin.** Discussion on 2026-11-17, after the supervisor meeting: what NVIDIA's Sionna is, and whether it could make the simulated cell more realistic.

**Status.** Open. Not decided, nothing started. The version details below come from memory and secondary sources; check the official docs (nvlabs.github.io/sionna) and releases (github.com/NVlabs/sionna) before planning.

### What Sionna is

NVIDIA's open-source (Apache-2.0), GPU-accelerated Python library for simulating the radio physical layer. It is differentiable, so it is also used to train neural receivers end to end. Since version 1.0 (2025) it has three parts:

| Part | What it does |
|---|---|
| **Sionna PHY** | Link-level simulation: OFDM, 5G NR (LDPC, polar codes, PUSCH/PDSCH), MIMO, and the 3GPP channel models of TR 38.901 (TDL, CDL, UMi/UMa/RMa). |
| **Sionna SYS** | System level: SINR → throughput and errors (PHY abstraction), link adaptation, scheduling and power control, for many users and cells without simulating every bit. |
| **Sionna RT** | Differentiable ray tracing through a 3D scene (buildings, streets): propagation paths, coverage maps and channels at real positions. The basis of NVIDIA's radio "digital twin" work. |

**What it is not:** a RAN. There is no gNB protocol stack, no E2 and no core, so it can't replace OAI or OCUDU or sit behind FlexRIC. It is not real time either. It computes channels, and we would use it offline.

### Where it would fit

Today the M10 simulator profile draws each slice's channel before the run ([sim/channel.py](../sim/channel.py), [M10a](learn/M10a-varying-channel.md)):
1. a mean SINR per UE;
2. AR(1) shadowing (σ 6 dB, 50 m decorrelation, 3 or 30 km/h, from TR 38.901 UMa NLOS);
3. the TR 36.942 A.2 Shannon mapping to b/s/Hz;
4. normalised to a per-slice "rate per PRB" factor, second by second.

That trace is the natural place to plug Sionna in, and nothing above it would change:

| Option | What Sionna would produce | What it replaces |
|---|---|---|
| A. 3GPP stochastic channels | Per-UE SINR traces from TR 38.901 UMi/UMa with mobility (Sionna PHY) | Steps 1–2 (our AR(1) shadowing) |
| B. Validated link abstraction | SINR → throughput with real NR MCS tables and BLER (Sionna SYS) | Step 3 (the TR 36.942 Shannon bound) |
| C. A site-specific scene | Channels from a ray-traced street or campus (Sionna RT), UEs on paths through it | Steps 1–2, for one named place |

The traces would be generated offline, saved like the current ones, and replayed. The RL policy, the verifier, the agent and the evaluation harness would stay unchanged.

### Why it could matter

- **The realism question.** A reviewer is likely to ask whether the simulated cell is realistic. 3GPP channels from a widely used library answer that better than our own AR(1) model, without hardware. It is the simulation counterpart of P-2 (a real radio).
- **The M10 result.** M10b's difficulty gate failed (C18), and on the M10 cell the formula still beat RL (C19). A more realistic channel could make the cell harder, or easier. Either way it would test whether "the formula is enough" survives a better channel. That needs a new pre-registered test, not a re-score of C19.

### Cost and constraints

- **Effort (estimate, 2026-11-17):** option B alone is a small change (one mapping function and its tests). Option A is a few days: generate traces, check their statistics, add a simulator profile. Option C is the most work: a scene, UE paths, and checking the result against something measured.
- **GPU:** the RTX 5070 has about 6.6 GB of usable VRAM, and Ollama's qwen3:8b takes about 5.2 GB while it is loaded. Sionna jobs must run while the LLM is unloaded. That's fine offline, since traces are generated ahead of the runs.
- **Dependencies:** Sionna PHY and SYS run on TensorFlow, which the project doesn't use today (stable-baselines3 is on PyTorch). Keep it in a separate tool environment that only writes trace files, so the main environment doesn't take on TensorFlow.
- **Calibration:** keep the normalisation rule of M10a (the ensemble mean factor is 1), so the cell still carries its calibrated capacity on average.

### How we would decide

1. **A short spike, 1–2 days:** generate per-slice SINR traces with Sionna PHY (TR 38.901 UMa, the same speeds as M10), and compare their statistics with our AR(1) traces: mean, spread, correlation over time.
2. **Go:** add a simulator profile that replays them (option A), and B if the mapping matters. Then run a new pre-registered RL-vs-formula test on it.
3. **Stop:** keep the M10 channel and record the reason in [DECISIONS.md](DECISIONS.md).

### Sources

- [Sionna: an open-source library for 6G research (NVIDIA Developer)](https://developer.nvidia.com/sionna.md)
- [NVIDIA Sionna, PHY and SYS (overview)](https://www.emergentmind.com/topics/nvidia-sionna-phy-sys)
- [NVIDIA Sionna (overview, including the RT history)](https://api.emergentmind.com/topics/nvidia-sionna)
- [GCN: NVIDIA 6G research tools pass 1 million downloads, 8,000 labs (2026-11-10)](https://gcn.com/?p=23146)
- [Rohde & Schwarz: site-specific channel emulation with the Sionna Research Kit, MWC 2026](https://www.rohde-schwarz.com/lat/_229356-1611024.html)
- [NVIDIA open-sources Aerial software for AI-native 6G](https://blogs.nvidia.com/blog/open-source-aerial-ai-native-6g)
