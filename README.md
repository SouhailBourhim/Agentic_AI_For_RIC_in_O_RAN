# Agentic AI for the RIC in O-RAN

INPT Smart ICT semester project — Souhail Bourhim, Ghita Boulmani. Supervisor: Pr. Mustapha Benjillali.

An operator states a goal in plain language. An LLM agent (LangGraph) turns it into an objective for a
goal-conditioned RL xApp (DDPG) that allocates PRBs between slices in the Near-RT RIC. A deterministic
verifier checks every objective before it is applied. Hermes Agent is only the operator chat interface,
behind an MCP gateway that exposes three tools. Testbed: OpenAirInterface (rfsim) + OAI 5G core + FlexRIC.

- Design: [`docs/Proposition_IA_agentique_RIC_ORAN_v2.pdf`](docs/Proposition_IA_agentique_RIC_ORAN_v2.pdf) (French)
- Rules for the coding agent: [`CLAUDE.md`](CLAUDE.md)
- Build order, one prompt per checkpoint: [`PROMPTS.md`](PROMPTS.md)
- Learning notes, one per checkpoint: `docs/learn/` (written as the project progresses)

Status: design done, implementation starts at checkpoint M00.
