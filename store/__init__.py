"""Shared state of the RIC processes: one SQLite file (var/ric.db) and one JSONL event log.

Every process (RL runner, agent service, operator console, Prometheus exporter) talks to the
others only through these tables, never directly. One module per table:
- kpi.py        KPI records written by the runner every tick, and the backend's cell info
- a1_policy.py  the objective timetable the runner follows (an emulation of O-RAN A1)
- intents.py    operator intents and their state
- events.py     the JSONL decision log
"""
