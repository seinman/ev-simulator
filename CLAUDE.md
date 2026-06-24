# CLAUDE.md

Guidelines for working in this codebase.

## Project purpose

EV user simulation with a Streamlit dashboard. Portfolio piece for an energy flexibility
start-up — production-grade code, test coverage, and clean architecture are non-negotiable.
PLANNING.md is the source of truth for all architectural and simulation design decisions.

## Source data — DO NOT MODIFY

`data/input/ev_driver_archetypes.csv` is provided directly by the company and must not be
modified under any circumstances unless the user explicitly instructs it. All archetype
parameters are loaded from this file at runtime; do not hardcode values from it elsewhere.

## Dependency management and tooling

- **Poetry** for all dependencies. Do not use pip directly.
- **Docker** single container deployment. One service, no database.
- `main.py` in the project root is a PyCharm placeholder — ignore it, it will be replaced.

## Code style

- Production-grade Python. Type hints throughout.
- No comments explaining what the code does — use well-named identifiers instead.
- Comments only for non-obvious invariants or constraints (see simulation invariants below).
- No docstrings beyond a single short line where genuinely useful.

## Architecture constraints

- Simulation compute: vectorised NumPy / Polars. Do not use row-wise Python loops over users.
- Dashboard state: `st.session_state` caches simulation results. Charts operate on cached
  results; the simulation only re-runs on an explicit "Run" button press.
- `ChargingStrategy` enum (`IMMEDIATE` | `SMART_SCHEDULED`) distinguishes Intelligent Octopus
  behaviour — do not subclass `EVUser` for this.
- All days are IID in V1. No cross-day SOC tracking.

## Simulation invariants — enforce these everywhere

- Plug-in SoC is hard-capped at 80%. If a draw exceeds 80%, curtail to 80%; no charging occurs.
- kWh needed, required charge duration, and (for IO users) the smart-scheduling window are all
  derived from the drawn plug-in SoC — never drawn independently.
- Type 3 (Infrequent chargers): plug-in is a Bernoulli(p=0.2) draw each day. Plug-in SoC is
  only drawn conditional on a plug-in event occurring.
- Type 4 (Infrequent drivers): plugs in every day (frequency=1.0). Parameters taken directly
  from the company CSV without modification.
- Type 6 (Always plugged-in): excluded from V1.
- Intelligent Octopus (Type 2): no emergency SoC override in V1. Smart scheduling always used.
- Population and per-archetype stats are verified within 5% of targets; max 3 re-draw attempts
  before raising an error.

## Stochastic draw parameters

| Parameter | Spread |
|-----------|--------|
| Plug-in time | Normal; 90% of draws within ±1 hour of archetype mean |
| Plug-out time | Normal; 90% of draws within ±1 hour of archetype mean |
| Plug-in SoC | Normal; 90% of draws within ±5% of archetype mean |

All random draws use a single seeded RNG passed at run-time. Never instantiate `np.random`
without the project-level seed.

## Testing

Unit tests are required. Cover at minimum:
- SoC cap invariant: no draw exceeds 80%
- Type 3 Bernoulli plug-in logic
- IO smart scheduling selects the correct lowest-price contiguous window
- Verifier correctly passes, fails, and retries

## Out of scope (V1)

Do not implement: geographic segmentation, individual-level parameter heterogeneity,
per-user drill-down, V2G discharge, multi-week simulation, real-time price feeds.
