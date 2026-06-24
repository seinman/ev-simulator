# EV Simulator — Planning Document

## Context

Take-home project for an energy flexibility start-up. The core business problem is
quantifying how much flexibility the grid has from distributed assets like EVs —
particularly the ability to shift charging load in time (and eventually discharge).

---

## Archetypes

Source: `data/input/ev_driver_archetypes.csv`

| # | Name | Pop % | Miles/yr | Battery (kWh) | Charger (kW) | Plug-in | Plug-out | Target SoC | Plug-in SoC |
|---|------|--------|----------|---------------|--------------|---------|----------|------------|-------------|
| 1 | Average (UK) | 40% | 9,435 | 60 | 7 | 18:00 | 07:00 | 80% | 68% |
| 2 | Intelligent Octopus | 30% | 28,105 | 72.5 | 7 | 18:00 | 07:00 | 80% | 52% |
| 3 | Infrequent charging | 10% | 9,435 | 60 | 7 | 18:00 | 07:00 | 80% | 18% |
| 4 | Infrequent driving | 10% | 5,700 | 60 | 7 | 18:00 | 07:00 | 80% | 73% |
| 5 | Scheduled charging | 9% | 9,435 | 60 | 7 | 22:00 | 09:00 | 80% | 68% |
| 6 | Always plugged-in | 1% | 9,435 | 60 | 7 | 00:00 | 23:59 | 80% | 68% |

---

## Simulation Design

### Population generation

- Draw N users (default 5,000–10,000) from the archetype distribution using archetype population weights.
- Each user is assigned a *personal* version of each archetype variable by sampling from a
  distribution centred on the archetype mean with some per-archetype spread.
  - *The exact distributional form for each variable is TBD — owner to decide.*
- Personal parameters are fixed for the lifetime of a simulation run (individual heterogeneity).
- Weekly behaviour for each user is then drawn stochastically around their personal parameters.

### Time resolution

- One week of simulated time, at **30-minute** resolution (to align with UK half-hourly settlement
  periods and day-ahead price data).
- *TBD: whether to simulate multiple weeks and report one, or a single representative week.*

### Core outputs per user per time-step

- **Plugged in**: boolean — is the user currently connected to a charger?
- **Plug-in SoC**: state of charge at the moment of connection (%).
- **SoC trajectory**: SoC at each 30-min step while plugged in.
- **Charging power**: kW drawn at each step (0 if not plugged in or already at target).

### Reproducibility

- All random draws seeded via a single integer seed passed at run-time.
- Seed exposed in the UI so runs can be replayed exactly.

---

## Intelligent Octopus Smart Charging

Archetype 2 users do **not** begin charging immediately on plug-in. Instead:

1. The user's required charge duration is known (enough kWh to reach target SoC at charger kW).
2. Over the plug-in window, identify the contiguous block of half-hour periods whose total
   duration equals the required charge time and whose summed day-ahead prices are lowest.
3. Charging is scheduled to that block only.

**Price data**: UK day-ahead electricity prices, one representative historical year
(source TBD — likely Elexon/Nord Pool API or a cached CSV).  
The same price year is used for every simulation run unless overridden.

*Open question: do Intelligent Octopus users have any SoC floor below which they override
smart scheduling and charge immediately (emergency override)?*

---

## Dashboard Features

All charts should be filterable by archetype segment and (where applicable) across a
user-defined date/hour range within the simulated week.

### Confirmed panels

- **SOC distribution over the day** — average and interquartile range at each 2-hour bloc
- **Plug-in rate** — proportion of fleet plugged in at each time step
- **Active charging rate** — proportion of fleet actively drawing power at each time step
- **Available flexibility** — total MWh available for charge / discharge at each time step
- *Further panels TBD*

### Simulation controls (sidebar)

- Number of users (N)
- Random seed
- Archetype filter (run simulation for a subset of archetypes)
- *Possibly: price year selector for Intelligent Octopus behaviour*

---

## Architecture

| Concern | Decision | Rationale |
|---------|----------|-----------|
| Language | Python | — |
| Dependency management | Poetry | Production-grade, lockfile, virtualenv |
| Dashboard | Streamlit | Fast to build, good for data exploration |
| Containerisation | Single Docker container | Single-service app; no DB needed |
| Simulation compute | Vectorised NumPy / Polars | 5–10k users runs in <1s; no async needed |
| Simulation triggered from | Streamlit UI | Re-runs are cheap enough to do in-process |
| Persistence | `st.session_state` | Simulation results cached in session; only recompute on explicit "Run" button press. Chart controls operate on cached results without re-running. Parquet export is a future option. |
| Data resolution | 30-minute half-hourly | Aligns with UK settlement periods |

---

## Open Questions

**Resolved**

- Plug-in/plug-out times: stochastic (jitter around archetype mean); 90% of draws within ±1 hour. ✓
- Plug-in SOC: stochastic; 90% of draws within ±5% of archetype mean; hard-capped at 80% on draw. ✓
- kWh/plug-in, SoC requirement, and charging duration: all derived from the drawn plug-in SOC, not drawn independently. ✓
- All days IID in V1; weekday/weekend variation deferred to V2. ✓
- Plug-in events are self-contained (no cross-day SOC tracking), but SoC is never allowed to exceed 80%. ✓
- Type 3 (infrequent chargers): plug-in probability modelled as Bernoulli(p=0.2) each day; plug-in SOC drawn conditionally on a plug-in day occurring. ✓
- Type 4 (infrequent driving): plugs in every day; parameters used as-is from company CSV. Behaviour is internally inconsistent (daily plug-in but all miles driven in one day); noted in README as a known limitation. ✓
- Type 6 (always plugged-in): deferred to V2; excluded from V1. ✓
- Intelligent Octopus (Type 2): required charge duration calculated from drawn plug-in SOC, not archetype mean; used to find lowest-price contiguous window. ✓
- Emergency SoC override for IO users: not modelled in V1. ✓
- Verification script: checks population and per-archetype stats are within 5% of targets; max 3 re-draw attempts before raising an error. ✓

**Open**

- [x] Per-user parameter draws: dropped. All users take their archetype's parameters exactly. Stochasticity comes from simulation draws (plug-in times, SOC, etc.), not individual heterogeneity. Deferred to V2. ✓
- [x] Time horizon: single representative week (7 IID days). ✓
- [x] Price data: Elexon BMRS API, APXMIDP provider (N2EX has no data; APX is the active UK day-ahead provider). One year of 2023 half-hourly prices fetched via `scripts/fetch_prices.py` and cached to `data/input/prices_apx_2023.csv`. No API key required. ✓

---

## Code Architecture

### Core classes

- `ArchetypeConfig` — dataclass, one per archetype; holds population-level parameters + `ChargingStrategy` enum (`IMMEDIATE` | `SMART_SCHEDULED`)
- `EVUser` — individual user with personal parameters drawn from their archetype
- `Simulator` — draws N users via multinomial over archetype weights, runs simulation, stores results in `st.session_state`
- `PopulationVerifier` — checks overall fleet stats are within 5% of targets; max 3 attempts
- `ArchetypeVerifier` — checks per-archetype stats are within 5% of targets; max 3 attempts

### Testing

Unit tests are required — this is a portfolio piece. Cover at minimum:
- Parameter draws respect bounds (SOC cap, non-negative durations, etc.)
- Type 3 Bernoulli plug-in logic
- IO smart scheduling selects the correct lowest-price window
- Verifier correctly passes/fails/retries

---

## TODOs

- [ ] **Price week selection**: expose a season selector (Spring/Summer/Autumn/Winter) in the
  Simulator and Streamlit sidebar. A representative week is drawn at random (seeded) from all
  weeks in the price dataset that fall within the chosen season. Currently hardcoded to the
  first 7 days of the price file.

---

## Out of Scope (v1)

- Geographic component
- Individual user drill-down
- V2G (vehicle-to-grid) discharge modelling
- Multi-week / multi-year simulation
- Real-time price feeds
