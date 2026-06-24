# EV Simulator

This simulator aims to explore how individual EV drivers' SOC, plug-in and charging behaviour evolves over the course of a representative week. It allows the simulation of a large number of users drawn from different archetypes. One of the types, Intelligent Octopus, charges only during the cheapest slot over their plug-in time, while the others charge immediately upon plug-in. Slot prices are taken from BMRS.

Validation notebook found in `notebooks` folder.

## Running the simulator

### Streamlit Community Cloud

The live app is deployed via Streamlit Community Cloud. Click [here](https://ev-simulator-tdj8w2huknjiwffwf9mnbj.streamlit.app/) to access the app.

### Local (Poetry)

```bash
poetry install
python scripts/fetch_prices.py   # one-time price data fetch
streamlit run app.py
```

### Docker (self-hosting)

```bash
# Fetch price data before building (included in the image)
python scripts/fetch_prices.py

docker build -t ev-simulator .
docker run -p 8501:8501 ev-simulator
```

The app will be available at `http://localhost:8501`. A Dockerfile is provided for
portability; the canonical deployment is Streamlit Community Cloud.

## Simulation model

Each simulated day, every user (Type 3 excepted — see below) generates a plug-in event with
parameters drawn stochastically around their archetype's means:

| Parameter | Distribution | Spread |
|-----------|-------------|--------|
| Plug-in time | Normal, archetype mean | 90% of draws within ±1 hour |
| Plug-out time | Normal, archetype mean | 90% of draws within ±1 hour |
| Plug-in SoC | Normal, archetype mean | 90% of draws within ±5% |

### All days are independent (IID)

Each simulated day is drawn independently with no carry-over of SoC between days. This holds
because the end state of every day is deterministic: every user always reaches their target SoC
(80%) before unplugging, so every day begins from the same known state and can be drawn fresh.

The assumption rests on no archetype's charge duration ever exceeding their plug-in window. This
holds for all V1 archetypes — the worst case is Type 3 (Infrequent chargers), who may need ~5
hours of charging from 18% SoC, comfortably within their 13-hour window (18:00–07:00). If a
future version introduces stochastic charge durations long enough to occasionally overrun the
plug-in window, the IID assumption would break down and SoC would need to be carried forward
across days.

At present, no day-of-week seasonality is in the model. This would be an important extension for V2.

**Type 3 (Infrequent chargers):** a Bernoulli(p=0.2) draw determines whether a plug-in event
occurs at all on a given day; plug-in SoC is only drawn conditional on a plug-in occurring.

**Type 4 (Infrequent drivers):** plugs in every day but does all their driving on a single day
of the week. This implies six days of zero driving followed by one very long trip, which is
not realistic behaviour. These parameters are taken directly from the company's archetype
specification and are used as-is. A future extension would replace this archetype with a more
behaviourally coherent model — for example, a mixture distribution reflecting occasional
long trips interspersed with genuinely inactive days.

**SoC curtailment:** because plug-in SoC is drawn stochastically, a small number of users will
draw a value above the 80% target SoC (e.g. a short trip since last charge). These are curtailed
to 80% — no charging occurs for that event.

<!-- TODO: add charging logic summary, IO smart scheduling -->

## Design decisions

### No per-user parameter heterogeneity (V1)

All users within an archetype share identical parameters; stochasticity arises entirely from
per-event draws (plug-in time, plug-out time, plug-in SoC). This keeps V1 simple and avoids
the need to specify within-archetype variance for every variable. Individual-level heterogeneity
is a natural V2 extension — see below.

### Derived quantities, not independent draws

kWh needed, required charge duration, and (for Intelligent Octopus users) the smart-scheduling
window are all derived from the drawn plug-in SoC rather than drawn independently. This ensures
internal consistency within a plug-in event and avoids implausible combinations (e.g. a low
plug-in SoC paired with a short required charge duration).

### Intelligent Octopus as a charging strategy, not a subclass

Type 2 (Intelligent Octopus) users have identical plug-in behaviour to other archetypes; they
differ only in *when* charging occurs within the plug-in window. This is modelled via a
`ChargingStrategy` enum (`IMMEDIATE` | `SMART_SCHEDULED`) on the shared `EVUser` class rather
than a subclass, keeping the class hierarchy flat.

### `EVUser` as a typed extension point

`EVUser` is a thin dataclass wrapping `(user_id, archetype)`. In V1 it adds no individual-level
state — all parameters come from the archetype. It is kept as an explicit type because it is the
natural home for per-user parameter heterogeneity in V2 (personal battery size, habitual plug-in
time, etc.). Using a named type also keeps `SimulationResult` readable: `list[EVUser]` conveys
intent more clearly than a raw index array.

### `ChargingStrategy` enum over subclassing

Intelligent Octopus users differ from other archetypes only in *when* charging occurs within
their plug-in window, not in any structural behaviour. This is modelled via a `ChargingStrategy`
enum (`IMMEDIATE` | `SMART_SCHEDULED`) on the shared `EVUser` class rather than a subclass,
keeping the hierarchy flat and avoiding fragile inheritance. New strategies (e.g. V2G discharge,
demand-response curtailment) can be added as enum variants without touching existing code.

### No bump charging (V1)

In reality, drivers occasionally top up their battery opportunistically — a short charge of
roughly one hour at a workplace, public charger, or friend's house when SoC has dropped to a
low level. These *bump charges* are short, occur at semi-random times during the day, and are
triggered by SoC state rather than a fixed schedule.

V1 does not model bump charging for two reasons. First, it requires knowing the current SoC
at the moment of an ad-hoc plug-in, which in turn requires carrying SoC forward across days —
incompatible with the IID day assumption. Second, bump charge timing is poorly captured by a
fixed plug-in time distribution; it would need its own behavioural model (e.g. a threshold
trigger on SoC). See the extensions section for how this could be added in V2.

### Vectorised simulation

The simulation is fully vectorised over the user population using NumPy / Polars. At 5–10k
users this runs in well under a second, making in-process re-simulation from the Streamlit UI
practical without async or background workers.

## Architecture

```
ev_driver_archetypes.csv ──┐
                           ├──▶  Simulator  ──▶  SimulationResult
    prices_apx_2025.csv ───┘     (seed, N)       (N_users × 336 arrays:
                                    │              plugged_in, charging_kw, soc)
                                    │
                           PopulationVerifier / ArchetypeVerifier
                           (retry up to 3× on out-of-tolerance draws)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          compute_metrics()            compute_session_metrics()
          (plug-in rate, SoC,          (plug-in SoC histogram,
           flex headroom by bloc)       kWh topped, plug-in/out density)
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                           Streamlit dashboard
                        (result cached in session_state;
                         charts re-filter without re-running)
```

The simulation is fully vectorised over the user population: `plugged_in`,
`charging_kw`, and `soc` are `(N_users, 336)` NumPy arrays filled in one pass
per archetype per day using the cumsum trick. At 5–10k users the full run
completes in under a second, making in-browser re-simulation practical.

The dashboard separates compute from display: clicking **Run** populates
`st.session_state`, and the archetype filter operates on the cached result
without re-running the simulation.

---

## Design decisions

### Error handling strategy

Errors are handled according to severity and recoverability:

- **Archetype data (`ev_driver_archetypes.csv`) unreadable or malformed** — the simulation cannot
  run with a broken or incomplete archetype definition, so this raises immediately and loudly.
  Presenting a dashboard backed by a partial population would silently produce wrong results, which
  is worse than crashing.

- **Price data missing or corrupt** — the simulator raises a `RuntimeError` pointing the user to
  `scripts/fetch_prices.py`. Falling back to zero prices is not a safe default: with flat prices,
  Intelligent Octopus smart scheduling degenerates (every window looks equally cheap), producing
  results that appear valid but are meaningless. A missing price file is a configuration problem,
  not a runtime edge case.

- **Invalid simulation parameters** — `Simulator.__init__` validates inputs (e.g. `n_users > 0`,
  seed is an integer) and raises `ValueError` with a descriptive message. In the Streamlit context
  this is caught at the dashboard layer and surfaced as `st.error()` rather than a traceback.

- **Price fetch failures (`fetch_prices.py`)** — HTTP errors are retried with exponential backoff.
  If all retries fail, the script exits with a clear error message. The saved CSV is validated on
  write (expected row count, no missing settlement periods) so a partial download fails loudly
  rather than silently corrupting the cache.

### Price data source and the case for a Nord Pool fallback

Day-ahead prices are sourced from the [Elexon BMRS API](https://bmrs.elexon.co.uk/) (APXMIDP
provider) via `scripts/fetch_prices.py`. This is the only freely accessible, unauthenticated
source of historical half-hourly UK day-ahead prices:

- **ENTSO-E Transparency Platform** — ceased publishing GB data in June 2021 following Brexit; it
  explicitly defers to BMRS for UK prices.
- **Nord Pool** — publishes N2EX GB day-ahead prices at half-hourly resolution, which would make
  it a natural fallback. However, Nord Pool's data portal requires a paid annual subscription, so
  it is not wired in here. With a Nord Pool account, `fetch_prices.py` would attempt BMRS first,
  fall back to the Nord Pool REST API on failure, and surface a clear error only if both sources
  are unavailable. API credentials would be injected via environment variables (`NORD_POOL_API_KEY`)
  and documented in `.env.example`, with the Docker container configured to accept them at runtime.
  
### Population and archetype verification

After each population draw, two verifiers run before the simulation result is returned:

- **`PopulationVerifier`** checks that each archetype's fraction of the drawn population is
  within 10 percentage points of its target weight. This catches gross errors — wrong weights
  in the CSV, a bug in the multinomial draw — without false-positiving on natural statistical
  variance. A tighter relative tolerance (e.g. 5 %) would fail frequently at the fleet sizes
  this simulator supports (1 000–10 000 users): for a 10 % archetype at 1 000 users, 5 %
  relative gives only ±5 users of slack against a multinomial standard deviation of ~9.5 users.

- **`ArchetypeVerifier`** checks that the per-archetype empirical plug-in rate is within
  10 percentage points of the archetype's target `plug_in_frequency`. Only stochastic archetypes
  (`plug_in_frequency < 1.0`, i.e. Type 3) are checked — deterministic archetypes always plug in
  by construction. The check uses rising-edge detection (session *starts* per day) rather than
  "any plugged-in period that day", which would incorrectly count overnight sessions as plug-in
  events on the following day.

If either verifier fails, the simulator re-draws the population and re-runs, up to three
attempts total. If all three attempts fail, a `VerificationError` is raised. In practice,
failures should be extremely rare at normal fleet sizes — the verifiers are safety nets against
configuration bugs, not statistical tests.

---

## Motivation and extensions

### Uncertainty quantification on flexibility estimates

One of the core outputs of this simulator is an estimate of available flexibility at each half-hour
settlement period. However, reporting a point estimate alone understates the problem: Axle's
business depends on committing flexibility to markets or grid operators, where overcommitting
creates penalties and undercommitting leaves revenue on the table.

This simulator is well-placed to surface the *distribution* of flexibility, not just its mean.
Because individual plug-in behaviour is stochastic, running the simulation across multiple seeds
(or bootstrapping over the user population) can yield a confidence interval on fleet-level flexibility
at any given time step. Key questions this would help answer:

- At what confidence level can Axle commit X MWh of flexibility in a given half-hour window?
- How does uncertainty vary by time of day / archetype mix?
- How does the confidence interval scale with fleet size — i.e. how many EVs does Axle need to
  bring onto the platform before commitments become reliable?

### Individual-level parameter heterogeneity

Currently all users within an archetype share identical parameters; stochasticity comes from
simulation draws alone. A natural extension is to give each user a personal version of each
parameter (e.g. battery size, efficiency, habitual plug-in time) drawn from a distribution
centred on the archetype mean. This would better reflect real-world within-archetype variation
and affect the tails of flexibility distributions.

### Geographic segmentation and regional parameter correlation

Individual parameters may be correlated by region — e.g. rural users likely have larger
batteries and higher annual mileage; urban users may have lower charger availability or
different plug-in windows. Regional parameter distributions would also interact with grid
constraints (network capacity, local generation mix), which is directly relevant to Axle's
problem of dispatching flexibility within a constrained network.

### Dynamic and forecast day-ahead pricing

The simulator currently uses a static cached year of APX day-ahead prices (2025) sourced from
the Elexon BMRS API. This is sufficient for a representative fleet flexibility analysis, but
two extensions would make it more useful in a live operational context:

- **Daily price refresh**: the BMRS API (`/balancing/pricing/market-index`) is freely accessible
  with no authentication. A scheduled job could pull the latest day-ahead prices each morning
  and append them to the cached dataset, keeping the Intelligent Octopus smart scheduling
  window grounded in current market conditions.

- **Price forecasting**: for forward-looking flexibility commitments (e.g. bidding into a
  day-ahead flexibility market before prices are published), a price forecast model — even a
  simple seasonal baseline — could replace the static historical prices. This would allow Axle
  to estimate expected flexibility revenue before the auction clears.

### Bump charging and opportunistic top-ups

V1 models only one plug-in event per user per day, anchored to an archetype's habitual home
charging window. In practice, drivers also take short opportunistic charges — roughly one hour
at a workplace, public charger, or destination — when their SoC falls below a comfortable
threshold. These *bump charges* have a meaningfully different flexibility profile from overnight
sessions: they are shorter, occur during the day, are distributed across a wider range of times,
and are triggered by SoC state rather than a fixed routine.

Modelling bump charging requires two additions beyond V1. First, cross-day SoC tracking: the
simulator would need to carry each user's end-of-day SoC forward rather than resetting
independently each day, since bump charge likelihood depends on how much energy was consumed
since the last session. Second, a threshold-based trigger: a bump charge event fires with some
probability when a user's SoC drops below a threshold (e.g. 20%), with plug-in time drawn from
a daytime distribution rather than an evening one. This extension would increase the realism of
the intra-day flexibility signal, particularly for high-mileage archetypes who are more likely
to need opportunistic top-ups.

### Policy evaluation: coordinated vs uncoordinated smart charging

The counterfactual simulation (moving all users to smart charging) demonstrates the
synchronisation problem: uncoordinated smart scheduling shifts the peak rather than eliminating
it. This is the core business case for coordinated charging.

