from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ev_simulator.constants import (
    HALF_HOUR_PERIODS_PER_DAY,
    N_SIMULATION_DAYS,
    PLUG_IN_SOC_SIGMA,
    PLUG_TIME_SIGMA_HOURS,
    SOC_CAP,
)
from ev_simulator.models.archetype import ARCHETYPES, ArchetypeConfig, ChargingStrategy
from ev_simulator.models.user import EVUser
from ev_simulator.simulation.verifier import (
    MAX_ATTEMPTS,
    ArchetypeVerifier,
    PopulationVerifier,
    VerificationError,
)

N_PERIODS = N_SIMULATION_DAYS * HALF_HOUR_PERIODS_PER_DAY  # 336

_PRICES_PATH = (
    Path(__file__).parent.parent.parent / "data" / "input" / "prices_apx_2025.csv"
)

_SEASON_MONTHS: dict[str, set[int]] = {
    "Spring": {3, 4, 5},
    "Summer": {6, 7, 8},
    "Autumn": {9, 10, 11},
    "Winter": {12, 1, 2},
}


@dataclass
class SimulationResult:
    users: list[EVUser]
    plugged_in: np.ndarray   # (N_users, 336) bool
    charging_kw: np.ndarray  # (N_users, 336) float
    soc: np.ndarray          # (N_users, 336) float


class Simulator:
    def __init__(
        self,
        n_users: int,
        seed: int,
        archetypes: tuple[ArchetypeConfig, ...] = ARCHETYPES,
        prices_path: Path = _PRICES_PATH,
        season: str | None = None,
    ) -> None:
        if not isinstance(n_users, int) or n_users <= 0:
            raise ValueError(f"n_users must be a positive integer, got {n_users!r}")
        if not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {type(seed).__name__!r}")
        if not archetypes:
            raise ValueError("archetypes must not be empty")
        if season is not None and season not in _SEASON_MONTHS:
            raise ValueError(f"season must be one of {list(_SEASON_MONTHS)}, got {season!r}")
        self.n_users = n_users
        self.seed = seed
        self.archetypes = archetypes
        self._rng = np.random.default_rng(seed)
        self._prices = self._load_prices(prices_path, season, seed)

    def run(self) -> SimulationResult:
        """Draw a population and simulate one representative week, retrying up to MAX_ATTEMPTS times."""
        pop_verifier = PopulationVerifier(self.archetypes)
        arch_verifier = ArchetypeVerifier(self.archetypes)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            users = self._draw_population()

            try:
                pop_verifier.verify(users)
            except VerificationError as exc:
                if attempt == MAX_ATTEMPTS:
                    raise VerificationError(
                        f"Population verification failed after {MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                continue

            plugged_in, charging_kw, soc = self._simulate_users(users)

            try:
                arch_verifier.verify(users, plugged_in)
            except VerificationError as exc:
                if attempt == MAX_ATTEMPTS:
                    raise VerificationError(
                        f"Archetype verification failed after {MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                continue

            return SimulationResult(
                users=users, plugged_in=plugged_in, charging_kw=charging_kw, soc=soc
            )

        raise VerificationError(f"Verification failed after {MAX_ATTEMPTS} attempts")  # unreachable

    def _simulate_users(
        self, users: list[EVUser]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(users)
        plugged_in = np.zeros((n, N_PERIODS), dtype=bool)
        charging_kw = np.zeros((n, N_PERIODS), dtype=float)
        soc = np.full((n, N_PERIODS), SOC_CAP, dtype=float)

        for archetype in self.archetypes:
            indices = np.array([u.user_id for u in users if u.archetype is archetype])
            if len(indices) == 0:
                continue
            self._simulate_archetype(indices, archetype, plugged_in, charging_kw, soc)

        return plugged_in, charging_kw, soc

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _draw_population(self) -> list[EVUser]:
        """Multinomial draw over archetype weights; returns list of EVUser ordered by archetype."""
        weights = np.array([a.population_weight for a in self.archetypes])
        weights /= weights.sum()
        counts = self._rng.multinomial(self.n_users, weights)
        users: list[EVUser] = []
        uid = 0
        for archetype, count in zip(self.archetypes, counts):
            for _ in range(count):
                users.append(EVUser(user_id=uid, archetype=archetype))
                uid += 1
        return users

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    @staticmethod
    def _load_prices(path: Path, season: str | None, seed: int) -> np.ndarray:
        """Returns (N_SIMULATION_DAYS, 48) price array in £/MWh.

        When season is given, a representative week is drawn at random (seeded) from all
        complete 7-day windows whose start date falls in that season. When season is None,
        the first 7 days in the price file are used."""
        if not path.exists():
            raise RuntimeError(
                f"Price data not found at '{path}'. "
                "Run `python scripts/fetch_prices.py` to download it. "
                "Zero-price fallback is intentionally disabled: Intelligent Octopus "
                "scheduling degenerates when all windows appear equally cheap."
            )
        try:
            df = pl.read_csv(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read price data at '{path}': {e}") from e

        required = {"settlement_date", "settlement_period", "price_gbp_per_mwh"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Price CSV at '{path}' is missing columns: {missing}. "
                "Re-run `python scripts/fetch_prices.py`."
            )

        if season is None:
            dates = df["settlement_date"].unique().sort().head(N_SIMULATION_DAYS)
        else:
            all_dates_in_file = {
                dt.date.fromisoformat(d)
                for d in df["settlement_date"].unique().to_list()
            }
            season_months = _SEASON_MONTHS[season]
            # Valid start dates: in the season AND have 6 following days anywhere in the file
            valid_starts = sorted(
                d for d in all_dates_in_file
                if d.month in season_months
                and all(
                    d + dt.timedelta(days=i) in all_dates_in_file
                    for i in range(1, N_SIMULATION_DAYS)
                )
            )
            if not valid_starts:
                raise RuntimeError(
                    f"No complete {N_SIMULATION_DAYS}-day window starting in {season} "
                    f"found in '{path}'. Try a different season or re-fetch prices."
                )
            # Separate RNG so season selection doesn't consume from the simulation stream
            week_rng = np.random.default_rng(seed ^ 0xCAFE_BABE)
            start = valid_starts[int(week_rng.integers(len(valid_starts)))]
            dates = pl.Series([
                (start + dt.timedelta(days=i)).isoformat()
                for i in range(N_SIMULATION_DAYS)
            ])

        df_week = df.filter(pl.col("settlement_date").is_in(dates.to_list())).sort(
            ["settlement_date", "settlement_period"]
        )
        expected_rows = N_SIMULATION_DAYS * HALF_HOUR_PERIODS_PER_DAY
        if len(df_week) != expected_rows:
            raise RuntimeError(
                f"Price data has {len(df_week)} rows for the selected week "
                f"(expected {expected_rows}). Re-run `python scripts/fetch_prices.py`."
            )
        return df_week["price_gbp_per_mwh"].to_numpy().reshape(
            N_SIMULATION_DAYS, HALF_HOUR_PERIODS_PER_DAY
        )

    # ------------------------------------------------------------------
    # Per-archetype simulation
    # ------------------------------------------------------------------

    def _simulate_archetype(
        self,
        user_indices: np.ndarray,
        archetype: ArchetypeConfig,
        plugged_in: np.ndarray,
        charging_kw: np.ndarray,
        soc: np.ndarray,
    ) -> None:
        """Simulate all 7 days for every user in this archetype group, writing results in place."""
        for day in range(N_SIMULATION_DAYS):
            active = self._active_users(user_indices, archetype)
            if len(active) == 0:
                continue

            plug_in_soc, global_in, global_out, charge_start, charge_end = (
                self._draw_event(day, len(active), archetype)
            )
            needs_charge = plug_in_soc < SOC_CAP

            self._fill_plugged_in(active, global_in, global_out, plugged_in)
            self._fill_charging_kw(
                active, charge_start, charge_end, needs_charge, archetype.charger_kw, charging_kw
            )
            self._fill_soc(
                active, global_in, global_out, charge_start, charge_end,
                plug_in_soc, needs_charge, archetype, soc,
            )

    def _active_users(
        self, user_indices: np.ndarray, archetype: ArchetypeConfig
    ) -> np.ndarray:
        """Return the subset of users who plug in today; Bernoulli draw for stochastic archetypes."""
        if archetype.plug_in_frequency < 1.0:
            mask = self._rng.random(len(user_indices)) < archetype.plug_in_frequency
            return user_indices[mask]
        return user_indices

    def _draw_event(
        self,
        day: int,
        n: int,
        archetype: ArchetypeConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Draw plug-in time, plug-out time, and plug-in SoC for n users on a given day.
        Returns (plug_in_soc, global_in, global_out, charge_start, charge_end) as period indices."""
        plug_in_soc = np.clip(
            self._rng.normal(archetype.mean_plug_in_soc, PLUG_IN_SOC_SIGMA, n),
            0.0, SOC_CAP,
        )

        # Always-plugged-in archetype (plug_in=00:00, plug_out=23:59): skip stochastic time
        # draws and cover the full day so consecutive days meet at midnight with no gap.
        if archetype.mean_plug_in_hour == 0.0 and archetype.mean_plug_out_hour > 23.5:
            global_in = np.full(n, day * HALF_HOUR_PERIODS_PER_DAY, dtype=int)
            global_out = np.minimum(
                np.full(n, (day + 1) * HALF_HOUR_PERIODS_PER_DAY, dtype=int), N_PERIODS
            )
        else:
            in_hours = self._rng.normal(archetype.mean_plug_in_hour, PLUG_TIME_SIGMA_HOURS, n)
            out_hours = self._rng.normal(archetype.mean_plug_out_hour, PLUG_TIME_SIGMA_HOURS, n)

            in_periods = np.clip(np.floor(in_hours * 2).astype(int), 0, 47)
            out_periods = np.clip(np.floor(out_hours * 2).astype(int), 0, 47)
            overnight = out_periods < in_periods

            global_in = day * HALF_HOUR_PERIODS_PER_DAY + in_periods
            global_out = np.where(
                overnight,
                (day + 1) * HALF_HOUR_PERIODS_PER_DAY + out_periods,
                day * HALF_HOUR_PERIODS_PER_DAY + out_periods,
            )
            global_out = np.minimum(global_out, N_PERIODS)

        kwh_needed = (SOC_CAP - plug_in_soc) * archetype.battery_capacity_kwh
        charge_periods = np.floor(kwh_needed / archetype.charger_kw * 2).astype(int)
        # Guarantee ≥1 period for any user who needs charge: floor gives 0 when kwh_needed
        # < one half-hour's delivery, which causes _fill_soc to write SOC_CAP at plug-in
        # rather than the true drawn SoC (charge_start == charge_end → empty in_charge).
        needs_charge = plug_in_soc < SOC_CAP
        charge_periods = np.where(needs_charge, np.maximum(charge_periods, 1), charge_periods)

        if archetype.charging_strategy is ChargingStrategy.SMART_SCHEDULED:
            charge_start = self._find_cheapest_windows(
                day, global_in, global_out, charge_periods, plug_in_soc
            )
        else:
            charge_start = global_in.copy()

        charge_end = np.minimum(
            np.minimum(charge_start + charge_periods, global_out), N_PERIODS
        )
        return plug_in_soc, global_in, global_out, charge_start, charge_end

    # ------------------------------------------------------------------
    # IO smart scheduling
    # ------------------------------------------------------------------

    def _find_cheapest_windows(
        self,
        day: int,
        global_in: np.ndarray,
        global_out: np.ndarray,
        charge_periods: np.ndarray,
        plug_in_soc: np.ndarray,
    ) -> np.ndarray:
        """Sliding-window prefix-sum over the plug-in window to find the cheapest contiguous
        block of charge_periods half-hours. Falls back to immediate start if window is too tight."""
        next_day = (day + 1) % N_SIMULATION_DAYS
        # Extend prices across midnight so overnight windows are priced correctly
        extended = np.concatenate([self._prices[day], self._prices[next_day]])  # (96,)
        day_offset = day * HALF_HOUR_PERIODS_PER_DAY
        charge_start = global_in.copy()

        for i, (gi, go, cp, si) in enumerate(
            zip(global_in, global_out, charge_periods, plug_in_soc)
        ):
            if si >= SOC_CAP or cp <= 0:
                continue
            local_in = gi - day_offset
            local_out = go - day_offset
            window = local_out - local_in
            if cp >= window:
                continue  # not enough room; fall back to immediate start
            segment = extended[local_in:local_out]
            prefix = np.empty(len(segment) + 1)
            prefix[0] = 0.0
            np.cumsum(segment, out=prefix[1:])
            window_costs = prefix[cp:] - prefix[:-cp]
            charge_start[i] = gi + int(np.argmin(window_costs))

        return charge_start

    # ------------------------------------------------------------------
    # Array filling — vectorised via cumsum trick
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_plugged_in(
        active: np.ndarray,
        global_in: np.ndarray,
        global_out: np.ndarray,
        plugged_in: np.ndarray,
    ) -> None:
        """Mark plug-in windows using a +1/-1 indicator array followed by cumsum."""
        n = len(active)
        indicator = np.zeros((n, N_PERIODS), dtype=np.int8)
        rows = np.arange(n)
        np.add.at(indicator, (rows, global_in), 1)
        in_bounds = global_out < N_PERIODS
        np.add.at(indicator, (rows[in_bounds], global_out[in_bounds]), -1)
        plugged_in[active] |= np.cumsum(indicator, axis=1).astype(bool)

    @staticmethod
    def _fill_charging_kw(
        active: np.ndarray,
        charge_start: np.ndarray,
        charge_end: np.ndarray,
        needs_charge: np.ndarray,
        charger_kw: float,
        charging_kw: np.ndarray,
    ) -> None:
        """Fill constant charger_kw over each user's charging window using the cumsum trick."""
        cs = charge_start[needs_charge]
        ce = charge_end[needs_charge]
        ui = active[needs_charge]
        n = len(ui)
        if n == 0:
            return
        rows = np.arange(n)
        indicator = np.zeros((n, N_PERIODS), dtype=np.float32)
        np.add.at(indicator, (rows, cs), charger_kw)
        in_bounds = ce < N_PERIODS
        np.add.at(indicator, (rows[in_bounds], ce[in_bounds]), -charger_kw)
        charging_kw[ui] += np.cumsum(indicator, axis=1)

    @staticmethod
    def _fill_soc(
        active: np.ndarray,
        global_in: np.ndarray,
        global_out: np.ndarray,
        charge_start: np.ndarray,
        charge_end: np.ndarray,
        plug_in_soc: np.ndarray,
        needs_charge: np.ndarray,
        archetype: ArchetypeConfig,
        soc: np.ndarray,
    ) -> None:
        """Write SoC trajectory via broadcasting: flat at plug_in_soc before charging,
        linear ramp during charging, flat at SOC_CAP after. Periods outside the session
        are left at their existing value (SOC_CAP from initialisation or a prior day's event)."""
        p = np.arange(N_PERIODS)
        gin = global_in[:, np.newaxis]
        gout = global_out[:, np.newaxis]
        cs = charge_start[:, np.newaxis]
        ce = charge_end[:, np.newaxis]
        si = plug_in_soc[:, np.newaxis]
        nc = needs_charge[:, np.newaxis]

        kwh_per_period = archetype.charger_kw * 0.5
        periods_charging = np.maximum(p - cs, 0)
        charging_soc = np.clip(
            si + periods_charging * kwh_per_period / archetype.battery_capacity_kwh,
            0.0, SOC_CAP,
        )

        in_session = (p >= gin) & (p < gout)
        in_charge = nc & (p >= cs) & (p < ce)
        post_charge = nc & (p >= ce) & (p < gout)

        session_soc = np.where(
            in_charge, charging_soc, np.where(post_charge, SOC_CAP, si)
        )
        soc[active] = np.where(in_session, session_soc, soc[active])
