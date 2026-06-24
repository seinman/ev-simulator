from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ev_simulator.constants import HALF_HOUR_PERIODS_PER_DAY, N_SIMULATION_DAYS, SOC_CAP
from ev_simulator.simulation import SimulationResult

PERIODS_PER_BLOC = 4
N_BLOCS = HALF_HOUR_PERIODS_PER_DAY // PERIODS_PER_BLOC  # 12 blocs of 2 hours
N_PERIODS = N_SIMULATION_DAYS * HALF_HOUR_PERIODS_PER_DAY
BLOC_LABELS = [f"{h:02d}:00" for h in range(0, 24, 2)]


def _typical_day(arr: np.ndarray) -> np.ndarray:
    """(N, 336) → (N, 48): mean over days 1–6, excluding day 0.

    Day 0 has no preceding overnight session, so its 00:00–07:00 slots are unpopulated
    for archetypes with overnight sessions. Including it would suppress early-morning
    plug-in rates to 6/7 of their true steady-state value."""
    return arr.reshape(arr.shape[0], N_SIMULATION_DAYS, HALF_HOUR_PERIODS_PER_DAY)[:, 1:, :].mean(axis=1)


def _typical_day_nan(arr: np.ndarray) -> np.ndarray:
    return np.nanmean(
        arr.reshape(arr.shape[0], N_SIMULATION_DAYS, HALF_HOUR_PERIODS_PER_DAY)[:, 1:, :], axis=1
    )


def _to_blocs(arr: np.ndarray) -> np.ndarray:
    """(N, 48) → (N, N_BLOCS): mean within each 2-hour bloc."""
    return arr.reshape(arr.shape[0], N_BLOCS, PERIODS_PER_BLOC).mean(axis=2)


def _to_blocs_nan(arr: np.ndarray) -> np.ndarray:
    return np.nanmean(arr.reshape(arr.shape[0], N_BLOCS, PERIODS_PER_BLOC), axis=2)


@dataclass
class FleetMetrics:
    bloc_labels: list[str]
    plug_in_rate: np.ndarray       # (N_BLOCS,) 0–1, fraction of fleet connected
    mean_soc: np.ndarray           # (N_BLOCS,) 0–1, plugged-in users only
    p25_soc: np.ndarray            # (N_BLOCS,)
    p75_soc: np.ndarray            # (N_BLOCS,)
    fleet_charging_kw: np.ndarray  # (N_BLOCS,) total kW across fleet
    available_flex_kw: np.ndarray  # (N_BLOCS,) dispatchable upward headroom kW
    n_users: int
    peak_plug_in_pct: float
    peak_charging_mw: float
    peak_flex_mw: float


def compute_metrics(
    result: SimulationResult,
    archetype_filter: set[str] | None = None,
) -> FleetMetrics:
    indices = np.array([
        u.user_id for u in result.users
        if archetype_filter is None or u.archetype.name in archetype_filter
    ])
    if len(indices) == 0:
        raise ValueError("No users match the selected archetype filter.")

    plugged_in  = result.plugged_in[indices].astype(float)
    charging_kw = result.charging_kw[indices]
    soc         = result.soc[indices]
    charger_kw  = np.array([result.users[i].archetype.charger_kw for i in indices])

    soc_masked   = np.where(plugged_in.astype(bool), soc, np.nan)
    connected_kw = charger_kw[:, np.newaxis] * plugged_in  # max kW each user could draw

    pi_hh  = _typical_day(plugged_in)
    ck_hh  = _typical_day(charging_kw)
    soc_hh = _typical_day_nan(soc_masked)
    con_hh = _typical_day(connected_kw)

    pi_bloc  = _to_blocs(pi_hh)
    ck_bloc  = _to_blocs(ck_hh)
    soc_bloc = _to_blocs_nan(soc_hh)
    con_bloc = _to_blocs(con_hh)

    fleet_charging = ck_bloc.sum(axis=0)
    available_flex = (con_bloc - ck_bloc).sum(axis=0)

    # Suppress all-NaN slice warnings: expected for blocs where no users are plugged in.
    with np.errstate(all="ignore"):
        mean_soc = np.nanmean(soc_bloc, axis=0)
        p25_soc  = np.nanpercentile(soc_bloc, 25, axis=0)
        p75_soc  = np.nanpercentile(soc_bloc, 75, axis=0)

    return FleetMetrics(
        bloc_labels=BLOC_LABELS,
        plug_in_rate=pi_bloc.mean(axis=0),
        mean_soc=mean_soc,
        p25_soc=p25_soc,
        p75_soc=p75_soc,
        fleet_charging_kw=fleet_charging,
        available_flex_kw=available_flex,
        n_users=len(indices),
        peak_plug_in_pct=float(pi_bloc.mean(axis=0).max() * 100),
        peak_charging_mw=float(fleet_charging.max() / 1000),
        peak_flex_mw=float(available_flex.max() / 1000),
    )


@dataclass
class SessionMetrics:
    plug_in_soc: np.ndarray   # SoC at plug-in moment, all sessions (0–1)
    kwh_topped: np.ndarray    # kWh added, charging sessions only
    plug_in_hod: np.ndarray   # plug-in hour of day, non-truncated sessions
    plug_out_hod: np.ndarray  # plug-out hour of day, non-truncated sessions


def compute_session_metrics(
    result: SimulationResult,
    archetype_filter: set[str] | None = None,
) -> SessionMetrics:
    indices = np.array([
        u.user_id for u in result.users
        if archetype_filter is None or u.archetype.name in archetype_filter
    ])
    if len(indices) == 0:
        raise ValueError("No users match the selected archetype filter.")

    plugged_in = result.plugged_in[indices]

    # Detect session boundaries via +1/-1 edges on the padded plugged_in array
    n = len(indices)
    padded = np.zeros((n, N_PERIODS + 2), dtype=np.int8)
    padded[:, 1:N_PERIODS + 1] = plugged_in.astype(np.int8)
    diff = np.diff(padded, axis=1)

    local_in,  period_in  = np.where(diff ==  1)
    local_out, period_out = np.where(diff == -1)

    # Map local row indices back to global user ids for SoC and battery lookups
    global_user_in = indices[local_in]

    plug_in_soc  = result.soc[global_user_in, period_in]
    battery_caps = np.array([result.users[u].archetype.battery_capacity_kwh for u in global_user_in])
    kwh_topped   = np.maximum(SOC_CAP - plug_in_soc, 0.0) * battery_caps

    not_truncated = period_out < N_PERIODS
    plug_in_hod  = (period_in[not_truncated]  % HALF_HOUR_PERIODS_PER_DAY) * 0.5
    plug_out_hod = (period_out[not_truncated] % HALF_HOUR_PERIODS_PER_DAY) * 0.5

    return SessionMetrics(
        plug_in_soc=plug_in_soc,
        kwh_topped=kwh_topped[kwh_topped > 0],
        plug_in_hod=plug_in_hod,
        plug_out_hod=plug_out_hod,
    )
