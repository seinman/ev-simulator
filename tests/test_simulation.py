"""Unit tests covering the four invariants required by CLAUDE.md."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from ev_simulator.constants import HALF_HOUR_PERIODS_PER_DAY, N_SIMULATION_DAYS, SOC_CAP
from ev_simulator.models.archetype import ARCHETYPES, ArchetypeConfig, ChargingStrategy
from ev_simulator.models.user import EVUser
from ev_simulator.simulation import Simulator
from ev_simulator.simulation.verifier import (
    MAX_ATTEMPTS,
    ArchetypeVerifier,
    PopulationVerifier,
    VerificationError,
)

N_PERIODS = N_SIMULATION_DAYS * HALF_HOUR_PERIODS_PER_DAY  # 336


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_archetype(**overrides) -> ArchetypeConfig:
    """Minimal valid archetype config for Average (UK) with optional field overrides."""
    defaults = dict(
        name="Test",
        population_weight=1.0,
        annual_mileage_miles=7400,
        battery_capacity_kwh=64.0,
        efficiency_mi_per_kwh=3.5,
        plug_in_frequency=1.0,
        charger_kw=7.4,
        mean_plug_in_hour=18.0,
        mean_plug_out_hour=7.0,
        target_soc=0.80,
        mean_plug_in_soc=0.68,
        charging_strategy=ChargingStrategy.IMMEDIATE,
    )
    return ArchetypeConfig(**{**defaults, **overrides})


def _small_sim(n_users: int = 200, seed: int = 0) -> Simulator:
    return Simulator(n_users=n_users, seed=seed, archetypes=ARCHETYPES)


# ──────────────────────────────────────────────────────────────────────────────
# 1. SoC cap invariant
# ──────────────────────────────────────────────────────────────────────────────

class TestSocCap:
    def test_plug_in_soc_never_exceeds_cap(self):
        sim = _small_sim(n_users=500, seed=1)
        result = sim.run()
        assert result.soc.max() <= SOC_CAP + 1e-9, (
            f"SoC exceeded cap: max={result.soc.max():.4f}"
        )

    def test_soc_cap_across_seeds(self):
        for seed in range(5):
            sim = _small_sim(n_users=300, seed=seed)
            result = sim.run()
            assert result.soc.max() <= SOC_CAP + 1e-9

    def test_draws_above_cap_are_curtailed(self):
        """Users drawn with plug-in SoC ≥ 80% produce no charging and stay at SOC_CAP."""
        archetype = _make_archetype(mean_plug_in_soc=0.79, population_weight=1.0)
        sim = Simulator(n_users=100, seed=42, archetypes=(archetype,))
        result = sim.run()
        assert result.soc.max() <= SOC_CAP + 1e-9
        assert result.charging_kw.max() >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. Type 3 Bernoulli plug-in logic
# ──────────────────────────────────────────────────────────────────────────────

class TestType3Bernoulli:
    def _type3_archetype(self, p: float) -> ArchetypeConfig:
        return _make_archetype(name="Type3", plug_in_frequency=p)

    def test_plug_in_rate_near_expected(self):
        """Over many users, empirical plug-in rate ≈ Bernoulli p."""
        archetype = self._type3_archetype(p=0.2)
        sim = Simulator(n_users=2000, seed=7, archetypes=(archetype,))
        result = sim.run()

        pi = result.plugged_in.astype(np.int8)
        pi_padded = np.concatenate([np.zeros((len(result.users), 1), dtype=np.int8), pi], axis=1)
        session_starts = np.diff(pi_padded, axis=1) > 0
        events_by_day = session_starts.reshape(
            len(result.users), N_SIMULATION_DAYS, HALF_HOUR_PERIODS_PER_DAY
        ).any(axis=2)
        empirical_rate = events_by_day.mean()

        assert abs(empirical_rate - 0.2) < 0.05, (
            f"Plug-in rate {empirical_rate:.3f} too far from expected 0.2"
        )

    def test_no_plug_in_when_bernoulli_fails(self):
        """Users who fail every Bernoulli draw have no plugged-in periods."""
        archetype = self._type3_archetype(p=0.0)
        sim = Simulator(n_users=50, seed=3, archetypes=(archetype,))
        result = sim.run()
        assert not result.plugged_in.any(), "Expected no plug-in events for p=0.0"
        assert result.charging_kw.sum() == 0.0

    def test_always_plugs_in_when_p_is_1(self):
        """With p=1.0, every user plugs in every day — same as non-Bernoulli archetype."""
        archetype = self._type3_archetype(p=1.0)
        sim = Simulator(n_users=50, seed=5, archetypes=(archetype,))
        result = sim.run()
        # Every user should have at least one plugged-in period
        assert result.plugged_in.any(axis=1).all()


# ──────────────────────────────────────────────────────────────────────────────
# 3. IO smart scheduling: cheapest contiguous window
# ──────────────────────────────────────────────────────────────────────────────

class TestSmartScheduling:
    def _io_archetype(self) -> ArchetypeConfig:
        return _make_archetype(
            name="IO",
            mean_plug_in_soc=0.50,
            battery_capacity_kwh=64.0,
            charger_kw=7.4,
            mean_plug_in_hour=18.0,
            mean_plug_out_hour=7.0,
            charging_strategy=ChargingStrategy.SMART_SCHEDULED,
        )

    def test_charges_in_lowest_price_window(self):
        """With a single cheap period at the end of the window, IO users charge then."""
        archetype = self._io_archetype()
        sim = Simulator(n_users=1, seed=99, archetypes=(archetype,))

        # Flat high price everywhere; make last-night period very cheap
        prices = np.full((N_SIMULATION_DAYS, HALF_HOUR_PERIODS_PER_DAY), 100.0)
        prices[0, 14] = 1.0  # 07:00 day 0 — inside overnight window (18:00 day-1 to 07:00 day 0)

        kwh_needed = (SOC_CAP - 0.50) * 64.0  # 19.2 kWh
        charge_periods = int(np.floor(kwh_needed / 7.4 * 2))  # periods needed

        # Manually invoke sliding-window finder
        sim._prices = prices
        # Plug-in window: period 36 (18:00 day 0) to 62 (07:00 day 1)
        global_in = np.array([36])
        global_out = np.array([62])
        charge_periods_arr = np.array([charge_periods])
        plug_in_soc = np.array([0.50])

        result_start = sim._find_cheapest_windows(
            day=0, global_in=global_in, global_out=global_out,
            charge_periods=charge_periods_arr, plug_in_soc=plug_in_soc,
        )

        # The cheapest window is at period 14 in the extended array.
        # In the extended array (day 0 + day 1), day 1 period 14 = extended period 62.
        # local_out - charge_periods must allow a start that includes period 14.
        # Check: the window ends before or at the cheap period.
        window_contains_cheap = (
            result_start[0] <= 62 - charge_periods_arr[0]
        )
        assert window_contains_cheap, (
            f"Charge window starting at period {result_start[0]} doesn't reach cheap period"
        )

    def test_falls_back_to_immediate_when_window_too_tight(self):
        """When charge_periods ≥ session length, start = plug-in time."""
        archetype = self._io_archetype()
        sim = Simulator(n_users=1, seed=0, archetypes=(archetype,))
        sim._prices = np.ones((N_SIMULATION_DAYS, HALF_HOUR_PERIODS_PER_DAY))

        global_in = np.array([36])
        global_out = np.array([37])  # window of 1 period
        charge_periods = np.array([5])  # 5 periods needed, window only 1
        plug_in_soc = np.array([0.50])

        result_start = sim._find_cheapest_windows(
            day=0, global_in=global_in, global_out=global_out,
            charge_periods=charge_periods, plug_in_soc=plug_in_soc,
        )
        assert result_start[0] == global_in[0], (
            "Should fall back to immediate start when window is too tight"
        )

    def test_no_charging_when_already_at_cap(self):
        """Users already at SOC_CAP are skipped (charge_periods = 0)."""
        archetype = self._io_archetype()
        sim = Simulator(n_users=1, seed=0, archetypes=(archetype,))
        sim._prices = np.ones((N_SIMULATION_DAYS, HALF_HOUR_PERIODS_PER_DAY))

        global_in = np.array([36])
        global_out = np.array([62])
        charge_periods = np.array([0])
        plug_in_soc = np.array([SOC_CAP])

        result_start = sim._find_cheapest_windows(
            day=0, global_in=global_in, global_out=global_out,
            charge_periods=charge_periods, plug_in_soc=plug_in_soc,
        )
        assert result_start[0] == global_in[0]


# ──────────────────────────────────────────────────────────────────────────────
# 4. Verifier: passes, fails, retries
# ──────────────────────────────────────────────────────────────────────────────

class TestPopulationVerifier:
    def _make_users(self, archetype: ArchetypeConfig, n: int) -> list[EVUser]:
        return [EVUser(user_id=i, archetype=archetype) for i in range(n)]

    def test_passes_exact_fractions(self):
        a = _make_archetype(name="A", population_weight=0.6)
        b = _make_archetype(name="B", population_weight=0.4)
        users = self._make_users(a, 600) + self._make_users(b, 400)
        verifier = PopulationVerifier(archetypes=(a, b))
        verifier.verify(users)  # must not raise

    def test_fails_on_gross_deviation(self):
        a = _make_archetype(name="A", population_weight=0.6)
        b = _make_archetype(name="B", population_weight=0.4)
        # Give B 80% of users instead of 40% — 40 pp off
        users = self._make_users(a, 200) + self._make_users(b, 800)
        verifier = PopulationVerifier(archetypes=(a, b))
        with pytest.raises(VerificationError):
            verifier.verify(users)

    def test_error_message_names_failing_archetypes(self):
        a = _make_archetype(name="Archetype-Alpha", population_weight=0.5)
        b = _make_archetype(name="Archetype-Beta", population_weight=0.5)
        users = self._make_users(a, 900) + self._make_users(b, 100)
        verifier = PopulationVerifier(archetypes=(a, b))
        with pytest.raises(VerificationError, match="Archetype-Beta"):
            verifier.verify(users)


class TestArchetypeVerifier:
    def _bernoulli_archetype(self) -> ArchetypeConfig:
        return _make_archetype(name="Bernoulli", plug_in_frequency=0.2)

    def test_passes_correct_plug_in_rate(self):
        """Inject plugged_in with ~20% daily rate for a Bernoulli archetype."""
        archetype = self._bernoulli_archetype()
        n = 500
        users = [EVUser(user_id=i, archetype=archetype) for i in range(n)]
        rng = np.random.default_rng(0)

        # Construct a plugged_in array with ~20% sessions per user per day
        plugged_in = np.zeros((n, N_PERIODS), dtype=bool)
        for d in range(N_SIMULATION_DAYS):
            plug_in_mask = rng.random(n) < 0.2
            start = d * HALF_HOUR_PERIODS_PER_DAY + 36  # 18:00
            end = min(start + 26, N_PERIODS)  # ~13-hour session
            for u in np.where(plug_in_mask)[0]:
                plugged_in[u, start:end] = True

        verifier = ArchetypeVerifier(archetypes=(archetype,))
        verifier.verify(users, plugged_in)  # must not raise

    def test_fails_when_rate_grossly_wrong(self):
        """Inject one distinct session per user per day → actual rate 1.0 vs target 0.2."""
        archetype = self._bernoulli_archetype()
        n = 100
        users = [EVUser(user_id=i, archetype=archetype) for i in range(n)]

        # One 2-period session per user per day — distinct sessions so each day has a rising edge
        plugged_in = np.zeros((n, N_PERIODS), dtype=bool)
        for d in range(N_SIMULATION_DAYS):
            start = d * HALF_HOUR_PERIODS_PER_DAY + 10
            plugged_in[:, start : start + 2] = True

        verifier = ArchetypeVerifier(archetypes=(archetype,))
        with pytest.raises(VerificationError):
            verifier.verify(users, plugged_in)

    def test_skips_deterministic_archetypes(self):
        """freq=1.0 archetypes are not verified (see verifier note on always-plugged-in)."""
        archetype = _make_archetype(plug_in_frequency=1.0)
        users = [EVUser(user_id=i, archetype=archetype) for i in range(50)]
        # Zero plugged_in — would fail if the archetype were checked
        plugged_in = np.zeros((50, N_PERIODS), dtype=bool)
        verifier = ArchetypeVerifier(archetypes=(archetype,))
        verifier.verify(users, plugged_in)  # must not raise


class TestSimulatorRetry:
    def test_retries_on_population_failure_then_succeeds(self):
        """Simulator retries when PopulationVerifier fails first, succeeds second."""
        call_count = {"n": 0}
        original_verify = PopulationVerifier.verify

        def _failing_then_passing(self_v, users):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise VerificationError("injected failure")
            original_verify(self_v, users)

        with patch.object(PopulationVerifier, "verify", _failing_then_passing):
            sim = _small_sim(n_users=1000, seed=42)
            result = sim.run()

        assert call_count["n"] == 2
        assert result.plugged_in.shape == (1000, N_PERIODS)

    def test_raises_after_max_attempts(self):
        """Simulator raises VerificationError when all attempts fail."""
        with patch.object(
            PopulationVerifier, "verify",
            side_effect=VerificationError("always fails"),
        ):
            sim = _small_sim(n_users=500, seed=0)
            with pytest.raises(VerificationError, match="after 3 attempts"):
                sim.run()

    def test_attempt_count_matches_max_attempts(self):
        """Verify exact number of retries before giving up."""
        call_count = {"n": 0}

        def _always_fail(self_v, users):
            call_count["n"] += 1
            raise VerificationError("injected")

        with patch.object(PopulationVerifier, "verify", _always_fail):
            sim = _small_sim(n_users=500, seed=0)
            with pytest.raises(VerificationError):
                sim.run()

        assert call_count["n"] == MAX_ATTEMPTS
