from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ev_simulator.constants import HALF_HOUR_PERIODS_PER_DAY, N_SIMULATION_DAYS
from ev_simulator.models.archetype import ArchetypeConfig
from ev_simulator.models.user import EVUser

# 10 pp absolute tolerance catches genuine bugs (wrong weights, zero-count archetypes,
# misapplied Bernoulli) without triggering on natural multinomial variance at the fleet
# sizes this simulator supports (1 000–10 000 users).
_TOLERANCE = 0.10

MAX_ATTEMPTS = 3


class VerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class PopulationVerifier:
    archetypes: tuple[ArchetypeConfig, ...]

    def verify(self, users: list[EVUser]) -> None:
        """Raise VerificationError if any archetype fraction deviates > 10 pp from target."""
        n = len(users)
        name_to_count: dict[str, int] = {}
        for u in users:
            name_to_count[u.archetype.name] = name_to_count.get(u.archetype.name, 0) + 1

        failures = []
        for archetype in self.archetypes:
            target = archetype.population_weight
            actual = name_to_count.get(archetype.name, 0) / n
            if abs(actual - target) > _TOLERANCE:
                failures.append(
                    f"{archetype.name}: target={target:.3f}, actual={actual:.3f}"
                )

        if failures:
            raise VerificationError(
                "Archetype fractions outside 10 pp tolerance:\n"
                + "\n".join(f"  {f}" for f in failures)
            )


@dataclass(frozen=True)
class ArchetypeVerifier:
    archetypes: tuple[ArchetypeConfig, ...]

    def verify(self, users: list[EVUser], plugged_in: np.ndarray) -> None:
        """Raise VerificationError if any archetype's plug-in rate deviates > 10 pp from target."""
        failures = []
        for archetype in self.archetypes:
            # Only Bernoulli archetypes have a stochastic plug-in rate worth verifying;
            # freq=1.0 archetypes always plug in by construction.
            if archetype.plug_in_frequency >= 1.0:
                continue
            indices = [u.user_id for u in users if u.archetype is archetype]
            if not indices:
                continue

            pi = plugged_in[np.array(indices)].astype(np.int8)  # (N_arch, N_PERIODS)
            # Count days on which a session STARTED (rising edge), not days with any plugged-in
            # period — overnight sessions would otherwise inflate the next day's count.
            pi_padded = np.concatenate(
                [np.zeros((len(indices), 1), dtype=np.int8), pi], axis=1
            )
            session_starts = np.diff(pi_padded, axis=1) > 0  # (N_arch, N_PERIODS)
            events_by_day = session_starts.reshape(
                len(indices), N_SIMULATION_DAYS, HALF_HOUR_PERIODS_PER_DAY
            ).any(axis=2)  # (N_arch, N_SIMULATION_DAYS) bool
            actual_rate = float(events_by_day.mean())
            target_rate = min(archetype.plug_in_frequency, 1.0)

            if abs(actual_rate - target_rate) > _TOLERANCE:
                failures.append(
                    f"{archetype.name}: target={target_rate:.3f}, actual={actual_rate:.3f}"
                )

        if failures:
            raise VerificationError(
                "Per-archetype plug-in rates outside 10 pp tolerance:\n"
                + "\n".join(f"  {f}" for f in failures)
            )
