from dataclasses import dataclass

from ev_simulator.models.archetype import ArchetypeConfig


@dataclass(frozen=True)
class EVUser:
    user_id: int
    archetype: ArchetypeConfig
