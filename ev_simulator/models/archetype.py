import csv
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

_CSV_PATH = Path(__file__).parent.parent.parent / "data" / "input" / "ev_driver_archetypes.csv"


class ChargingStrategy(Enum):
    IMMEDIATE = auto()
    SMART_SCHEDULED = auto()


@dataclass(frozen=True)
class ArchetypeConfig:
    name: str
    population_weight: float
    annual_mileage_miles: float
    battery_capacity_kwh: float
    efficiency_mi_per_kwh: float
    plug_in_frequency: float  # expected plug-in events per day; <1.0 triggers Bernoulli draw
    charger_kw: float
    mean_plug_in_hour: float  # hours since midnight, 0–24
    mean_plug_out_hour: float  # hours since midnight, 0–24; if < mean_plug_in_hour, session spans midnight
    target_soc: float  # 0–1
    mean_plug_in_soc: float  # 0–1
    charging_strategy: ChargingStrategy


def _parse_hour(time_str: str) -> float:
    t = datetime.strptime(time_str.strip(), "%I:%M %p")
    return t.hour + t.minute / 60


def _parse_percent(pct_str: str) -> float:
    return float(pct_str.strip().rstrip("%")) / 100


def load_archetypes(csv_path: Path = _CSV_PATH) -> tuple[ArchetypeConfig, ...]:
    if not csv_path.exists():
        raise RuntimeError(
            f"Archetype CSV not found at '{csv_path}'. "
            "This file is required; do not move or delete it."
        )
    archetypes: list[ArchetypeConfig] = []
    try:
        with csv_path.open(newline="") as f:
            for i, row in enumerate(csv.DictReader(f), start=2):
                try:
                    strategy_str = row["ChargingStrategy"].strip()
                    try:
                        strategy = ChargingStrategy[strategy_str]
                    except KeyError:
                        valid = [s.name for s in ChargingStrategy]
                        raise RuntimeError(
                            f"Unknown ChargingStrategy '{strategy_str}' in row {i}. "
                            f"Valid values: {valid}"
                        ) from None
                    archetypes.append(ArchetypeConfig(
                        name=row["Name"].strip(),
                        population_weight=_parse_percent(row["% of population"]),
                        annual_mileage_miles=float(row["Miles/yr"]),
                        battery_capacity_kwh=float(row["Battery (kWh)"]),
                        efficiency_mi_per_kwh=float(row["Efficiency (mi/kWh)"]),
                        plug_in_frequency=float(row["Plug-in frequency (per day)"]),
                        charger_kw=float(row["Charger kW"]),
                        mean_plug_in_hour=_parse_hour(row["Plug-in time"]),
                        mean_plug_out_hour=_parse_hour(row["Plug-out time"]),
                        target_soc=_parse_percent(row["Target SoC"]),
                        mean_plug_in_soc=_parse_percent(row["Plug-in SoC"]),
                        charging_strategy=strategy,
                    ))
                except RuntimeError:
                    raise
                except KeyError as e:
                    raise RuntimeError(
                        f"Missing column {e} in archetype CSV row {i}."
                    ) from e
                except ValueError as e:
                    raise RuntimeError(
                        f"Invalid value in archetype CSV row {i}: {e}"
                    ) from e
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to read archetype CSV at '{csv_path}': {e}") from e

    if not archetypes:
        raise RuntimeError(f"Archetype CSV at '{csv_path}' contains no data rows.")

    total_weight = sum(a.population_weight for a in archetypes)
    if not (0.99 <= total_weight <= 1.01):
        raise RuntimeError(
            f"Archetype population weights sum to {total_weight:.4f}, expected 1.0. "
            "Check the '% of population' column in the archetype CSV."
        )

    return tuple(archetypes)


ARCHETYPES: tuple[ArchetypeConfig, ...] = load_archetypes()
