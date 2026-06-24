N_SIMULATION_DAYS: int = 7
HALF_HOUR_PERIODS_PER_DAY: int = 48
SOC_CAP: float = 0.80

# 90% of draws within ±1 hour → σ = 1 / Φ⁻¹(0.95) ≈ 0.608 hours
PLUG_TIME_SIGMA_HOURS: float = 0.608

# 90% of draws within ±5% → σ = 0.05 / Φ⁻¹(0.95) ≈ 0.030
PLUG_IN_SOC_SIGMA: float = 0.030
