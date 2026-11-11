"""All tunable settings in one place."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # Data
    start: str = "2016-01-01"
    end: str | None = None          # None: latest date in the FRED snapshot
    seed: int = 42
    # Corridor -> (FRED series of the currency per USD, currency code)
    corridors: tuple[tuple[str, str, str], ...] = (
        ("MX", "DEXMXUS", "MXN"), ("IN", "DEXINUS", "INR"), ("BR", "DEXBZUS", "BRL"),
    )
    dollar_index: str = "DTWEXBGS"  # broad trade-weighted U.S. dollar index

    # Simulated pricing policy (markup on the customer rate)
    markup_base: tuple[float, ...] = (0.008, 0.009, 0.010)   # MX, IN, BR
    markup_vol_sensitivity: float = 0.6   # extra markup per unit of 20-day daily FX volatility
    markup_bounds: tuple[float, float] = (0.005, 0.02)
    t_plus_2_share: float = 0.3           # share of trades that settle T+2 instead of T+1

    # Stage 1
    known_lag_days: int = 4         # a day's margin is certainly settled and known 4 days later
    sma_window: int = 28
    ewma_halflife: float = 10.0
    seasonal_window: int = 182

    # Stage 2 and backtest
    horizons: tuple[int, ...] = (1, 5)
    test_days: int = 730            # walk-forward over the last 2 years
    refit_every: int = 30           # refit stage 2 every 30 days
    mar_quantile: float = 0.05      # margin at risk: 5th percentile
    calibration_days: int = 180     # held-back recent days used to calibrate the band
    optuna_trials: int = 30
    cv_splits: int = 5

    reports_dir: Path = field(default=PROJECT_ROOT / "reports")
    fred_dir: Path = field(default=PROJECT_ROOT / "data" / "fred")


CONFIG = Config()
