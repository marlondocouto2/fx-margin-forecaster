"""Simulated daily payout volume per corridor.

The same generator as Project 2 (cash-flow-forecaster): trend x weekday x payday x annual x
holiday x scheduled spikes x random spikes x noise. Copied so this repo runs on its own;
Brazil is added as a corridor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import holiday_window

RAW_WEEKDAY_SHAPE = np.array([1.10, 1.00, 0.95, 1.00, 1.25, 0.80, 0.70])
WEEKDAY_FACTORS = RAW_WEEKDAY_SHAPE / RAW_WEEKDAY_SHAPE.mean()
NOISE_SIGMA = 0.06
SPIKE_PROB, SPIKE_RANGE = 3 / 30.44, (1.5, 3.0)
SCHEDULED_FIRST, SCHEDULED_FACTOR = "2021-01-08", 1.6
HOLIDAY_EFFECTS = {("major", -3): 1.25, ("major", -2): 1.40, ("major", -1): 1.55, ("major", 0): 0.60,
                   ("major", 1): 1.20, ("minor", 0): 0.60, ("minor", 1): 1.15}

# Corridor (receiving country): (daily volume in USD at the start date, annual growth)
CORRIDORS = {"MX": (25_000_000.0, 0.12), "IN": (15_000_000.0, 0.18), "BR": (10_000_000.0, 0.14)}


def _month_position(dates: pd.DatetimeIndex) -> np.ndarray:
    day = dates.day.to_numpy()
    eff = np.ones(len(dates))
    eff[day == 1] *= 1.25
    eff[day == 2] *= 1.12
    eff[day == 3] *= 1.05
    eff[day == 15] *= 1.20
    eff[(day == 14) | (day == 16)] *= 1.06
    month_end = dates + pd.offsets.MonthEnd(0)
    shift = np.where(month_end.day_of_week == 5, 1, np.where(month_end.day_of_week == 6, 2, 0))
    lbd = (month_end - pd.to_timedelta(shift, unit="D")).to_numpy()
    to_lbd = (lbd - dates.to_numpy()).astype("timedelta64[D]").astype(int)
    eff[to_lbd == 0] *= 1.30
    eff[to_lbd == 1] *= 1.08
    return eff


def _annual(dates: pd.DatetimeIndex) -> np.ndarray:
    doy = dates.day_of_year.to_numpy()
    dec20 = pd.to_datetime([f"{y}-12-20" for y in dates.year]).day_of_year.to_numpy()
    return (1 + 0.04 * np.sin(2 * np.pi * (doy - 80) / 365.25)) * (
        1 + 0.12 * np.exp(-0.5 * ((doy - dec20) / 6.0) ** 2))


def generate_volume(country: str, dates: pd.DatetimeIndex, seed: int) -> pd.Series:
    """Daily USD volume for one corridor over `dates` (calendar days)."""
    base, growth = CORRIDORS[country]
    rng = np.random.default_rng(seed)
    years = (dates - dates[0]).days.to_numpy() / 365.25
    window = holiday_window(dates, country)
    holiday = np.array([HOLIDAY_EFFECTS.get(k, 1.0)
                        for k in zip(window["holiday_tier"], window["holiday_offset"])])
    scheduled = np.where((dates - pd.Timestamp(SCHEDULED_FIRST)).days % 14 == 0, SCHEDULED_FACTOR, 1.0)
    spike = np.where(rng.random(len(dates)) < SPIKE_PROB, rng.uniform(*SPIKE_RANGE, len(dates)), 1.0)
    noise = rng.lognormal(0.0, NOISE_SIGMA, len(dates))
    vol = (base * (1 + growth) ** years * WEEKDAY_FACTORS[dates.day_of_week] * _month_position(dates)
           * _annual(dates) * holiday * scheduled * spike * noise)
    return pd.Series(vol, index=dates, name="volume")
