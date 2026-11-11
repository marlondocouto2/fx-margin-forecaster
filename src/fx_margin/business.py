"""Simulated business layer on top of real FX rates (clearly simulated; see the README).

For each corridor and calendar day t:

- Pricing policy: markup m_t = base + sensitivity x (20-day FX volatility up to t-1) + a slow
  mean-reverting drift + occasional step changes (pricing reviews), clipped to 0.5%-2%.
- Customer rate C_t = M_{t-1} x (1 - m_t): the customer gets yesterday's market rate minus the markup.
- Settlement: the business buys the currency on the next business day (T+1), or the one after
  (T+2) for some trades, at that day's market rate S_t.
- Realized margin rate r_t = 1 - C_t / S_t. If the currency strengthens before settlement,
  the margin erodes.
- Volume from the Project 2 generator; margin in USD = volume x r_t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG
from .volume import generate_volume


def _next_business_day(dates: pd.DatetimeIndex, k: int) -> pd.DatetimeIndex:
    return dates + pd.offsets.BDay(k)


def simulate_corridor(market: pd.DataFrame, corridor: str, series: str, currency: str,
                      base: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = market.index
    m_rate = market[series]

    # Volatility of daily log returns on trading days, known at the end of each day.
    trading = market.loc[market["is_trading_day"], series]
    ret = np.log(trading).diff()
    vol20 = ret.rolling(20).std().reindex(dates).ffill().bfill()
    vol_known = vol20.shift(1).bfill()  # pricing for day t uses volatility up to t-1

    # Slow drift (AR(1)) plus step changes from occasional pricing reviews (~every 60 days).
    n = len(dates)
    drift = np.zeros(n)
    steps = np.cumsum(np.where(rng.random(n) < 1 / 60, rng.normal(0, 0.002, n), 0.0))
    shocks = rng.normal(0, 0.00025, n)
    for i in range(1, n):
        drift[i] = 0.99 * drift[i - 1] + shocks[i]
    steps = steps - pd.Series(steps).rolling(365, min_periods=1).mean().to_numpy()  # keep near base
    lo, hi = CONFIG.markup_bounds
    markup = np.clip(base + CONFIG.markup_vol_sensitivity * vol_known.to_numpy() + drift + steps, lo, hi)

    customer = m_rate.shift(1).bfill().to_numpy() * (1 - markup)
    lag = np.where(rng.random(n) < CONFIG.t_plus_2_share, 2, 1)
    settle_dates = pd.DatetimeIndex([d + pd.offsets.BDay(k) for d, k in zip(dates, lag)])
    settle_rate = m_rate.reindex(settle_dates).to_numpy()
    margin_rate = 1 - customer / settle_rate

    volume = generate_volume(corridor, dates, seed + 100).to_numpy()
    out = pd.DataFrame({
        "date": dates, "corridor": corridor, "currency": currency,
        "market_rate": m_rate.to_numpy(), "markup": markup, "customer_rate": customer,
        "settle_date": settle_dates, "settle_rate": settle_rate, "margin_rate": margin_rate,
        "volume": volume, "vol20": vol_known.to_numpy(),
    })
    out["margin_usd"] = out["volume"] * out["margin_rate"]
    # The last few days settle after the data ends: their margin isn't known yet.
    return out[out["settle_date"] <= dates.max()].reset_index(drop=True)


def simulate_business(market: pd.DataFrame) -> pd.DataFrame:
    """All corridors in long format (one row per corridor and calendar day)."""
    frames = [
        simulate_corridor(market, corridor, series, currency, CONFIG.markup_base[i], CONFIG.seed + i)
        for i, (corridor, series, currency) in enumerate(CONFIG.corridors)
    ]
    return pd.concat(frames, ignore_index=True)
