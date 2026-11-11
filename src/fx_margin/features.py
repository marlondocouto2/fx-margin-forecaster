"""Stage 1 trend forecasts and the leakage-safe feature set for stage 2.

Everything here is computed per corridor on calendar days, as of a forecast origin o
(end of day o), for a target trade day t = o + h:

- FX market data is known through day o.
- A trade day's realized margin is known only once it has settled. To stay simple and safe,
  the models only use margins at least `known_lag_days` (4) days old, which have always settled.
- The business's own markup for day o is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import holiday_window
from .config import CONFIG

K = CONFIG.known_lag_days

FEATURES = [
    "ret_1", "ret_2", "ret_5", "ret_10", "vol_5", "vol_20", "vol_ratio",
    "dxy_ret_1", "dxy_ret_5", "markup_now", "dev_lag_0", "dev_lag_1", "dev_lag_2", "dev_mean_7",
    "trend_slope", "markup_gap", "dow", "is_weekend", "is_month_end", "is_holiday", "horizon",
    "corr_MX", "corr_IN", "corr_BR",
]


def stage1_components(r: pd.Series) -> pd.DataFrame:
    """Per origin: moving-average and EWMA trend of known (settled) margins.

    The stage 1 forecast for every future day is the latest trend value. `dev` is each trade
    day's deviation from the trend known at that day, used as a stage 2 feature.
    """
    known = r.shift(K)
    sma = known.rolling(CONFIG.sma_window, min_periods=10).mean()
    ewma = known.ewm(halflife=CONFIG.ewma_halflife, adjust=False).mean()
    return pd.DataFrame({"sma": sma, "ewma": ewma, "dev": r - ewma})


def _trading_returns(price: pd.Series, trading: pd.Series) -> pd.DataFrame:
    p = np.log(price[trading])
    ret = p.diff()
    out = pd.DataFrame({f"ret_{k}": p.diff(k) for k in (1, 2, 5, 10)})
    out["vol_5"] = ret.rolling(5).std()
    out["vol_20"] = ret.rolling(20).std()
    return out.reindex(price.index).ffill()


def build_dataset(business: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """One row per (corridor, origin, horizon): features known at the origin, stage 1 forecasts,
    and the realized margin of the target day."""
    frames = []
    trading = market["is_trading_day"]
    dxy = _trading_returns(market[CONFIG.dollar_index], trading)
    for corridor, series, currency in CONFIG.corridors:
        b = business[business["corridor"] == corridor].set_index("date").sort_index()
        r = b["margin_rate"]
        s1 = stage1_components(r)
        fx = _trading_returns(market[series], trading).reindex(r.index)
        origin = pd.DataFrame(index=r.index)
        origin[["ret_1", "ret_2", "ret_5", "ret_10", "vol_5", "vol_20"]] = fx
        origin["vol_ratio"] = origin["vol_5"] / origin["vol_20"]
        origin["dxy_ret_1"] = dxy["ret_1"].reindex(r.index)
        origin["dxy_ret_5"] = dxy["ret_5"].reindex(r.index)
        origin["markup_now"] = b["markup"]
        origin["dev_lag_0"] = s1["dev"].shift(K)
        origin["dev_lag_1"] = s1["dev"].shift(K + 1)
        origin["dev_lag_2"] = s1["dev"].shift(K + 2)
        origin["dev_mean_7"] = s1["dev"].rolling(7).mean().shift(K)
        origin["trend_slope"] = s1["ewma"] - s1["ewma"].shift(7)
        hol = holiday_window(r.index, corridor)["holiday_tier"].notna().to_numpy()

        for h in CONFIG.horizons:
            t = r.index + pd.Timedelta(days=h)
            ok = t.isin(r.index)
            o_idx, t_idx = r.index[ok], t[ok]
            d = origin.loc[o_idx].copy()
            tdow = t_idx.day_of_week.to_numpy()
            d["s1_sma"] = s1.loc[o_idx, "sma"].to_numpy()
            d["s1_ewma"] = s1.loc[o_idx, "ewma"].to_numpy()
            d["markup_gap"] = d["markup_now"] - d["s1_ewma"]  # today's markup vs the stage 1 forecast
            d["dow"] = tdow
            d["is_weekend"] = (tdow >= 5).astype(int)
            d["is_month_end"] = (t_idx.days_in_month - t_idx.day < 2).astype(int)
            d["is_holiday"] = pd.Series(hol, index=r.index).reindex(t_idx).to_numpy().astype(int)
            d["horizon"] = h
            for c, _, _ in CONFIG.corridors:
                d[f"corr_{c}"] = int(c == corridor)
            d["corridor"] = corridor
            d["currency"] = currency
            d["origin"] = o_idx
            d["target"] = t_idx
            d["settle_date"] = b.loc[t_idx, "settle_date"].to_numpy()
            d["margin_rate"] = r.loc[t_idx].to_numpy()
            d["volume"] = b.loc[t_idx, "volume"].to_numpy()
            frames.append(d.reset_index(drop=True))
    data = pd.concat(frames, ignore_index=True)
    data["resid"] = data["margin_rate"] - data["s1_ewma"]
    return data.dropna(subset=FEATURES + ["s1_sma", "s1_ewma"]).reset_index(drop=True)
