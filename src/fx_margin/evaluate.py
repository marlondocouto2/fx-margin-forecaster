"""Walk-forward backtest, accuracy and risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG
from .models import TwoStage, ladder_columns

BPS = 1e4


def walk_forward(data: pd.DataFrame, business: pd.DataFrame, params: dict | None = None,
                 test_days: int = CONFIG.test_days, refit_every: int = CONFIG.refit_every) -> pd.DataFrame:
    """Rolling-origin backtest over the last `test_days` days.

    Stage 1 is causal at every origin by construction. Stage 2 is refit every `refit_every`
    days on rows whose target margin had settled by the refit date, then predicts every
    origin until the next refit.
    """
    last = data["origin"].max()
    test_start = last - pd.Timedelta(days=test_days)
    cutoffs = pd.date_range(test_start, last, freq=f"{refit_every}D")
    preds = []
    for i, cut in enumerate(cutoffs):
        nxt = cutoffs[i + 1] if i + 1 < len(cutoffs) else last + pd.Timedelta(days=1)
        train = data[data["settle_date"] < cut]
        block = data[(data["origin"] >= cut) & (data["origin"] < nxt)]
        if block.empty:
            continue
        model = TwoStage(params).fit(train)
        preds.append(block.join(model.predict(block)))
    out = pd.concat(preds)
    # Oracle reference: knows the target day's markup exactly, but not the FX move during settlement.
    markup = business.set_index(["corridor", "date"])["markup"]
    out["oracle"] = markup.reindex(pd.MultiIndex.from_arrays([out["corridor"], out["target"]])).to_numpy()
    return out.reset_index(drop=True)


def accuracy_table(bt: pd.DataFrame) -> pd.DataFrame:
    """One row per (horizon, model): error on the margin rate (bps) and on USD margin."""
    cols = {**ladder_columns(), "oracle": "Oracle (knows the markup, not the FX move)"}
    rows = []
    for h, g in bt.groupby("horizon"):
        base_mae = np.mean(np.abs(g["margin_rate"] - g["stage1_ewma"]))
        oracle_mae = np.mean(np.abs(g["margin_rate"] - g["oracle"]))
        for col, name in cols.items():
            err = g["margin_rate"] - g[col]
            usd = err * g["volume"]
            mae = np.mean(np.abs(err))
            rows.append({
                "horizon": h, "model": name,
                "MAE_bps": mae * BPS, "RMSE_bps": np.sqrt(np.mean(err**2)) * BPS,
                "MSE": np.mean(err**2),
                "MAE_usd": np.mean(np.abs(usd)), "RMSE_usd": np.sqrt(np.mean(usd**2)),
                "share_of_predictable_error_captured": (base_mae - mae) / (base_mae - oracle_mae),
            })
    return pd.DataFrame(rows)


def risk_table(bt: pd.DataFrame) -> pd.DataFrame:
    """Margin at risk: how often realized margin fell below the 5% band, and by how much."""
    rows = []
    for (h, c), g in bt.groupby(["horizon", "corridor"]):
        rows.append(_risk_row(g, h, c))
    for h, g in bt.groupby("horizon"):
        rows.append(_risk_row(g, h, "All"))
    return pd.DataFrame(rows)


def _risk_row(g: pd.DataFrame, h: int, corridor: str) -> dict:
    below = g["margin_rate"] < g["mar_band"]
    shortfall_usd = ((g["mar_band"] - g["margin_rate"]) * g["volume"])[below]
    return {"horizon": h, "corridor": corridor, "days": len(g),
            "breach_rate": below.mean(), "target": CONFIG.mar_quantile,
            "avg_shortfall_usd": shortfall_usd.mean() if below.any() else 0.0,
            "avg_band_bps": (g["stage1_ewma"] - g["mar_band"]).mean() * BPS}
