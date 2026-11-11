"""Stage 2 residual models and the margin-at-risk quantile model.

Each model learns the stage 1 error (residual = realized margin rate - stage 1 EWMA forecast)
from features known at the origin. The final forecast is stage 1 + predicted residual.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CONFIG
from .features import FEATURES

# Sensible defaults; replaced by tuned values from reports/best_params.json when present.
DEFAULT_PARAMS = {
    "xgboost": {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8,
                "colsample_bytree": 0.8, "min_child_weight": 20.0, "reg_lambda": 1.0},
    "lightgbm": {"n_estimators": 400, "num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 100,
                 "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    "ridge": {"alpha": 1.0},
}
PARAMS_FILE = CONFIG.reports_dir / "best_params.json"


def load_params(path: Path = PARAMS_FILE) -> dict:
    params = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}
    if path.exists():
        for k, v in json.loads(path.read_text()).items():
            params[k].update(v)
    return params


def make_model(kind: str, params: dict, quantile: float | None = None):
    """A stage 2 regressor. L1 (median) loss for point forecasts; pinball loss for quantiles."""
    p = params[kind]
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"]))
    if kind == "xgboost":
        return xgb.XGBRegressor(objective="reg:absoluteerror", tree_method="hist", n_jobs=1,
                                random_state=CONFIG.seed, **p)
    if kind == "lightgbm":
        objective = {"objective": "quantile", "alpha": quantile} if quantile else {"objective": "l1"}
        return lgb.LGBMRegressor(**objective, subsample_freq=1, n_jobs=1, verbose=-1,
                                 random_state=CONFIG.seed, **p)
    raise ValueError(kind)


STAGE2_KINDS = ["ridge", "xgboost", "lightgbm"]


class TwoStage:
    """Stage 1 forecasts come precomputed in the dataset; this fits and applies stage 2."""

    def __init__(self, params: dict | None = None):
        self.params = params or load_params()

    def fit(self, train: pd.DataFrame) -> "TwoStage":
        X, y = train[FEATURES], train["resid"]
        self.models_ = {k: make_model(k, self.params).fit(X, y) for k in STAGE2_KINDS}
        # Margin at risk: LightGBM quantile regression, then conformal calibration. The quantile
        # model is fit on all but the last `calibration_days`; on those held-back days the band is
        # shifted so that exactly the target share of realized margins falls below it.
        cut = train["settle_date"].max() - pd.Timedelta(days=CONFIG.calibration_days)
        fit_part, cal = train[train["settle_date"] <= cut], train[train["settle_date"] > cut]
        self.mar_ = make_model("lightgbm", self.params, quantile=CONFIG.mar_quantile).fit(
            fit_part[FEATURES], fit_part["resid"])
        score = self.mar_.predict(cal[FEATURES]) - cal["resid"].to_numpy()  # > 0 means a breach
        n = len(score)
        level = min(1.0, (1 - CONFIG.mar_quantile) * (n + 1) / n)
        self.mar_shift_ = float(np.quantile(score, level))
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = rows[FEATURES]
        out = pd.DataFrame(index=rows.index)
        out["stage1_sma"] = rows["s1_sma"]
        out["stage1_ewma"] = rows["s1_ewma"]
        for k in STAGE2_KINDS:
            out[f"plus_{k}"] = rows["s1_ewma"] + self.models_[k].predict(X)
        out["mar_band"] = rows["s1_ewma"] + self.mar_.predict(X) - self.mar_shift_
        return out


def ladder_columns() -> dict[str, str]:
    """Prediction column -> display name, in ladder order."""
    return {"stage1_sma": "1. Stage 1: moving average", "stage1_ewma": "2. Stage 1: EWMA",
            "plus_ridge": "3. + Ridge", "plus_xgboost": "4. + XGBoost", "plus_lightgbm": "5. + LightGBM"}


def as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)
