"""Hyperparameter tuning with Optuna and time-ordered cross-validation.

Runs only on data whose margins had settled before the test period starts, so the test
period never influences the chosen parameters. Trials run in parallel (`n_jobs`).

The same search would run as a managed tuning job on a cloud platform (for example a
SageMaker hyperparameter tuning job with the objective "validation MAE"), but nothing here
requires a cloud account.
"""

from __future__ import annotations

import json

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import CONFIG
from .features import FEATURES
from .models import PARAMS_FILE, load_params, make_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

SPACES = {
    "xgboost": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": t.suggest_int("max_depth", 2, 6),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": t.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    },
    "lightgbm": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
        "num_leaves": t.suggest_int("num_leaves", 4, 48),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 20, 400, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    },
    "ridge": lambda t: {"alpha": t.suggest_float("alpha", 1e-3, 1e3, log=True)},
}


def cv_mae(kind: str, params: dict, data: pd.DataFrame, n_splits: int = CONFIG.cv_splits) -> float:
    """Mean validation MAE over expanding time-ordered folds (split on origins, never shuffled)."""
    origins = np.sort(data["origin"].unique())
    scores = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(origins):
        cut = origins[val_idx[0]]
        train = data[data["settle_date"] < cut]  # only margins settled before the fold starts
        val = data[data["origin"].isin(origins[val_idx])]
        model = make_model(kind, {kind: params}).fit(train[FEATURES], train["resid"])
        scores.append(np.mean(np.abs(val["resid"] - model.predict(val[FEATURES]))))
    return float(np.mean(scores))


def tune(data: pd.DataFrame, test_start: pd.Timestamp, n_trials: int = CONFIG.optuna_trials,
         n_jobs: int = -1) -> dict:
    """Tune each stage 2 model on pre-test data and save the best parameters."""
    pre = data[data["settle_date"] < test_start]
    best = {}
    for kind, space in SPACES.items():
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=CONFIG.seed))
        study.optimize(lambda t: cv_mae(kind, space(t), pre), n_trials=n_trials, n_jobs=n_jobs)
        best[kind] = study.best_params
        print(f"  {kind:<9} best CV MAE {study.best_value * 1e4:.2f} bps after {n_trials} trials")
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_FILE.write_text(json.dumps(best, indent=2))
    return load_params()
