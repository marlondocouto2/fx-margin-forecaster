import numpy as np
import pandas as pd
import pytest

from fx_margin.business import simulate_business
from fx_margin.config import CONFIG
from fx_margin.data import load_market
from fx_margin.evaluate import accuracy_table, risk_table, walk_forward
from fx_margin.features import FEATURES, build_dataset
from fx_margin.models import load_params

TARGET_SIDE = ["margin_rate", "resid", "volume", "settle_date"]  # describe the target day, not the origin


@pytest.fixture(scope="module")
def market():
    return load_market()


@pytest.fixture(scope="module")
def business(market):
    return simulate_business(market)


@pytest.fixture(scope="module")
def data(business, market):
    return build_dataset(business, market)


@pytest.fixture(scope="module")
def backtest(data, business):
    return walk_forward(data, business, load_params())


def _alter_after(market: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Shock every FX rate and the dollar index after the cutoff with a growing drift (up to +30%),
    so both price levels and returns change."""
    m = market.copy()
    cols = [s for _, s, _ in CONFIG.corridors] + [CONFIG.dollar_index]
    after = m.index > cutoff
    m.loc[after, cols] = m.loc[after, cols].mul(np.linspace(1.0, 1.3, after.sum()), axis=0)
    return m


# ----------------------------------------------------------------- simulated business layer
def test_margin_formula_and_markup_bounds(business):
    b = business
    expected = 1 - b["customer_rate"] / b["settle_rate"]
    np.testing.assert_allclose(b["margin_rate"], expected)
    lo, hi = CONFIG.markup_bounds
    assert b["markup"].between(lo, hi).all()
    lag_days = (b["settle_date"] - b["date"]).dt.days
    assert lag_days.between(1, 4).all()  # T+1 or T+2 business days


# ----------------------------------------------------------------- no feature uses the future
def test_no_feature_uses_data_after_the_forecast_date(market):
    cutoff = pd.Timestamp("2023-06-15")
    a = build_dataset(simulate_business(market), market)
    shocked = _alter_after(market, cutoff)
    b = build_dataset(simulate_business(shocked), shocked)
    key = ["corridor", "origin", "horizon"]
    cols = [c for c in FEATURES + ["s1_sma", "s1_ewma"] if c not in key]
    ka = a[a["origin"] <= cutoff].set_index(key)[cols].sort_index()
    kb = b[b["origin"] <= cutoff].set_index(key)[cols].sort_index()
    pd.testing.assert_frame_equal(ka, kb)
    # The shock is real: features after the cutoff do change.
    later = lambda d: d[d["origin"] > cutoff + pd.Timedelta(days=30)].set_index(key)["ret_5"].sort_index()  # noqa: E731
    assert not np.allclose(later(a), later(b))


def test_walk_forward_has_no_leakage(market):
    """Forecasts made up to a date don't change when the market after that date changes."""
    cutoff = market.index.max() - pd.Timedelta(days=60)
    shocked = _alter_after(market, cutoff)
    runs = []
    for m in (market, shocked):
        bus = simulate_business(m)
        runs.append(walk_forward(build_dataset(bus, m), bus, load_params(), test_days=120))
    key = ["corridor", "origin", "horizon"]
    cols = ["stage1_ewma", "plus_ridge", "plus_xgboost", "plus_lightgbm", "mar_band"]
    a, b = (r[r["origin"] <= cutoff].set_index(key)[cols].sort_index() for r in runs)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b)


# ----------------------------------------------------------------- results
def test_stage2_improves_on_stage1_mae(backtest):
    acc = accuracy_table(backtest).set_index(["horizon", "model"])["MAE_bps"]
    for h in CONFIG.horizons:
        stage1 = acc[(h, "2. Stage 1: EWMA")]
        for model in ("3. + Ridge", "4. + XGBoost", "5. + LightGBM"):
            assert acc[(h, model)] < stage1, (h, model)


def test_margin_at_risk_coverage_within_3_to_7_percent(backtest):
    risk = risk_table(backtest)
    overall = risk[risk["corridor"] == "All"]
    assert overall["breach_rate"].between(0.03, 0.07).all(), overall
    by_corridor = risk[risk["corridor"] != "All"]
    assert by_corridor["breach_rate"].between(0.03, 0.08).all(), by_corridor


def test_oracle_is_the_floor(backtest):
    acc = accuracy_table(backtest)
    for h, g in acc.groupby("horizon"):
        oracle = g[g["model"].str.startswith("Oracle")]["MAE_bps"].iloc[0]
        assert (g["MAE_bps"] >= oracle - 1e-9).all()
