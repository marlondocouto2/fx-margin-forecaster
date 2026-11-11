"""One command: load FX data -> simulate margins -> (tune) -> walk-forward -> results and figures.

    python scripts/run_pipeline.py            # uses the tuned parameters in reports/best_params.json
    python scripts/run_pipeline.py --tune     # re-run the Optuna search first (~1 minute)
    python scripts/run_pipeline.py --refresh  # re-download the FRED data snapshot first
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from fx_margin import plots  # noqa: E402
from fx_margin.business import simulate_business  # noqa: E402
from fx_margin.config import CONFIG  # noqa: E402
from fx_margin.data import download_fred, load_market  # noqa: E402
from fx_margin.evaluate import accuracy_table, risk_table, walk_forward  # noqa: E402
from fx_margin.features import FEATURES, build_dataset  # noqa: E402
from fx_margin.models import load_params, make_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tune", action="store_true", help="re-run hyperparameter tuning")
    parser.add_argument("--refresh", action="store_true", help="re-download FRED data")
    args = parser.parse_args()
    out, fig = CONFIG.reports_dir, CONFIG.reports_dir / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    if args.refresh:
        for _, s, _ in CONFIG.corridors:
            download_fred(s)
        download_fred(CONFIG.dollar_index)
    market = load_market()
    business = simulate_business(market)
    data = build_dataset(business, market)
    test_start = data["origin"].max() - pd.Timedelta(days=CONFIG.test_days)
    print(f"Market data {market.index.min():%Y-%m-%d} to {market.index.max():%Y-%m-%d}; "
          f"{len(data):,} rows; test period from {test_start:%Y-%m-%d}")

    if args.tune:
        from fx_margin.tuning import tune
        print("Tuning (time-ordered CV on pre-test data):")
        tune(data, test_start)
    params = load_params()

    t0 = time.time()
    bt = walk_forward(data, business, params)
    print(f"Walk-forward: {len(bt):,} forecasts in {time.time() - t0:.0f}s")
    acc, risk = accuracy_table(bt), risk_table(bt)
    acc.to_csv(out / "results.csv", index=False, float_format="%.6g")
    risk.to_csv(out / "risk.csv", index=False, float_format="%.6g")
    keep = ["corridor", "currency", "origin", "target", "horizon", "margin_rate", "volume", "oracle",
            "stage1_sma", "stage1_ewma", "plus_ridge", "plus_xgboost", "plus_lightgbm", "mar_band"]
    bt[keep].to_csv(out / "backtest_forecasts.csv.gz", index=False, float_format="%.6g",
                    compression={"method": "gzip", "mtime": 0})

    # SHAP on the best stage 2 model, fit on everything settled by the end of the data.
    import shap
    train = data[data["settle_date"] < data["origin"].max()]
    model = make_model("xgboost", params).fit(train[FEATURES], train["resid"])
    sample = train[FEATURES].sample(3000, random_state=CONFIG.seed)
    sv = shap.TreeExplainer(model).shap_values(sample)
    importance = pd.Series(abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    importance.rename("mean_abs_shap").to_csv(out / "shap_importance.csv", float_format="%.6g")

    # Charts. The band chart shows the most volatile 3 months of the test period.
    plots.plot_ladder(acc, 1, fig / "model_ladder.png")
    plots.plot_table(acc, risk, 1, fig / "results_table.png")
    plots.plot_shap(sv, sample, fig / "shap_summary.png")
    corr, start, end = volatile_window(bt)
    plots.plot_band(bt, corr, start, end, fig / "risk_band.png")

    show = acc[acc["horizon"] == 1][["model", "MAE_bps", "RMSE_bps", "MAE_usd",
                                      "share_of_predictable_error_captured"]].round(3)
    print("\n1 day ahead:\n" + show.to_string(index=False))
    r = risk[risk["corridor"] == "All"]
    print("\nMargin at risk (5% band) breach rate:",
          ", ".join(f"{h}d {b:.1%}" for h, b in zip(r["horizon"], r["breach_rate"])))
    print("Top SHAP features:", ", ".join(importance.index[:5]))
    print(f"Saved results and figures to {out}")


def volatile_window(bt: pd.DataFrame, days: int = 90) -> tuple[str, str, str]:
    """Corridor and 90-day window with the most volatile realized margin in the test period."""
    g = bt[bt["horizon"] == 1].sort_values("target")
    best = None
    for corridor, s in g.groupby("corridor"):
        vol = s.set_index("target")["margin_rate"].rolling(f"{days}D").std()
        end = vol.idxmax()
        if best is None or vol.max() > best[0]:
            best = (vol.max(), corridor, end - pd.Timedelta(days=days), end)
    _, corridor, start, end = best
    return corridor, f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"


if __name__ == "__main__":
    main()
