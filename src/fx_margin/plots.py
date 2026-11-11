"""Static charts for the README and notebooks (PNG, matplotlib). With path=None they return the figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
MODEL_COLORS = {"1. Stage 1: moving average": "#8a8984", "2. Stage 1: EWMA": "#eb6834",
                "3. + Ridge": "#1baf7a", "4. + XGBoost": "#2a78d6", "5. + LightGBM": "#6250d6"}
BAND, BREACH = "#cfe0f6", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "legend.frameon": False, "legend.labelcolor": INK_2,
})


def _save(fig, path: Path | None):
    plt.close(fig)
    if path is None:
        return fig
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def plot_ladder(acc: pd.DataFrame, horizon: int, path: Path | None) -> Path:
    """MAE of each version, with the oracle floor and the share of predictable error captured."""
    t = acc[acc["horizon"] == horizon]
    oracle = t[t["model"].str.startswith("Oracle")].iloc[0]
    t = t[~t["model"].str.startswith("Oracle")]
    base = t[t["model"] == "2. Stage 1: EWMA"]["MAE_bps"].iloc[0]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(t))
    ax.bar(x, t["MAE_bps"], 0.62, color=[MODEL_COLORS[m] for m in t["model"]], zorder=2)
    for i, (_, r) in enumerate(t.iterrows()):
        label = f"{r['MAE_bps']:.1f} bps"
        if r["model"] not in ("1. Stage 1: moving average", "2. Stage 1: EWMA"):
            label += f"\n{r['share_of_predictable_error_captured']:.0%} of predictable error"
        ax.annotate(label, (i, r["MAE_bps"]), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(oracle["MAE_bps"], color=INK, lw=1.2, ls="--", zorder=3,
               label=f"Oracle floor, {oracle['MAE_bps']:.1f} bps: knows the markup but not the FX move during settlement")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16))
    ax.set_xticks(x, [m.split(". ", 1)[1].replace("Stage 1: ", "Stage 1\n") for m in t["model"]])
    ax.tick_params(axis="x", length=0)
    lo = oracle["MAE_bps"] - 4
    ax.set_ylim(lo, base + 2.5)
    ax.set_ylabel("Mean absolute error (bps of margin)")
    ax.set_title(f"Margin-rate error by model, {horizon} day{'s' if horizon > 1 else ''} ahead "
                 "(y-axis starts near the floor)")
    return _save(fig, path)


def plot_band(bt: pd.DataFrame, corridor: str, start: str, end: str, path: Path | None,
              model_col: str = "plus_xgboost", horizon: int = 1) -> Path:
    """Realized margin vs forecast and the 5% margin-at-risk band over one stretch."""
    g = bt[(bt["corridor"] == corridor) & (bt["horizon"] == horizon)
           & bt["target"].between(start, end)].sort_values("target")
    bps = 1e4
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.fill_between(g["target"], g["mar_band"] * bps, g[model_col] * bps, color=BAND, zorder=1,
                    label="Between forecast and 5% margin-at-risk band")
    ax.plot(g["target"], g[model_col] * bps, color=MODEL_COLORS["4. + XGBoost"], lw=2, label="Forecast (+ XGBoost)", zorder=3)
    ax.plot(g["target"], g["mar_band"] * bps, color=MODEL_COLORS["4. + XGBoost"], lw=1, ls=":", zorder=3)
    ax.plot(g["target"], g["margin_rate"] * bps, color=INK, lw=1.1, marker="o", ms=2.5, label="Realized margin", zorder=4)
    br = g[g["margin_rate"] < g["mar_band"]]
    ax.scatter(br["target"], br["margin_rate"] * bps, s=60, facecolors="none", edgecolors=BREACH, lw=1.6,
               zorder=5, label=f"Below the band ({len(br)} of {len(g)} days)")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylabel("Margin rate (bps)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d\n%Y"))
    names = {"MX": "U.S. → Mexico (MXN)", "IN": "U.S. → India (INR)", "BR": "U.S. → Brazil (BRL)"}
    ax.set_title(f"{names.get(corridor, corridor)}: forecast, 5% risk band and realized margin, {horizon} day ahead")
    ax.legend(ncols=2, loc="upper left", bbox_to_anchor=(0, -0.14))
    return _save(fig, path)


def plot_shap(shap_values, X: pd.DataFrame, path: Path | None, max_display: int = 12) -> Path:
    """SHAP summary (beeswarm) of the stage 2 model."""
    import shap

    shap.summary_plot(shap_values, X, max_display=max_display, show=False, plot_size=(9, 5.5),
                      color_bar_label="Feature value")
    fig = plt.gcf()
    fig.axes[0].set_title("What drives the stage 2 correction (SHAP, XGBoost)", loc="left")
    fig.axes[0].set_xlabel("Impact on predicted margin correction (rate units)")
    return _save(fig, path)


def plot_table(acc: pd.DataFrame, risk: pd.DataFrame, horizon: int, path: Path | None) -> Path:
    t = acc[acc["horizon"] == horizon]
    cells = [[r["model"], f"{r['MAE_bps']:.1f}", f"{r['RMSE_bps']:.1f}",
              f"${r['MAE_usd'] / 1e3:,.0f}k",
              "–" if r["model"].startswith(("1.", "2.")) else
              ("100%" if r["model"].startswith("Oracle") else f"{r['share_of_predictable_error_captured']:.0%}")]
             for _, r in t.iterrows()]
    headers = ["Model", "MAE (bps)", "RMSE (bps)", "MAE (USD/day)", "Predictable error captured"]
    fig, ax = plt.subplots(figsize=(11, 0.42 * (len(cells) + 1) + 0.3))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.55)
    table.auto_set_column_width(list(range(len(headers))))
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor("#eef4fc" if r and cells[r - 1][0].startswith("4.") else SURFACE)
        cell.get_text().set_color(INK_2 if r == 0 else INK)
        if r == 0:
            cell.get_text().set_weight("bold")
    return _save(fig, path)
