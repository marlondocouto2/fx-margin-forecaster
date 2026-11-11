"""Public market data from FRED, cached as CSV snapshots in data/fred/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CONFIG

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def download_fred(series: str, out_dir: Path = CONFIG.fred_dir, start: str = "2015-01-01") -> Path:
    """Download one FRED series (no API key needed) and save it from `start` onwards."""
    import requests

    resp = requests.get(FRED_CSV.format(series=series), timeout=60)
    resp.raise_for_status()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{series}.csv"
    from io import StringIO

    df = pd.read_csv(StringIO(resp.text))
    df = df[df.iloc[:, 0] >= start]
    df.to_csv(path, index=False)
    return path


def load_fred(series: str, fred_dir: Path = CONFIG.fred_dir) -> pd.Series:
    """One FRED series from the local snapshot, as a float Series indexed by date."""
    df = pd.read_csv(fred_dir / f"{series}.csv")
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")  # FRED marks missing days with '.' or blanks
    return pd.Series(s.to_numpy(), index=pd.to_datetime(df.iloc[:, 0]), name=series).dropna()


def load_market(start: str = CONFIG.start, end: str | None = CONFIG.end,
                fred_dir: Path = CONFIG.fred_dir) -> pd.DataFrame:
    """Daily calendar-day market data: one column per FRED series, non-trading days forward-filled.

    Also returns `is_trading_day` (all FX series observed that day).
    """
    series = [s for _, s, _ in CONFIG.corridors] + [CONFIG.dollar_index]
    raw = pd.concat([load_fred(s, fred_dir) for s in series], axis=1, sort=True)
    last = raw.dropna().index.max() if end is None else pd.Timestamp(end)
    days = pd.date_range(start, last, freq="D")
    fx = [s for _, s, _ in CONFIG.corridors]
    trading = raw[fx].notna().all(axis=1).reindex(days, fill_value=False)
    out = raw.reindex(raw.index.union(days)).ffill().reindex(days)
    out["is_trading_day"] = trading.to_numpy()
    return out
