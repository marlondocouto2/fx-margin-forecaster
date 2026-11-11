# Data

## Real: FX rates from FRED (`data/fred/`)

Daily series downloaded from [FRED](https://fred.stlouisfed.org/) as CSV (no API key), from
2015-01-01 onwards, and committed as a snapshot so every run is reproducible and offline:

| File | Series |
|---|---|
| `DEXMXUS.csv` | Mexican pesos per U.S. dollar |
| `DEXINUS.csv` | Indian rupees per U.S. dollar |
| `DEXBZUS.csv` | Brazilian reais per U.S. dollar |
| `DTWEXBGS.csv` | Broad trade-weighted U.S. dollar index |

Refresh with `python scripts/run_pipeline.py --refresh`. FRED marks days without a value with
a blank or `.`; those are dropped, and the pipeline forward-fills weekends and holidays.

## Simulated: everything about the business

Generated in memory on every run, from fixed seeds (`src/fx_margin/business.py`, `volume.py`):

- **Markup policy:** a base per currency, plus a reaction to recent FX volatility, a slow drift and
  occasional pricing reviews, kept within 0.5–2%.
- **Customer rate:** the previous day's market rate × (1 − markup).
- **Settlement:** T+1 business day, or T+2 for 30% of trades, at that day's market rate.
- **Volumes:** the Project 2 generator for the Mexico, India and Brazil corridors.

None of this represents any real company's data.
