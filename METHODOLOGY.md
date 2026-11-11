# Methodology

This document explains how the FX margin forecaster works: how the data is built, how each stage
is estimated, how it is tested, and what the results do and don't show. The [README](README.md)
has the headline results; the [walkthrough notebook](notebooks/walkthrough.ipynb) shows each step
with charts. Numbers come from `python scripts/run_pipeline.py` with the settings in
[`config.py`](src/fx_margin/config.py).

## 1. The business problem

A money-transfer business takes U.S. dollars from customers and pays out pesos, rupees and reais.
It quotes each customer a rate based on the previous day's market rate minus a markup. It then buys
the foreign currency one or two business days later, at the market rate on that day. The
difference between the quoted rate and the rate it actually pays is its FX margin.

Because settlement happens later, the margin is exposed to the currency's move in between:

- **The currency weakens** (more units per dollar): the business pays less than expected, and the
  margin grows.
- **The currency strengthens:** the margin shrinks, and can turn negative.

Treasury wants two numbers per corridor and day: the **expected margin**, and the **margin at
risk**, a level the margin falls below only on 5% of days.

## 2. Data

### 2.1 Market data (real)

Four daily series from FRED, downloaded as CSV with no API key and saved in
[`data/fred/`](data/fred), so runs are reproducible offline:

| Series | Meaning |
|---|---|
| `DEXMXUS` | Mexican pesos per U.S. dollar |
| `DEXINUS` | Indian rupees per U.S. dollar |
| `DEXBZUS` | Brazilian reais per U.S. dollar |
| `DTWEXBGS` | broad trade-weighted U.S. dollar index |

The data covers 2016-01-01 to 2026-11-06. Values are placed on a calendar-day grid, with weekends
and holidays carrying the last trading day's value forward. A day counts as a trading day when all
three FX series are observed.

### 2.2 Business layer (simulated)

For each corridor and calendar day $`t`$, with market rate $`M_t`$ in foreign units per dollar:

**Markup policy.**

```math
m_t = \mathrm{clip}\left(b + 0.6\,\sigma^{(20)}_{t-1} + d_t + j_t,\ 0.5\%,\ 2\%\right)
```

- $`b`$: the base markup, 0.8% for MXN, 0.9% for INR and 1.0% for BRL.
- $`\sigma^{(20)}_{t-1}`$: the 20-trading-day volatility of daily log returns, known the evening
  before. Pricing widens when markets are volatile.
- $`d_t`$: a slow AR(1) drift, $`d_t = 0.99\,d_{t-1} + \epsilon_t`$.
- $`j_t`$: step changes from pricing reviews, about every 60 days, kept centred on the base.

**Customer rate.** The customer receives yesterday's market rate minus the markup:

```math
C_t = M_{t-1}\,(1 - m_t)
```

**Settlement.** The business buys the currency on the next business day (T+1), or the one after
(T+2) for 30% of trades. It pays $`S_t = M_{\mathrm{settle}(t)}`$.

**Realized margin rate.** A dollar from the customer buys $`C_t`$ units of currency, which cost
the business $`C_t / S_t`$ dollars:

```math
r_t = 1 - \frac{C_t}{S_t} = 1 - \frac{M_{t-1}}{S_t}\,(1 - m_t) \approx m_t + \log\frac{S_t}{M_{t-1}}
```

So the margin is the markup plus the currency's move from yesterday's close to settlement.

**Volume and USD margin.** Daily volume per corridor comes from the
[Project 2](https://github.com/marlondocouto2/cash-flow-forecaster) generator: trend, weekday,
paydays, holidays, scheduled and random spikes. Brazil is added with $10M a day growing 14% a
year. The daily margin in USD is $`\text{volume}_t \times r_t`$.

On average, margins come out at 1.1–1.5%. They go negative on about 1–10% of days, depending on
the currency's volatility.

## 3. What is knowable when

A forecast is made at the end of day $`o`$ (the origin) for trade day $`t = o + h`$, with
$`h \in \{1, 5\}`$. At that moment:

| Information | Known? |
|---|---|
| FX rates and the dollar index through day $`o`$ | yes |
| The business's own markup for day $`o`$ | yes |
| A trade day's realized margin | only after it settles, which takes up to 4 calendar days (Friday T+2 settles on Tuesday) |

For simplicity and safety, the models only use margins at least **4 days old**, which have
always settled. Tests check that no feature changes when future market data is altered.

## 4. Stage 1: trend plus weekday profile

Let $`\tilde r_o = r_{o-4}`$ be the most recent margin certainly known at origin $`o`$.

- **Moving average:** $`T^{\mathrm{SMA}}_o = \frac{1}{28}\sum_{i=0}^{27} \tilde r_{o-i}`$.
- **EWMA:** $`T^{\mathrm{EW}}_o = \lambda\,\tilde r_o + (1-\lambda)\,T^{\mathrm{EW}}_{o-1}`$, with
  a half-life of 10 days.
- **Weekday profile:** the average deviation $`r_s - T^{\mathrm{EW}}_s`$ for each weekday over
  the last 182 known days, centred so the seven values average to zero. This is the $`P_o(d)`$ in
  the forecast below.
- **Stage 1 forecast:** $`\hat r^{(1)}_{o \to t} = T_o + P_o(\mathrm{weekday}(t))`$.

The two stage 1 variants are compared. The EWMA forecast is the base that stage 2 corrects.

## 5. Stage 2: learning what the trend misses

**Target.** The stage 1 error, $`e_{o \to t} = r_t - \hat r^{(1),\mathrm{EW}}_{o \to t}`$.

**Features at the origin:**

| Group | Features |
|---|---|
| **FX moves** | log returns over 1, 2, 5 and 10 trading days; 5- and 20-day volatility and their ratio |
| **Dollar** | broad dollar index returns over 1 and 5 days |
| **Pricing** | today's markup $`m_o`$, and the **markup gap** $`m_o - \hat r^{(1)}_{o \to t}`$ |
| **Recent errors** | the last three known stage 1 deviations and their 7-day mean; the 7-day trend slope |
| **Calendar** | the target's weekday, a weekend flag, a month-end flag and a receiving-country holiday flag; the horizon; the corridor |

**Models.** All three are trained on all corridors together, with corridor indicators:

| Model | Loss |
|---|---|
| Ridge regression | squared error, on standardized features |
| XGBoost | absolute error |
| LightGBM | absolute error |

The absolute-error loss targets the median, which fits a heavy-tailed target.

**Final forecast.** $`\hat r_{o \to t} = \hat r^{(1)}_{o \to t} + \hat e_{o \to t}`$.

**Tuning.**

- **Search:** Optuna (TPE sampler, seed 42, 30 trials per model, run in parallel) minimizes the
  mean absolute error under 5-fold time-ordered cross-validation.
- **No look-ahead:** each fold trains only on margins settled before the fold starts. The search
  uses only data settled before the test period.
- **Saved results:** the chosen parameters are in [`reports/best_params.json`](reports/best_params.json).
- **Cloud option:** the same search maps directly onto a managed tuning job such as SageMaker, with
  "validation MAE" as the objective. It isn't needed to run the repo.

## 6. Margin at risk

The band is the stage 1 forecast plus the 5th-percentile stage 1 error:

```math
\mathrm{MaR}_{o \to t} = \hat r^{(1)}_{o \to t} + \hat q_{0.05}(x_o) - \delta
```

- **$`\hat q_{0.05}`$** is a LightGBM quantile regression (pinball loss at 0.05). It's fitted on all
  training data except the most recent 180 days.
- **$`\delta`$** is a conformal correction. On the held-back 180 days it's the
  $`(1-\alpha)(n+1)/n`$ quantile of $`\hat q_{0.05}(x) - e`$, which shifts the band so that 5% of
  those days fall below it.

Without the correction, the quantile model alone was breached on about 7% of test days.

## 7. Evaluation design

**Walk-forward.**

- **Test period:** the last 730 days of origins (4 November 2024 to 4 November 2026), for 3
  corridors and 2 horizons. That's 4,372 forecasts.
- **Stage 1:** causal at every origin by construction.
- **Stage 2 and the band:** refit every 30 days on targets settled before the refit date, then
  used for the next 30 days of origins.

**Metrics.**

- **Error on the margin rate:** MAE, RMSE and MSE, in basis points (1 bp = 0.01%).
- **Error on USD margin:** volume × rate error, per corridor-day. Actual volume is used, so this
  isolates margin-rate error from volume error.
- **Share of the predictable error captured:**

```math
\frac{\mathrm{MAE}_{\text{stage 1}} - \mathrm{MAE}_{\text{model}}}{\mathrm{MAE}_{\text{stage 1}} - \mathrm{MAE}_{\text{oracle}}}
```

  The oracle forecasts $`r_t = m_t`$: it knows the true markup, but expects no FX move. No model can
  beat it on average, so it marks the floor set by unpredictable FX moves.
- **Risk:** the breach rate (the share of days with realized margin below the band; target 5%)
  and the average shortfall in USD on breach days.

**Explainability.** SHAP values (TreeExplainer) for the XGBoost stage 2 model, fitted on all
settled data.

## 8. Results

1 day ahead:

| Model | MAE (bps) | RMSE (bps) | MAE (USD/day) | Predictable error captured |
|---|---|---|---|---|
| Stage 1: moving average | 57.8 | 87.9 | $429k | 3% |
| Stage 1: EWMA | 58.0 | 88.1 | $430k | (base) |
| + Ridge | 54.0 | 82.7 | $402k | 79% |
| **+ XGBoost** | **53.3** | **82.7** | **$397k** | **92%** |
| + LightGBM | 53.9 | 83.6 | $401k | 81% |
| Oracle | 52.9 | 81.9 | $395k | 100% |

5 days ahead, the order is the same: XGBoost captures 78%, LightGBM 71% and Ridge 69%.

The margin-at-risk band is breached on **5.1%** of days at both horizons:

| Corridor | Breached |
|---|---|
| Mexico | 5.2% |
| India | 3.4% |
| Brazil | 6.6% |

**What the results show:**

- **Most of the error is unforecastable by design.** About 53 of 58 bps is the FX move during
  settlement. Pretending otherwise would mean overfitting noise.
- **The markup gap carries most of the signal.** It accounts for most of what stage 2 learns:
  stage 1 is a lagged average of noisy margins, while the business already knows today's markup.
- **Tree models beat Ridge through non-linear effects.** SHAP shows small extra effects from
  volatility, 10-day FX momentum and the dollar index.
- **The value is in risk as much as accuracy.** A calibrated 5% band tells treasury how much margin
  could be lost on a bad day, per corridor. Brazil, the most volatile, gets the widest band.

## 9. Assumptions and limitations

- **The business layer is simulated.** Real pricing policies, settlement patterns and hedging are
  more complex. The FX rates are real.
- **Weak signal.** A pricing policy that's highly predictable from known information makes stage 2
  look good. With less predictable markups, the gain would be smaller, but the method and the
  oracle comparison still apply.
- **Volumes are taken as known** when translating to USD.
- **Conformal calibration assumes the recent past resembles the near future.** In a regime change,
  coverage can drift (for example Brazil at 6.6%) until the next refit.
- **The 4-day knowledge lag is conservative.** Some margins are known after 1–2 days.

## 10. Reproducing

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py            # about 25 seconds
python scripts/run_pipeline.py --tune     # re-run the Optuna search first
pytest
```

The notebook version of this document, [`notebooks/methodology.ipynb`](notebooks/methodology.ipynb),
is generated from this file by `python scripts/build_methodology_notebook.py`.
