# ETF strategy - trend, rotation, buy & hold

History: 2013-07-31 .. 2026-10-26 (3216 trading days, 3 ETFs).
Costs: 0.03% per side on turnover. Long/cash only, no leverage.
Sizing: inverse volatility, target 15% annualised, cap 50% per ETF.

## Backtest, net of costs

| strategy | CAGR % | vol % | Sharpe | max DD % | Calmar | exposure % | turnover/yr |
|---|---|---|---|---|---|---|---|
| buy_and_hold | 19.46 | 22.87 | 0.89 | -35.96 | 0.54 | 100.0 | 0.44 |
| vol_target_core | 13.58 | 17.71 | 0.81 | -33.83 | 0.4 | 71.9 | 1.98 |
| trend_ma20_60 | 6.54 | 9.77 | 0.7 | -17.72 | 0.37 | 30.7 | 14.56 |
| rotation | 7.52 | 13.88 | 0.59 | -30.53 | 0.25 | 48.2 | 10.85 |

## Per-ETF trend sleeve vs buy & hold

| ETF | trend CAGR % | trend max DD % | buy&hold CAGR % | buy&hold max DD % |
|---|---|---|---|---|
| 159142.SZ | 0.08 | -6.84 | 0.1 | -38.1 |
| 512760.SH | 2.54 | -19.11 | 4.57 | -62.23 |
| 513100.SH | 3.87 | -15.06 | 20.91 | -28.57 |

## The same idea at other MA lengths (sensitivity, not a menu to pick from)

| fast | slow | CAGR % | Sharpe | max DD % | exposure % |
|---|---|---|---|---|---|
| 10 | 30 | 6.15 | 0.67 | -15.96 | 29.1 |
| 20 | 60 | 6.54 | 0.7 | -17.72 | 30.7 |
| 60 | 120 | 6.34 | 0.66 | -15.0 | 35.2 |
| 120 | 200 | 6.37 | 0.66 | -20.19 | 38.4 |

## Current readings and target weights (as of 2026-10-26, for the next session)

| asset | close | MA20 | MA60 | 20d mom % | 60d mom % | 20d vol % | dd from 60d high % | above MA20 | MA20>MA60 | core w | trend w | final w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 159142.SZ | 1.006 | 1.086 | 1.161 | -16.58 | -20.03 | 41.7 | -22.91 | False | False | 36.0% | 0.0% | 0.0% |
| 512760.SH | 0.858 | 0.965 | 1.051 | -22.00 | -29.09 | 31.8 | -29.38 | False | False | 47.1% | 0.0% | 0.0% |
| 513100.SH | 2.236 | 2.329 | 2.245 | -1.45 | +6.27 | 20.0 | -6.83 | False | True | 50.0% | 0.0% | 0.0% |

Final allocation: invested 0.0%, cash 100.0% (final = min(core, trend)).

