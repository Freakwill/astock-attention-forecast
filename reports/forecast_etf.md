# Forward forecast - as of 2026-10-26

Target trading days: 2026-10-27, 2026-10-28, 2026-10-29, 2026-10-30, 2026-11-02  (exchange calendar (akshare/sina))

Values are forward log returns in %, relative to the as-of close.

## Raw model output

| asset | t+1 | t+2 | t+3 | t+4 | t+5 |
|---|---|---|---|---|---|
| ChiNext AI ETF (159142.SZ) | +0.18 | +0.35 | +0.53 | +0.62 | +0.73 |
| ChinaAMC Semiconductor ETF (512760.SH) | -0.37 | -0.73 | -1.10 | -1.54 | -1.99 |
| Guotai Nasdaq-100 ETF (513100.SH) | +0.17 | +0.36 | +0.53 | +0.67 | +0.84 |

## Validation-slice bias (mean prediction - realised, pp)

| asset | t+1 | t+2 | t+3 | t+4 | t+5 |
|---|---|---|---|---|---|
| ChiNext AI ETF (159142.SZ) | -0.21 | -0.15 | -0.22 | +0.08 | +0.31 |
| ChinaAMC Semiconductor ETF (512760.SH) | -0.12 | -0.11 | -0.41 | -0.37 | -0.17 |
| Guotai Nasdaq-100 ETF (513100.SH) | -0.29 | -0.46 | -0.60 | -0.66 | -0.60 |

## Bias-corrected forecast (raw minus validation bias)

| asset | t+1 | t+2 | t+3 | t+4 | t+5 |
|---|---|---|---|---|---|
| ChiNext AI ETF (159142.SZ) | +0.39 | +0.51 | +0.75 | +0.55 | +0.42 |
| ChinaAMC Semiconductor ETF (512760.SH) | -0.25 | -0.61 | -0.69 | -1.18 | -1.82 |
| Guotai Nasdaq-100 ETF (513100.SH) | +0.47 | +0.82 | +1.12 | +1.33 | +1.44 |

The correction is estimated on the validation slice, which precedes the as-of date, so it
is not look-ahead. It is still an estimate: if the current regime differs from validation,
the correction is wrong by that difference.
