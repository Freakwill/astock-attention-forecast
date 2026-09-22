# Forward forecast - as of 2026-09-22

Target trading days: 2026-09-23, 2026-09-24, 2026-09-28, 2026-09-29, 2026-09-30  (exchange calendar (akshare/sina))

Values are forward log returns in %, relative to the as-of close.

## Raw model output

| asset | t+1 | t+2 | t+3 | t+4 | t+5 |
|---|---|---|---|---|---|
| CATL (300750.SZ) | -1.26 | -1.16 | -1.06 | -0.97 | -0.88 |
| China Merchants Bank (600036.SH) | -1.34 | -1.32 | -1.30 | -1.28 | -1.26 |
| Kweichow Moutai (600519.SH) | -1.36 | -1.35 | -1.35 | -1.34 | -1.34 |
| SMIC (688981.SH) | -1.30 | -1.24 | -1.18 | -1.12 | -1.06 |

## Validation-slice bias (mean prediction - realised, pp)

| asset | t+1 | t+2 | t+3 | t+4 | t+5 |
|---|---|---|---|---|---|
| CATL (300750.SZ) | -0.52 | -0.69 | -0.86 | -1.05 | -1.24 |
| China Merchants Bank (600036.SH) | -0.42 | -0.49 | -0.57 | -0.65 | -0.74 |
| Kweichow Moutai (600519.SH) | -0.34 | -0.33 | -0.33 | -0.32 | -0.32 |
| SMIC (688981.SH) | -0.49 | -0.63 | -0.78 | -0.97 | -1.17 |

## Bias-corrected forecast (raw minus validation bias)

| asset | t+1 | t+2 | t+3 | t+4 | t+5 |
|---|---|---|---|---|---|
| CATL (300750.SZ) | -0.74 | -0.48 | -0.21 | +0.08 | +0.36 |
| China Merchants Bank (600036.SH) | -0.92 | -0.83 | -0.73 | -0.63 | -0.52 |
| Kweichow Moutai (600519.SH) | -1.01 | -1.02 | -1.02 | -1.02 | -1.02 |
| SMIC (688981.SH) | -0.81 | -0.61 | -0.40 | -0.15 | +0.12 |

The correction is estimated on the validation slice, which precedes the as-of date, so it
is not look-ahead. It is still an estimate: if the current regime differs from validation,
the correction is wrong by that difference.
