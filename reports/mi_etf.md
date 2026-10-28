# Lag-1 mutual information: I(x_{t-1}; y_t)

Estimator: kNN (k=10) in nats, converted to bits. Null: 200 shuffles of the
target series, which keeps the marginals and destroys the dependence - so `shuffle mean`
is what the bias alone produces and `excess bits` is the estimate that matters.

Observations per series: 159142.SZ=207, 512760.SH=1785, 513100.SH=3213. Pairs involving 159142 are limited by its 2025-12-12 listing.

Bonferroni threshold for 9 directed pairs: p < 0.0056.

## Signed returns (direction)

| source (t-1) | target (t) | n | Pearson | MI bits | shuffle mean | shuffle p95 | excess bits | p |
|---|---|---|---|---|---|---|---|---|
| 159142.SZ | 159142.SZ | 206 | -0.0839 | 0.0554 | 0.0153 | 0.0706 | +0.0401 | 0.100 |
| 513100.SH | 159142.SZ | 206 | -0.0777 | 0.0484 | 0.0128 | 0.0516 | +0.0356 | 0.075 |
| 159142.SZ | 513100.SH | 206 | -0.0652 | 0.0373 | 0.0151 | 0.0574 | +0.0221 | 0.154 |
| 512760.SH | 159142.SZ | 206 | -0.0248 | 0.0211 | 0.0133 | 0.0533 | +0.0078 | 0.249 |
| 513100.SH | 513100.SH | 3212 | -0.0705 | 0.0076 | 0.0039 | 0.0158 | +0.0037 | 0.219 |
| 512760.SH | 513100.SH | 1782 | -0.0035 | 0.0000 | 0.0050 | 0.0198 | -0.0050 | 1.000 |
| 512760.SH | 512760.SH | 1784 | +0.0327 | 0.0000 | 0.0053 | 0.0217 | -0.0053 | 1.000 |
| 513100.SH | 512760.SH | 1782 | +0.0008 | 0.0000 | 0.0058 | 0.0218 | -0.0058 | 1.000 |
| 159142.SZ | 512760.SH | 206 | +0.0564 | 0.0000 | 0.0136 | 0.0544 | -0.0136 | 1.000 |

## Absolute returns (volatility clustering)

| source (t-1) | target (t) | n | Pearson | MI bits | shuffle mean | shuffle p95 | excess bits | p |
|---|---|---|---|---|---|---|---|---|
| 513100.SH | 513100.SH | 3212 | +0.2451 | 0.0157 | 0.0035 | 0.0145 | +0.0122 | 0.045 |
| 512760.SH | 513100.SH | 1782 | +0.0883 | 0.0033 | 0.0045 | 0.0184 | -0.0012 | 0.383 |
| 513100.SH | 512760.SH | 1782 | +0.0628 | 0.0031 | 0.0057 | 0.0225 | -0.0026 | 0.383 |
| 159142.SZ | 512760.SH | 206 | +0.0204 | 0.0115 | 0.0150 | 0.0620 | -0.0035 | 0.393 |
| 512760.SH | 512760.SH | 1784 | +0.0476 | 0.0011 | 0.0055 | 0.0221 | -0.0044 | 0.498 |
| 513100.SH | 159142.SZ | 206 | -0.0393 | 0.0089 | 0.0140 | 0.0586 | -0.0051 | 0.393 |
| 159142.SZ | 513100.SH | 206 | +0.0815 | 0.0000 | 0.0109 | 0.0544 | -0.0109 | 1.000 |
| 159142.SZ | 159142.SZ | 206 | +0.0080 | 0.0000 | 0.0132 | 0.0514 | -0.0132 | 1.000 |
| 512760.SH | 159142.SZ | 206 | -0.0284 | 0.0000 | 0.0145 | 0.0593 | -0.0145 | 1.000 |

Reading notes:

* `excess bits` is the number to trust; a raw MI that equals its shuffle mean is bias.
* `Pearson` is there to show when MI and correlation agree - if `excess bits` is
  positive while `Pearson` is ~0, the dependence is nonlinear (or the estimate is noise).
* Volatility clustering is expected to be the *real* dependence: it is the one thing
  daily equity data reliably has. Directional predictability is what the forecasting
  model would need, and it is what this table tests.

