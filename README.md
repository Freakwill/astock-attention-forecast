# astock-attention-forecast

Multivariate A-share forecasting end to end: a **token-less** data layer, a **self-attention**
(Transformer) multi-step forecaster, an honest baseline ladder, and an **LLM news-sentiment
overlay** that reads the last few days of headlines and filings.

Nothing here needs an API key except the optional DeepSeek overlay and the optional Tushare adapter
(this repo was built and run without a Tushare token, see [Data sources](#data-sources)).

**Headline result** — 4 assets, horizon H = 5 days, lookback L = 30 days, out-of-sample 2025-10-28 →
2026-09-22 (218 windows, cross-sector group):

* **No learned model beats predicting zero on MAE** (2.47 % vs 2.92 %). That is the expected outcome
  for daily returns — the loss-minimising forecast of the conditional mean is a shrink toward zero —
  which is why this repo reports *both* metric families instead of only the flattering one.
* A single chronological split *suggests* the attention model is the best directional forecaster
  (53.0 % vs 50.3 % momentum, 45.6 % trailing mean). **A 5-seed sweep collapses that claim**:
  52.3 % ± 1.1 % for the attention model vs 51.4 % ± 0.6 % for the LSTM, i.e. the effect is the same
  size as the seed variance; on the same-sector group the attention model falls to 48.7 % ± 5.7 %
  while the LSTM reaches 51.7 % ± 2.7 %.
* So the deliverable is **not** "attention beats the market" — it is a pipeline plus an evaluation
  ladder honest enough to catch its own false positive. See [Seed sweep](#seed-sweep-are-the-results-real).

---

## Data sources

Market data is pluggable: `config/data.yaml` declares a **priority chain** and the first usable
source wins per symbol.

| source | token | role in this repo | what it provides |
|---|---|---|---|
| [baostock](https://pypi.org/project/baostock/) | none | **primary** (all market data here) | pre-adjusted (qfq) OHLCV, amount, turnover, PE/PB, `tradestatus`; also indices |
| [akshare](https://akshare.akfamily.xyz/) | none | fallback for prices; **primary for news** | Sina daily bars; news / filings / market newswire |
| [tushare](https://tushare.pro/) | **required** | adapter present, **not exercised** (no token) | at the free 120-credit tier only unadjusted `daily` |

> The Tushare adapter is included so the pipeline can switch vendors with `--source tushare`, but it
> has **not** been run against the live API in this repo — only its unit conversion and schema
> pinning are covered by tests (`tests/test_sources.py`).

**Universe (cross-sector group)** — four different industries so a single attention budget has to
model heterogeneous regimes:

| code | name | sector |
|---|---|---|
| 600519.SH | Kweichow Moutai | Consumer staples |
| 300750.SZ | CATL | Battery |
| 600036.SH | China Merchants Bank | Bank |
| 688981.SH | SMIC | Semiconductor |
| 000300.SH | CSI300 | benchmark, used as exogenous context |

A same-sector control group (white spirit: 600519 / 000858 / 000568 / 600809) is configured as
`group: same_sector`.

### Normalization decisions that matter

* **Units**: everything is converted to CNY and shares (`baostock` already is; the Tushare adapter
  multiplies `vol × 100` and `amount × 1000`). Turnover is normalized to percent of tradable shares.
* **Adjustment**: qfq (forward-adjusted) prices, so a dividend does not look like a crash.
* **Suspensions** are dropped, not filled. Baostock reports a halted day with empty
  `volume/amount/turn`; SMIC has six such days (2025-09-01 → 2025-09-08). The panel builder keeps a
  date only when **every** symbol actually traded (`tradestatus == 1`), so the model never sees a
  fabricated zero-volume day. When a symbol lists later (SMIC: 2020-07-16), the intersection alignment
  drops the earlier dates and reports how many.
* **Provenance**: every cached panel is written with a `*.meta.json` recording the requested range,
  the vendor that served each symbol, the row count and the actual first/last date.

---

## Dataset format

`aaf tensors` turns the long panel into arrays plus a sidecar of split bookkeeping:

```
x : (1477, 4, 12)      # timesteps x assets x features  (z-scored on the training slice only)
y : (1477, 4, 5)       # forward log returns, y[t, n, h-1] = log(close[t+h] / close[t])
dates: 2020-08-13 .. 2026-09-22   (1477 trading days)
```

Sliding windows for the model are `(S, L, N, F) = (999, 30, 4, 12)` train / `(216, …)` val /
`(218, …)` test.

Features (12 channels): `log_return`, `intraday_range`, `log_volume`, `log_amount`, `turn`,
`mom_5`, `mom_20`, `vol_20`, plus benchmark context `bench_return` and relative strength
`rel_log_return`, `rel_mom_5`, `rel_mom_20`.

**Split hygiene** — chronological 70/15/15 with a **purge gap of H = 5 steps** between train and
validation and between validation and test, so no training window's 5-day target can reach into the
next split. Standardization statistics come from the training slice alone; the test rows keep their
real scale. Both properties are asserted in `tests/test_features.py`.

---

## Model

`AttentionForecaster` (`src/astock_af/models/attention.py`):

1. each **timestep is a token** carrying all assets' standardized features (`N×F → d_model`), plus a
   learned positional embedding — so self-attention mixes information across assets in the
   representation;
2. a 2-layer Transformer encoder (`d_model=64`, 4 heads, pre-norm, GELU) produces one memory vector
   per past day;
3. **`N × H` learned queries cross-attend** onto that memory. Each query owns one (asset, horizon)
   pair, and the returned attention map answers *"which past days drove this asset's h-step-ahead
   forecast"* — the model is auditable instead of a black box.

Baselines, scored on identical splits: `lstm`, `linear` (flattened window → linear head), `var`
(ridge vector-autoregression on 5 lags of all assets' returns), and three no-fit references
`naive:zero`, `naive:mean`, `naive:momentum`. The attention weights are also stored per test sample
(`attention.npz`); note they are only meaningful from an `eval()` forward pass, because dropout
perturbs attention weights during training (asserted by a test).

Train with `aaf benchmark`, which writes one run directory per model under `reports/runs/` containing
`metrics.json`, `history.json`, `predictions.npz`, `model.pt` and a three-panel `report.png`
(prediction vs actual, MAE by horizon, attention heatmap).

---

## Results

Cross-sector group, test window 2025-10-28 → 2026-09-22 (218 samples), MPS on an M5 Pro:

| model | epochs | sec | val MSE | MAE % | RMSE % | direction acc | bias % | MAE h1 | h5 |
|---|---|---|---|---|---|---|---|---|---|
| naive:zero | – | 0.0 | 11.56 | **2.471** | **3.654** | 0.003 * | 0.12 | 1.49 | 3.36 |
| naive:mean | – | 0.0 | 11.45 | 2.486 | 3.668 | 0.456 | 0.26 | 1.50 | 3.39 |
| var | – | 0.0 | 12.01 | 2.529 | 3.726 | 0.501 | 0.27 | 1.52 | 3.45 |
| **transformer (self-attention)** | 57 | 8.7 | 14.51 | 2.916 | 4.051 | **0.530** | −1.25 | 2.13 | 3.65 |
| naive:momentum | – | 0.0 | 16.33 | 2.925 | 4.324 | 0.503 | 0.08 | 2.21 | 3.62 |
| lstm | 41 | 4.4 | 14.78 | 3.260 | 4.496 | 0.522 | −0.54 | 2.24 | 4.21 |
| linear | 122 | 2.7 | 19.99 | 4.051 | 5.409 | 0.513 | −0.54 | 2.52 | 5.27 |

\* A constant-zero forecast makes no directional call, so `sign(pred) == sign(true)` is false by
construction; read it as "n/a", not as 0.3 % skill.

Per-asset breakdown for the attention model — MAE tracks each name's volatility, and the directional
edge is concentrated in the two large-cap names:

| asset | MAE % | RMSE % | direction acc |
|---|---|---|---|
| Kweichow Moutai | 2.034 | 2.741 | **0.579** |
| China Merchants Bank | 2.044 | 2.441 | 0.524 |
| CATL | 3.198 | 4.193 | 0.499 |
| SMIC | 4.387 | 5.883 | 0.517 |

Run artefacts: [`reports/benchmark.md`](reports/benchmark.md) (generated), one
`report.png` per model under `reports/runs/`.

### How to read this

* **MAE is a shrinkage metric.** Predicting 0 % every day wins, because daily log returns have a mean
  near zero and a large noise term. Any honest daily-return benchmark has to show this; a repo that
  reports only MAE will accidentally "prove" that a naive constant beats every model.
* **The attention model's directional numbers do not survive reseeding.** The single-split 53.0 %
  looked like a signal and the model's attention maps are genuinely interpretable, but the
  [seed sweep](#seed-sweep-are-the-results-real) shows the edge over an LSTM is inside the noise
  band. Whatever is claimed here has to survive that check first.
* **218 (or 274) overlapping test windows are not independent samples.** Overlapping 5-day targets
  share most of their horizon, so even a seed-stable result would need a block bootstrap or
  walk-forward design before it meant anything.
* **Rolling-origin evaluation would be stronger.** The current split is a single chronological
  holdout. A walk-forward loop (retrain monthly) is the natural next step and is not implemented.

### Same-sector control group

The white-spirit group (Moutai, Wuliangye, Luzhou Laojiao, Shanxi Fenjiu) has a longer history — all
four listed before 2019 — so its panel starts 2019-01-02 and the chronological split lands on a
different test window (2025-08-01 → 2026-09-22, 274 windows). The two groups are therefore **not**
strictly comparable; treat this as a second, independent look rather than a controlled A/B test.

| model | epochs | sec | val MSE | MAE % | RMSE % | direction acc | MAE h1 | h5 |
|---|---|---|---|---|---|---|---|---|
| naive:zero | – | 0.0 | 17.06 | **2.016** | **2.743** | 0.002 * | 1.18 | 2.73 |
| naive:mean | – | 0.0 | 17.24 | 2.081 | 2.807 | 0.395 | 1.20 | 2.85 |
| var | – | 0.0 | 17.05 | 2.091 | 2.830 | 0.434 | 1.20 | 2.85 |
| naive:momentum | – | 0.0 | 19.36 | 2.397 | 3.249 | 0.515 | 1.78 | 3.00 |
| lstm | 39 | 3.8 | 19.87 | 2.520 | 3.359 | **0.526** | 1.81 | 3.37 |
| transformer (self-attention) | 29 | 6.7 | 19.78 | 2.790 | 3.664 | 0.450 | 2.07 | 3.45 |
| linear | 124 | 3.3 | 26.62 | 3.233 | 4.329 | 0.468 | 1.69 | 4.57 |

This is the *opposite* ranking from the cross-sector group: with four nearly collinear consumer
staples, cross-asset attention has little independent information to exploit, and the attention model
is the only one below chance on direction. Reported as-is — it is the reason the seed sweep below
exists.

---

## Seed sweep: are the results real?

A single split with a single seed cannot separate skill from initialisation. `scripts/seed_sweep.py`
retrains the attention model and its LSTM baseline across 5 seeds on both groups:

| group | model | runs | direction acc (mean ± std) | MAE % (mean ± std) |
|---|---|---|---|---|
| cross_sector | transformer (self-attention) | 5 | 0.523 ± 0.011 | 3.226 ± 0.413 |
| cross_sector | lstm | 5 | 0.514 ± 0.006 | 3.495 ± 0.246 |
| same_sector | transformer (self-attention) | 5 | 0.487 ± 0.057 | 2.716 ± 0.396 |
| same_sector | lstm | 5 | 0.517 ± 0.027 | 2.430 ± 0.192 |

Reading:

* the cross-sector attention edge over the LSTM shrinks from "53.0 % vs 52.2 %" (single run) to
  **+0.9 pp** on average, while the seed spread is ±1.1 pp — the effect is **within noise**;
* the same-sector single run (45.0 %) turns out to sit inside a 48.7 % ± 5.7 % distribution: that
  number was an unlucky seed, not a property of the architecture;
* MAE differences of a few tenths of a percent are likewise not separable from seed variance.

**Conclusion: this repo does not demonstrate a robust directional edge for self-attention over an
LSTM on daily A-share returns.** It demonstrates the pipeline, the leakage-safe evaluation, and the
diagnostic that falsifies the tempting single-split claim.

---

## LLM overlay

`aaf llm` asks a large language model (DeepSeek `deepseek-chat`, temperature 0.2, JSON mode) for a
5-day directional view per asset, grounded in:

* the symbol's recent price action (last 10 sessions, 20/60-day returns, realised volatility);
* company news (Eastmoney search API, 8–10 items);
* exchange filings for the last few days;
* market-wide newswire (CLS telegraph) plus the CCTV broadcast (policy context).

The answer is compared against the attention model's live forecast for the same date. Real output
from 2026-09-22:

| asset | LLM view | conf. | expected % | model h1 | agree |
|---|---|---|---|---|---|
| 600519.SH | down | 0.55 | −1.20 | up (+0.28 %) | no |
| 300750.SZ | up | 0.55 | +2.50 | up (+0.40 %) | yes |
| 600036.SH | flat | 0.55 | −0.20 | up (+0.25 %) | no |
| 688981.SH | up | 0.55 | +1.50 | up (+0.34 %) | yes |

The reasoning is grounded in retrievable facts, e.g. for CATL: *"repurchased ~3.69m shares for
RMB1.1bn on 9/21, ~1.51m for RMB456m on 9/18 … execution prices cluster at RMB296–307"*, and for
Moutai: *"2026 interim report showed net profit down 1.95 % YoY"*. Full responses, drivers and risks:
[`reports/llm_overlay.md`](reports/llm_overlay.md); raw headlines: `reports/llm_overlay.news.json`.

**Honest limits.** The free news endpoints only serve *recent* items — there is no archive to replay
last quarter's headlines — so the overlay is **forward-looking and cannot be backtested**. Instead,
every call is persisted with its `as_of` date and `score_previous()` grades it automatically once its
horizon has elapsed, so a track record accumulates across runs rather than being asserted. On a first
run the report says so explicitly: *"no earlier calls with an elapsed horizon yet"*.

Implementation note worth knowing: `akshare.stock_news_em` breaks on pandas 3.x
(`ArrowInvalid: invalid escape sequence: \u`) because string columns are PyArrow-backed and RE2
rejects `\u` escapes, so the overlay fetches Eastmoney over HTTP directly and keeps akshare as a
fallback.

---

## Reproduce

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv -e .

source .venv/bin/activate
aaf sources                       # which vendors are usable right now
aaf fetch                         # baostock panel -> data/cache/panel.parquet (+ meta.json)
aaf tensors                       # panel -> data/cache/tensors.npz
aaf benchmark --epochs=150        # all models, same splits -> reports/benchmark.md + per-run PNGs
aaf evaluate --run reports/runs/transformer_<stamp>     # metrics + attention profile
aaf llm --run reports/runs/transformer_<stamp>          # DeepSeek news overlay
aaf fetch --group=same_sector --name=panel_same         # white-spirit control group
aaf tensors --name=panel_same --out=data/cache/tensors_same.npz
aaf benchmark --tensors_path=data/cache/tensors_same.npz --report=reports/benchmark_same_sector.md
python scripts/seed_sweep.py --seeds=5 --epochs=150      # mean +/- std across seeds
pytest -q                         # 36 tests, no network required
```

`config/data.yaml` (universe, vendors, alignment), `config/dataset.yaml` (L, H, features, split
ratios), `config/model.yaml` (architecture + optimiser), `config/llm.yaml` (model, horizon, prompt
budget).

## Layout

```
src/astock_af/
  config.py              typed YAML config (dataclasses)
  cli.py                 fire CLI: sources | fetch | coverage | tensors | train | benchmark | evaluate | llm
  data/symbols.py        symbol parsing, vendor-specific rendering, ambiguity guard
  data/sources.py        baostock / akshare / tushare adapters, one canonical schema
  data/panel.py          fetch chain, alignment, caching, provenance, coverage report
  data/features.py       feature frame -> (T, N, F) tensor + targets, split bookkeeping
  models/attention.py    the self-attention forecaster
  models/baselines.py    linear / LSTM / ridge-VAR / naive references
  models/dataset.py      windowing, purge-aware splits, torch loaders
  train.py               training loop, early stopping, run artefacts
  evaluate.py            metric tables, benchmark aggregation, attention summary, plots
  llm/news.py            news + filings + market newswire collectors
  llm/client.py          OpenAI-compatible chat client (JSON mode, retries)
  llm/overlay.py         prompt assembly, model-vs-LLM comparison, self-scoring
scripts/seed_sweep.py    seed-variance check: is the directional edge real?
```

## Known limitations

* Daily-return forecasting is close to a noise floor: report direction and calibration, not just MAE,
  and reseed before believing any of it.
* One chronological holdout; no walk-forward retraining, no transaction costs, no position sizing.
* The two asset groups cover different test periods, so cross-group comparison is indicative only.
* The LLM overlay is a single call per asset with no self-consistency check, and its numeric
  `expected_return_pct` should be read as a coarse sentiment score, not a forecast.
* The Tushare adapter is untested against the live API here (no token).
* `akshare`'s Eastmoney price endpoints (push2his) were rejected from this network, which is why the
  Sina endpoints are used for prices.
