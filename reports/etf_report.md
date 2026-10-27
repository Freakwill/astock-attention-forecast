# ETF experiment: 159142 / 512760 / 513100

Data through the **2026-10-26** close, generated 2026-10-27. Everything below is reproducible with
the commands at the end; no number here is hand-entered.

Universe:

| code | name | what it tracks | history used |
|---|---|---|---|
| 159142.SZ | ChiNext AI ETF | ChiNext AI index (domestic growth / AI capex) | 2025-12-12 → 2026-10-26 (208 days) |
| 512760.SH | ChinaAMC Semiconductor ETF | CSI semiconductor index | 2019-06-12 → 2026-10-26 (1787 days) |
| 513100.SH | Guotai Nasdaq-100 ETF | Nasdaq-100, QDII, CNY-denominated | 2013-07-31 → 2026-10-26 (3215 days) |
| 000300.SH | CSI300 | benchmark / exogenous feature | 2019-01-02 → 2026-10-26 |

## 1. Data layer change: akshare first

`config/data.yaml` now declares `sources: [akshare, baostock, tushare]`. What that required:

* **ETF support**: `Symbol.kind = "etf"`, bare codes resolved by prefix (51x/52x/56x/58x → SH,
  15x/16x/18x → SZ), and a `fund_etf_hist_sina` branch in the akshare adapter (0.2–0.3 s/symbol).
* **A feature had to go**: the Sina ETF route exposes no turnover and no outstanding share count, so
  `turn` is absent and `config/dataset_etf.yaml` drops it rather than inventing a proxy. The tensor is
  11 channels (7 price + 4 benchmark-relative) instead of 12.
* **One fallback was still exercised**: akshare's `stock_zh_index_daily` returned a malformed JSON
  payload for CSI300 on this run, so the chain fell through to baostock for the benchmark only
  (`sources used: {'akshare': [3 ETFs], 'baostock': ['000300.SH']}`). The chain exists for exactly
  this; if you want zero baostock involvement, use a CSI300 ETF (e.g. 510300) as the benchmark.
* **Prices are unadjusted** on the ETF route. ETF distributions are rare; a large distribution would
  show up as a fake drawdown, so re-check if either fund distributes.
* The common window is set by 159142's listing: `aaf fetch` dropped 1685 dates and kept 208.

## 2. The self-attention model has no skill here

`x = (188, 3, 11)`, 127 train / 24 validation / 29 test timesteps. Test window 2026-09-08 → 2026-10-26.

| model | epochs | val MSE | MAE % | RMSE % | direction | bias pp |
|---|---|---|---|---|---|---|
| naive:momentum | – | 34.19 | **3.225** | **4.228** | **0.631** | +0.70 |
| naive:mean | – | 20.55 | 3.302 | 4.299 | 0.553 | +1.12 |
| naive:zero | – | 19.79 | 3.411 | 4.374 | n/a | +1.16 |
| **transformer (self-attention)** | 21 | 20.71 | 3.422 | 4.383 | 0.414 | +1.22 |
| var | – | 25.71 | 3.585 | 4.690 | 0.481 | +1.07 |
| lstm | 21 | 21.95 | 3.614 | 4.731 | 0.442 | +1.24 |
| linear | 23 | 37.47 | 4.088 | 5.352 | 0.661 | −1.54 |

Read it plainly: **the attention model is indistinguishable from predicting zero** (3.422 % vs
3.411 % MAE) and its direction accuracy is below a coin flip (0.414 on 87 observations). `linear` has
the best direction (0.661) and the worst MAE — with 87 overlapping observations every one of these
gaps is noise. The model cannot be rescued by tuning: 127 windows is not enough to fit a cross-asset
attention model, and the honest statement is that **the forecast below is not a validated signal**.

## 3. Forward forecast, 2026-10-27 → 2026-11-02

Next five trading days from the 2026-10-26 close (calendar-checked):

`2026-10-27, 2026-10-28, 2026-10-29, 2026-10-30, 2026-11-02`

Raw model output, forward log return in %:

| asset | t+1 | t+2 | t+3 | t+4 | t+5 |
|---|---|---|---|---|---|
| 159142.SZ ChiNext AI | +0.18 | +0.35 | +0.53 | +0.62 | +0.73 |
| 512760.SH Semiconductor | −0.37 | −0.73 | −1.10 | −1.54 | −1.99 |
| 513100.SH Nasdaq-100 | +0.17 | +0.36 | +0.53 | +0.67 | +0.84 |

Bias-corrected with the validation-slice error (24 windows, so the estimate is itself noisy):

| asset | t+1 | t+2 | t+3 | t+4 | t+5 |
|---|---|---|---|---|---|
| 159142.SZ | +0.39 | +0.51 | +0.75 | +0.55 | +0.42 |
| 512760.SH | −0.25 | −0.61 | −0.69 | −1.18 | −1.82 |
| 513100.SH | +0.47 | +0.82 | +1.12 | +1.33 | +1.44 |

The *shape* is consistent across both views: chips down and worsening with horizon, the two growth
ETFs mildly up. Treat it as the model's opinion, not an expectation — see section 2.

## 4. LLM view (DeepSeek `deepseek-chat`, news-grounded)

| asset | view | confidence | expected % |
|---|---|---|---|
| 159142.SZ | flat | 0.42 | +0.30 |
| 512760.SH | down | 0.55 | −2.50 |
| 513100.SH | down | 0.55 | −1.20 |

The grounding is concrete and checkable, which is the point of quoting it:

* **159142**: 20-day −16.6 %, 60-day −20.0 %, near one-year valuation lows; rebound to 1.03 on 10-23
  (+4.98 %) followed by −2.52 % to 1.01 → two-way, not a clean trend; reported consecutive net
  inflows; no exchange filings. Risks: 2.6 % daily realised vol, high-beta exposure to hawkish rates.
* **512760**: 20-day −22.0 %, 60-day −29.1 %, fresh low 0.86 on 10-26 with −3.7 % that erased the
  prior +3.0 % bounce; ~RMB 638–722 mn of net inflows over 30 days that have *not* stopped the
  decline. Risks: oversold short-covering, policy/national-fund buying.
* **513100**: five consecutive down sessions, 2.38 → 2.24; 20-day −1.45 % against a still-positive
  60-day +6.27 %; **repeated 10:30 trading halts** (QDII premium/discount stress); sticky core PCE as
  a valuation headwind.

Agreement with the model: chips down in both views, Nasdaq-100 down for the LLM vs mildly up for the
model, AI flat vs mildly up. Neither has a track record: the overlay stores every call with its
`as_of` and grades it once the horizon elapses, and says *"no earlier calls with an elapsed horizon
yet"*.

## 5. Trading rules and what they actually did

Three rules, all long/cash only, **no leverage**, 3 bp per side on turnover, weights set from the
previous close only. Sizing is inverse volatility: `weight = min(50 %, 15 % / realised vol)`, then
scaled down if the sum would exceed 100 %.

* **trend** — hold a sleeve when `close > MA20 > MA60`, else cash.
* **vol_target_core** — always invested, sized by inverse volatility (no trend filter).
* **rotation** — hold the top 2 by 60-day momentum, only while that momentum is positive.
* **buy_and_hold** — equal weight across the ETFs listed on each date, always invested.

Backtest 2013-07-31 → 2026-10-26 (3216 days), net of costs:

| strategy | CAGR % | vol % | Sharpe | max DD % | Calmar | exposure % | turnover/yr |
|---|---|---|---|---|---|---|---|
| **buy_and_hold** | **19.46** | 22.87 | **0.89** | −35.96 | **0.54** | 100.0 | 0.44 |
| vol_target_core | 13.58 | 17.71 | 0.81 | −33.83 | 0.40 | 71.9 | 1.98 |
| trend (MA20/60) | 6.54 | **9.77** | 0.70 | **−17.72** | 0.37 | 30.7 | 14.56 |
| rotation (60d, top 2) | 7.52 | 13.88 | 0.59 | −30.53 | 0.25 | 48.2 | 10.85 |

**No rule beat buy & hold on return or Sharpe.** The trend rule's only win is drawdown: −17.7 %
against −36.0 %, at the cost of two thirds of the return. That is the honest summary of "a profitable
strategy" on this basket — the profitable strategy was to own it, and the rules are drawdown
insurance with a large premium.

Per-ETF, trend sleeve vs hold:

| ETF | trend CAGR % | trend max DD % | buy&hold CAGR % | buy&hold max DD % |
|---|---|---|---|---|
| 159142.SZ | 0.08 | −6.84 | 0.10 | −38.10 |
| 512760.SH | 2.54 | −19.11 | 4.57 | −62.23 |
| 513100.SH | 3.87 | −15.06 | 20.91 | −28.57 |

513100 is the clearest case: the trend rule steps out of a structurally rising asset and gives up
17 pp of annual return to avoid a drawdown that was smaller than a third of the return it lost.

**Parameter sensitivity** — the trend result is not an unlucky parameter choice:

| fast | slow | CAGR % | Sharpe | max DD % | exposure % |
|---|---|---|---|---|---|
| 10 | 30 | 6.15 | 0.67 | −15.96 | 29.2 |
| 20 | 60 | 6.54 | 0.70 | −17.72 | 30.7 |
| 60 | 120 | 6.34 | 0.66 | −15.00 | 35.2 |
| 120 | 200 | 6.37 | 0.66 | −20.19 | 38.4 |

Six percent CAGR and a 0.66–0.70 Sharpe across every MA pair: the idea fails here, not the numbers.

## 6. What the indicators say today, and the resulting allocation

As of the 2026-10-26 close:

| asset | close | MA20 | MA60 | 20d mom % | 60d mom % | 20d vol % | dd from 60d high % | above MA20 | MA20>MA60 |
|---|---|---|---|---|---|---|---|---|---|
| 159142.SZ | 1.006 | 1.086 | 1.161 | −16.58 | −20.03 | 41.7 | −22.91 | no | no |
| 512760.SH | 0.858 | 0.965 | 1.051 | −22.00 | −29.09 | 31.8 | −29.38 | no | no |
| 513100.SH | 2.236 | 2.329 | 2.245 | −1.45 | +6.27 | 20.0 | −6.83 | no | yes |

All three are below their MA20; two are also below MA60. The rule therefore holds **0 % and 100 %
cash**, which is a position, not a non-answer: the strategy is designed to be out when nothing is
above its short-term trend.

If you must be invested (or want a core while waiting), the inverse-vol core sizes to
**159142 27 %, 512760 35 %, 513100 38 %** (the raw 36/47/50 request is scaled to 100 %); realised vol
is 41.7 %, 31.8 % and 20.0 %, so the AI ETF gets the smallest slice despite the highest expected
upside.

### The rule, stated so it can be executed tomorrow

1. **Sizing**: `w_i = min(50 %, 15 % / vol20_i)`, normalised to ≤ 100 %, remainder in cash. Recompute
   weekly, not daily, to keep turnover near 2×/yr (the daily version costs 14×/yr in the trend sleeve).
2. **Entry** for a sleeve currently at zero: `close > MA20` **and** `MA20 > MA60` for two consecutive
   closes. Today that needs 159142 above 1.086, 512760 above 0.965, 513100 above 2.329 *and* MA20
   above 2.245 for 513100.
3. **Exit**: `close < MA20` for two consecutive closes → back to cash for that sleeve.
4. **Hard stop** independent of the rule: −12 % from the sleeve's 60-day high, or a single −7 % day.
   This is what the trend rule is *for*; do not let the entry rule override it.
5. **513100 special cases**: it is a QDII. Check the premium/discount before buying — the LLM flagged
   repeated 10:30 halts, which is premium stress, and buying a 3 % premium is a guaranteed loss if it
   normalises. Also carry the USD/CNY exposure and the fact that the overnight Nasdaq move is already
   in the open.
6. **512760**: the driver is the chip capex cycle and policy, not momentum. Momentum says down; the
   inflow data (RMB 638–722 mn/30d) says someone is buying. The rule stays out — that is the correct
   response to a −29 % 60-day trend even if a bounce follows.

## 7. What would invalidate this

* Buy & hold's 19.46 % CAGR is dominated by a Nasdaq-100 ETF bought in 2013; a repeat requires the
  same US tech regime. The three-ETF basket is not diversified — all three are long-duration growth.
* 159142 has ten months of history. Nothing about it is estimable: its trend sleeve "won" 0.08 % vs
  0.10 % and both numbers are noise.
* The 29-window test set (87 observations) cannot rank the models. The ranking in section 2 is a
  description of this window, not of the models.
* Costs are 3 bp/side. The trend sleeve turns over 14.6×/yr; at 10 bp/side its edge disappears
  entirely.

## 8. Reproduce

```bash
source .venv/bin/activate
aaf fetch --group=etf --name=panel_etf                       # akshare first, 3 ETFs + CSI300
aaf tensors --name=panel_etf --feature_config=config/dataset_etf.yaml \
            --out=data/cache/tensors_etf.npz
aaf benchmark --tensors_path=data/cache/tensors_etf.npz --config=config/model_etf.yaml \
              --out_dir=reports/runs/etf --report=reports/benchmark_etf.md
aaf forecast --run reports/runs/etf/transformer_<stamp> \
             --tensors_path=data/cache/tensors_etf.npz --out=reports/forecast_etf.md
aaf llm --config=config/llm_etf.yaml --symbols="159142,512760,513100" \
        --run=reports/runs/etf/transformer_<stamp> --out=reports/llm_overlay_etf.json
aaf strategy --codes=159142,512760,513100 --out=reports/etf_strategy.md
```

Generated artefacts: [`benchmark_etf.md`](benchmark_etf.md), [`forecast_etf.md`](forecast_etf.md),
[`llm_overlay_etf.md`](llm_overlay_etf.md), [`etf_strategy.md`](etf_strategy.md).
