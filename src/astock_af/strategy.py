"""Rules-based ETF allocation: trend following, volatility targeting, momentum rotation.

Why rules rather than the attention model: on the three-ETF panel the model gets
only ~127 training windows (159142 listed 2025-12-12) and the benchmark shows it
is indistinguishable from predicting zero (MAE 3.42 % vs 3.41 %, direction 0.414).
A parameter-light rule set that can be validated over 6-13 years of history is the
honest alternative, and it is also what "what indicator, what weight" needs.

Everything here is long/cash only, uses data up to ``t-1`` to set the weight for
``t``, and charges a cost on every unit of turnover, so the reported numbers are
not the usual no-cost, same-bar fantasy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252
COST_PER_SIDE = 0.0003  # 3 bp: ETF commission ~1-2 bp plus slippage


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """Per-day portfolio path plus the summary statistics."""

    name: str
    returns: pd.Series
    weights: pd.DataFrame
    stats: dict[str, float]


def load_history(codes: Sequence[str], cache_dir: str | Path = "data/cache") -> pd.DataFrame:
    """Close-price matrix (one column per ETF) from the token-less akshare route.

    The panel is *not* reused: it is intersected on 159142's 2025-12-12 listing,
    which would leave a backtest of a few months. Each ETF is fetched over its own
    full history instead, so 512760 (2019) and 513100 (2013) can be validated.
    """
    import akshare as ak

    from .data.symbols import parse

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.Series] = {}
    for raw in codes:
        symbol = parse(raw, kind="etf")
        path = cache / f"etf_hist_{symbol.code}.parquet"
        if path.is_file():
            frame = pd.read_parquet(path)
        else:
            frame = ak.fund_etf_hist_sina(symbol=symbol.sina)
            frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
            frame = frame[["date", "close", "amount"]].sort_values("date")
            frame.to_parquet(path, index=False)
        series = frame.set_index("date")["close"].astype(float)
        series.index = pd.to_datetime(series.index)
        frames[symbol.ts_code] = series

    prices = pd.DataFrame(frames).sort_index()
    # Outer join on purpose: dropping rows where an ETF has no price yet would cut
    # the history back to 159142's 2025-12-12 listing and make a 6-year backtest
    # impossible. A NaN means "not listed / not trading", which every strategy
    # treats as a zero weight rather than as a price.
    prices = prices.loc[prices.notna().any(axis=1)]
    if prices.empty:
        raise ValueError("no overlapping ETF history - check the codes")
    return prices


def indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-ETF trend / momentum / risk readings in long format (asset, date, ...)."""
    rows = []
    for code in prices.columns:
        close = prices[code].astype(float)
        log_return = np.log(close).diff()
        frame = pd.DataFrame(
            {
                "asset": code,
                "date": close.index,
                "close": close.to_numpy(),
                "ma20": close.rolling(20, min_periods=20).mean().to_numpy(),
                "ma60": close.rolling(60, min_periods=60).mean().to_numpy(),
                "mom20": (close / close.shift(20) - 1.0).to_numpy(),
                "mom60": (close / close.shift(60) - 1.0).to_numpy(),
                "vol20": log_return.rolling(20, min_periods=20).std().to_numpy() * np.sqrt(TRADING_DAYS),
                "dd60": (close / close.rolling(60, min_periods=20).max() - 1.0).to_numpy(),
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _vol_target_weights(vol: np.ndarray, active: np.ndarray, target_vol: float, cap: float) -> np.ndarray:
    """Inverse-volatility weights among the active sleeves, capped per ETF.

    A sleeve with ``vol`` above ``target_vol`` gets less than an equal share; a
    quiet sleeve gets more, but never more than ``cap``. Inactive sleeves get 0 and
    the remainder stays in cash - there is no leverage in this implementation.
    """
    weights = np.zeros_like(vol, dtype=float)
    if not active.any():
        return weights
    # A NaN volatility (not listed yet, or not enough history) must not become a
    # weight: treat it as infinitely volatile so the sleeve stays in cash.
    vol = np.where(np.isfinite(vol), vol, np.inf)
    raw = np.zeros_like(weights)
    raw[active] = np.minimum(cap, target_vol / np.maximum(vol[active], 1e-6))
    # Never exceed 100% invested: inverse-vol sizing can ask for more than the
    # portfolio has (three sleeves at 15% target against 20-40% realised vol), and
    # silently going to 133% would be leverage the strategy claims not to use.
    total = float(raw.sum())
    if total > 1.0:
        raw = raw / total
    return raw


def trend_strategy(
    prices: pd.DataFrame,
    target_vol: float = 0.15,
    cap: float = 0.5,
    cost: float = COST_PER_SIDE,
    fast: int = 20,
    slow: int = 60,
    name: str | None = None,
) -> StrategyResult:
    """Long when price > MA(fast) > MA(slow), flat otherwise, inverse-vol sized.

    The MA-cross filter is deliberately slow: it is there to avoid owning a
    downtrend, not to time turns. Signals are shifted one day, so the weight for
    day ``t`` only uses closes through ``t-1``. ``fast``/``slow`` are exposed so
    the sensitivity table in the report can show how much of the result is the
    parameter choice rather than the idea.
    """
    close = prices.astype(float)
    ma_fast = close.rolling(fast, min_periods=fast).mean()
    ma_slow = close.rolling(slow, min_periods=slow).mean()
    vol20 = np.log(close).diff().rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS)

    active = (close > ma_fast) & (ma_fast > ma_slow)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for i in range(len(close)):
        if i == 0:
            continue
        signals = active.iloc[i - 1].to_numpy()  # decision from yesterday's close
        vol = vol20.iloc[i - 1].to_numpy()
        weights.iloc[i] = _vol_target_weights(vol, signals, target_vol, cap)

    returns = (close.pct_change(fill_method=None).fillna(0.0) * weights.shift(1).fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    returns = returns - turnover * cost
    label = name or f"trend_ma{fast}_{slow}"
    return StrategyResult(name=label, returns=returns, weights=weights, stats=_stats(returns, weights, turnover))


def vol_target_strategy(
    prices: pd.DataFrame,
    target_vol: float = 0.15,
    cap: float = 0.5,
    cost: float = COST_PER_SIDE,
    name: str = "vol_target_core",
) -> StrategyResult:
    """Always invested, sized by inverse volatility: the risk-managed core.

    No trend filter, so it never pays the whipsaw cost of stepping out of a
    structural uptrend; the only risk control is that a volatile sleeve gets a
    smaller weight and the leftover stays in cash.
    """
    close = prices.astype(float)
    vol20 = np.log(close).diff().rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for i in range(len(close)):
        if i == 0:
            continue
        available = close.iloc[i - 1].notna().to_numpy()
        vol = vol20.iloc[i - 1].to_numpy()
        weights.iloc[i] = _vol_target_weights(vol, available, target_vol, cap)

    returns = (close.pct_change(fill_method=None).fillna(0.0) * weights.shift(1).fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    returns = returns - turnover * cost
    return StrategyResult(name=name, returns=returns, weights=weights, stats=_stats(returns, weights, turnover))


def rotation_strategy(
    prices: pd.DataFrame,
    lookback: int = 60,
    top_k: int = 2,
    target_vol: float = 0.15,
    cap: float = 0.5,
    cost: float = COST_PER_SIDE,
    name: str = "rotation",
) -> StrategyResult:
    """Hold the ``top_k`` ETFs by ``lookback``-day momentum, only when it is positive.

    Absolute momentum is the risk switch: if fewer than ``top_k`` names have a
    positive lookback return, the remainder sits in cash rather than being forced
    into the least-bad loser.
    """
    close = prices.astype(float)
    momentum = close / close.shift(lookback) - 1.0
    vol20 = np.log(close).diff().rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS)

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for i in range(len(close)):
        if i < max(lookback, 20):
            continue
        mom = momentum.iloc[i - 1]
        vol = vol20.iloc[i - 1].to_numpy()
        ranked = mom.sort_values(ascending=False)
        chosen = [code for code in ranked.index[:top_k] if ranked[code] > 0]
        active = np.array([code in chosen for code in close.columns])
        weights.iloc[i] = _vol_target_weights(vol, active, target_vol, cap)

    returns = (close.pct_change(fill_method=None).fillna(0.0) * weights.shift(1).fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    returns = returns - turnover * cost
    return StrategyResult(name=name, returns=returns, weights=weights, stats=_stats(returns, weights, turnover))


def buy_and_hold(prices: pd.DataFrame, cost: float = COST_PER_SIDE) -> StrategyResult:
    """Equal-weight, always invested: the comparison any strategy has to beat."""
    close = prices.astype(float)
    # Equal weight across the ETFs that are listed and trading on that day, so a
    # later listing does not silently get 1/N of the portfolio from the start.
    available = close.notna()
    weights = available.div(available.sum(axis=1), axis=0).fillna(0.0)
    returns = (close.pct_change(fill_method=None).fillna(0.0) * weights.shift(1).fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    returns = returns - turnover * cost
    return StrategyResult(
        name="buy_and_hold", returns=returns, weights=weights, stats=_stats(returns, weights, turnover)
    )


def _stats(returns: pd.Series, weights: pd.DataFrame, turnover: pd.Series) -> dict[str, float]:
    """CAGR / vol / Sharpe / drawdown / exposure, all from the daily net path."""
    clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = (1.0 + clean).cumprod()
    years = max(len(clean) / TRADING_DAYS, 1e-9)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if equity.iloc[-1] > 0 else -1.0
    vol = float(clean.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(clean.mean() / clean.std() * np.sqrt(TRADING_DAYS)) if clean.std() > 0 else 0.0
    drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "cagr_pct": round(cagr * 100.0, 2),
        "vol_pct": round(vol * 100.0, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(drawdown * 100.0, 2),
        "calmar": round(cagr / abs(drawdown), 2) if drawdown < 0 else float("nan"),
        "exposure_pct": round(float(weights.sum(axis=1).mean() * 100.0), 1),
        "turnover_yr": round(float(turnover.sum() / years), 2),
        "days": int(len(clean)),
    }


def current_view(prices: pd.DataFrame, target_vol: float = 0.15, cap: float = 0.5) -> pd.DataFrame:
    """Latest readings, the trend rule's weight, and the vol-target core weight.

    This is the table to look at each evening: it answers "is it above its moving
    averages, how fast is it, how volatile, how deep is the drawdown, and what
    weight does that imply".
    """
    table = indicators(prices)
    latest = table.sort_values("date").groupby("asset", as_index=False).last()
    latest["above_ma20"] = latest["close"] > latest["ma20"]
    latest["ma20_above_ma60"] = latest["ma20"] > latest["ma60"]
    core = np.minimum(cap, target_vol / latest["vol20"].clip(lower=1e-6))
    latest["core_weight"] = np.where(latest["vol20"].notna(), core, 0.0).round(3)
    active = (latest["above_ma20"] & latest["ma20_above_ma60"]).to_numpy()
    trend_weight = np.zeros(len(latest))
    trend_weight[active] = core.to_numpy()[active]
    latest["trend_weight"] = trend_weight.round(3)
    latest["final_weight"] = np.minimum(latest["core_weight"], latest["trend_weight"]).round(3)
    return latest[
        [
            "asset",
            "date",
            "close",
            "ma20",
            "ma60",
            "mom20",
            "mom60",
            "vol20",
            "dd60",
            "above_ma20",
            "ma20_above_ma60",
            "core_weight",
            "trend_weight",
            "final_weight",
        ]
    ]
