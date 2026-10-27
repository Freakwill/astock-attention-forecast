"""Strategy invariants: no leverage, no same-bar look-ahead, honest accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astock_af.strategy import (
    buy_and_hold,
    current_view,
    rotation_strategy,
    trend_strategy,
    vol_target_strategy,
)


def _prices(periods: int = 300, seed: int = 0) -> pd.DataFrame:
    """Two synthetic ETFs with a trend, a crash and different volatilities."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-01", periods=periods)
    trend = np.linspace(0.0, 0.25, periods)
    columns = {
        "AAA.SZ": 1.0 * np.exp(trend + np.cumsum(rng.normal(0, 0.008, periods))),
        "BBB.SH": 1.0 * np.exp(0.5 * trend + np.cumsum(rng.normal(0, 0.020, periods))),
    }
    return pd.DataFrame(columns, index=index)


@pytest.mark.parametrize("factory", [trend_strategy, vol_target_strategy, rotation_strategy, buy_and_hold])
def test_weights_never_exceed_the_portfolio(factory) -> None:
    """Inverse-vol sizing must not quietly become leverage.

    Three sleeves targeting 15% against 20-40% realised vol used to sum to 133%,
    i.e. a strategy advertised as long/cash-only was borrowing 33%.
    """
    result = factory(_prices())
    exposure = result.weights.sum(axis=1)
    assert exposure.max() <= 1.0 + 1e-9, f"{result.name} reached {exposure.max():.3f} exposure"
    assert (result.weights >= -1e-12).all().all(), "long/cash only: no short weights"


def test_trend_uses_yesterday_close_only() -> None:
    """Changing today's close must not change today's weight (no same-bar look-ahead)."""
    prices = _prices()
    base = trend_strategy(prices)
    perturbed = prices.copy()
    perturbed.iloc[-1] *= 1.5  # a jump that happens *after* the decision
    after = trend_strategy(perturbed)
    assert np.allclose(base.weights.iloc[-1], after.weights.iloc[-1])


def test_buy_and_hold_is_equal_weighted_across_listed_assets() -> None:
    """A later listing must not receive weight before its first price exists."""
    prices = _prices(periods=200)
    prices["CCC.SH"] = np.nan
    prices.iloc[120:, 2] = 1.0
    result = buy_and_hold(prices)
    assert np.allclose(result.weights.iloc[10].sum(), 1.0)
    assert result.weights.iloc[10]["CCC.SH"] == 0.0
    assert result.weights.iloc[-1]["CCC.SH"] == pytest.approx(1 / 3)


def test_current_view_reports_the_rule_components() -> None:
    view = current_view(_prices())
    assert {"core_weight", "trend_weight", "final_weight"} <= set(view.columns)
    assert (view["final_weight"] <= view["core_weight"] + 1e-9).all()
    assert view["final_weight"].sum() <= 1.0 + 1e-9
