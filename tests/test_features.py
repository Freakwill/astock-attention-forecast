"""Tensor / feature-engineering tests on synthetic panels (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astock_af.config import FeatureConfig
from astock_af.data.features import build_tensors, feature_frame
from astock_af.data.panel import align_panel
from astock_af.models import make_windows, split_windows

SYMBOLS = ["600519.SH", "300750.SZ", "600036.SH"]


def make_panel(n_days: int = 120, symbols: list[str] | None = None, seed: int = 7) -> pd.DataFrame:
    """Canonical-schema panel with a reproducible random walk."""
    symbols = symbols or SYMBOLS
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days).strftime("%Y-%m-%d")
    rows = []
    for index, ts_code in enumerate(symbols):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_days)))
        preclose = np.concatenate([[close[0]], close[:-1]])
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ts_code": ts_code,
                    "name": f"asset{index}",
                    "open": close * 0.995,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "preclose": preclose,
                    "volume": rng.integers(1_000_000, 5_000_000, n_days).astype(float),
                    "amount": rng.integers(100_000_000, 500_000_000, n_days).astype(float),
                    "turn": rng.uniform(0.2, 2.0, n_days),
                    "pct_chg": (close / preclose - 1) * 100,
                    "pe_ttm": 20.0,
                    "pb_mrq": 3.0,
                    "tradestatus": 1,
                    "source": "synthetic",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_feature_frame_columns() -> None:
    frame = feature_frame(make_panel())
    for column in ("log_return", "intraday_range", "log_volume", "turn", "mom_5", "mom_20", "vol_20"):
        assert column in frame.columns


def test_tensor_shapes_and_standardization() -> None:
    cfg = FeatureConfig(lookback=10, horizon=3, features=["log_return", "mom_5", "turn"], include_benchmark=False)
    tensors = build_tensors(make_panel(150), cfg)
    assert tensors.x.shape == (len(tensors.dates), 3, 3)
    assert tensors.y.shape == (len(tensors.dates), 3, 3)
    train = tensors.x[: tensors.n_train]
    # Train slice is standardized on its own statistics.
    assert np.allclose(train.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(train.std(axis=0), 1.0, atol=1e-4)
    assert not np.isnan(tensors.x).any()


def test_targets_are_forward_log_returns() -> None:
    panel = make_panel(150)
    cfg = FeatureConfig(lookback=10, horizon=4, features=["log_return"], include_benchmark=False)
    tensors = build_tensors(panel, cfg)
    close = panel.pivot(index="date", columns="ts_code", values="close").loc[tensors.dates][tensors.symbols]
    for t in (0, 40, len(tensors.dates) - tensors.horizon - 1):
        for h in range(1, tensors.horizon + 1):
            expected = np.log(close.iloc[t + h].to_numpy() / close.iloc[t].to_numpy())
            assert np.allclose(tensors.y[t, :, h - 1], expected, atol=1e-5)


def test_split_bounds_leave_a_purge_gap() -> None:
    cfg = FeatureConfig(lookback=10, horizon=5, features=["log_return"], include_benchmark=False)
    tensors = build_tensors(make_panel(200), cfg)
    splits = split_windows(tensors)
    assert len(splits.x_train) and len(splits.x_val) and len(splits.x_test)
    bounds = tensors.split_bounds(purge=tensors.horizon)
    assert bounds["train"][1] + tensors.horizon == bounds["val"][0]
    assert bounds["test"][0] - bounds["val"][1] == tensors.horizon


def test_train_windows_stop_before_test_starts() -> None:
    """A training window's 5-day target must not reach into the test period."""
    cfg = FeatureConfig(lookback=10, horizon=5, features=["log_return"], include_benchmark=False)
    tensors = build_tensors(make_panel(200), cfg)
    splits = split_windows(tensors)
    test_start = tensors.split_bounds(purge=tensors.horizon)["test"][0]
    train_origins = np.arange(0, len(splits.x_train)) + (tensors.lookback - 1)
    assert train_origins.max() + tensors.horizon <= test_start


def test_make_windows_matches_manual_slice() -> None:
    cfg = FeatureConfig(lookback=6, horizon=2, features=["log_return"], include_benchmark=False)
    tensors = build_tensors(make_panel(60), cfg)
    x, y, origins = make_windows(tensors)
    assert x[5].shape == (6, tensors.n_assets, tensors.n_features)
    assert np.allclose(x[5], tensors.x[origins[5] - 5 : origins[5] + 1])
    assert np.allclose(y[5], tensors.y[origins[5]])


def test_align_panel_drops_suspension_dates() -> None:
    panel = make_panel(10)
    suspended = panel[(panel["ts_code"] == "300750.SZ") & (panel["date"] == panel["date"].iloc[3])].index
    panel.loc[suspended, "tradestatus"] = 0
    panel.loc[suspended, ["volume", "amount", "turn"]] = np.nan
    aligned = align_panel(panel, how="intersection", verbose=False)
    assert panel["date"].iloc[3] not in set(aligned["date"])
    assert aligned.groupby("date")["ts_code"].nunique().eq(3).all()


def test_align_panel_outer_keeps_everything() -> None:
    panel = make_panel(10, symbols=["600519.SH", "300750.SZ"])
    extra = panel[panel["ts_code"] == "600519.SH"].copy()
    extra["ts_code"] = "600036.SH"
    extra["date"] = extra["date"].shift(-1).ffill()
    combined = pd.concat([panel, extra], ignore_index=True)
    assert len(align_panel(combined, how="outer", verbose=False)) == len(combined)


def test_empty_panel_raises() -> None:
    panel = make_panel(5)
    panel.loc[panel["tradestatus"] == 0] = 0
    panel["tradestatus"] = 0
    with pytest.raises(Exception):
        aligned = align_panel(panel, how="intersection", verbose=False)
        build_tensors(aligned, FeatureConfig(lookback=2, horizon=1, features=["log_return"]))
