"""Forward-forecast tests: the calendar fallback and the bias correction arithmetic."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from astock_af.forecast import format_forecast, next_trading_days, validation_bias


def test_next_trading_days_uses_the_exchange_calendar(monkeypatch) -> None:
    """The exchange calendar is preferred when it answers, holidays included."""

    class FakeAkshare(types.ModuleType):
        @staticmethod
        def tool_trade_date_hist_sina() -> pd.DataFrame:
            # 2026-10-01..05 is the National Day holiday: it must be skipped, not guessed.
            return pd.DataFrame(
                {"trade_date": ["2026-09-29", "2026-09-30", "2026-10-06", "2026-10-07", "2026-10-08"]}
            )

    monkeypatch.setitem(sys.modules, "akshare", FakeAkshare("akshare"))

    days, source = next_trading_days("2026-09-28", count=4)
    assert days == ["2026-09-29", "2026-09-30", "2026-10-06", "2026-10-07"]
    assert "calendar" in source


def test_next_trading_days_labels_the_weekday_fallback(monkeypatch) -> None:
    """With no calendar available the dates are labelled as a fallback, never presented as facts."""

    class BrokenAkshare(types.ModuleType):
        @staticmethod
        def tool_trade_date_hist_sina() -> pd.DataFrame:
            raise RuntimeError("endpoint rot")

    monkeypatch.setitem(sys.modules, "akshare", BrokenAkshare("akshare"))

    days, source = next_trading_days("2026-09-25", count=3)  # a Friday
    assert days == ["2026-09-28", "2026-09-29", "2026-09-30"]  # weekend skipped
    assert "fallback" in source


def test_validation_bias_is_prediction_minus_realised(tmp_path) -> None:
    """The correction must come from the validation slice, in percentage points."""
    val = np.full((10, 2, 3), 0.01, dtype=np.float32)
    y_val = np.zeros((10, 2, 3), dtype=np.float32)
    np.savez(tmp_path / "predictions.npz", val=val, y_val=y_val)

    bias = validation_bias(tmp_path)
    assert bias is not None
    assert bias.shape == (2, 3)
    assert np.allclose(bias, 1.0)  # 0.01 in log-return units is 1 pp


def test_validation_bias_is_none_without_validation_predictions(tmp_path) -> None:
    """A run with no stored validation predictions reports no estimate instead of zero."""
    assert validation_bias(tmp_path) is None


def test_format_forecast_shows_raw_bias_and_corrected() -> None:
    """All three views appear, and the corrected table is raw minus bias."""
    forecast = {
        "run_dir": "reports/runs/x",
        "as_of": "2026-09-22",
        "targets": ["2026-09-23", "2026-09-24"],
        "calendar_source": "exchange calendar (akshare/sina)",
        "symbols": ["600519.SH"],
        "names": ["Kweichow Moutai"],
        "raw": np.array([[-1.36, -1.35]]),
        "bias": np.array([[-1.00, -0.90]]),
        "corrected": np.array([[-0.36, -0.45]]),
    }
    text = format_forecast(forecast)
    assert "Raw model output" in text
    assert "Validation-slice bias" in text
    assert "Bias-corrected forecast" in text
    assert "-1.36" in text and "-1.00" in text and "-0.36" in text
    assert "not look-ahead" in text


@pytest.mark.parametrize("count", [1, 5])
def test_next_trading_days_returns_the_requested_count(monkeypatch, count: int) -> None:
    class FakeAkshare(types.ModuleType):
        @staticmethod
        def tool_trade_date_hist_sina() -> pd.DataFrame:
            return pd.DataFrame({"trade_date": pd.bdate_range("2026-09-23", periods=30)})

    monkeypatch.setitem(sys.modules, "akshare", FakeAkshare("akshare"))
    days, _ = next_trading_days("2026-09-22", count=count)
    assert len(days) == count
    assert days[0] == "2026-09-23"
