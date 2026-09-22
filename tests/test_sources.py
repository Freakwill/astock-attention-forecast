"""Data source adapter tests: unit normalization and schema pinning (no network)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from astock_af.data.sources import CANONICAL_COLUMNS, TushareSource, available_sources
from astock_af.data.symbols import parse


def test_tushare_units_are_normalized() -> None:
    """Tushare reports 手 / 千元; the canonical schema is shares / CNY."""

    class FakePro:
        def daily(self, **kwargs):
            return pd.DataFrame(
                {
                    "trade_date": ["20240102", "20240103"],
                    "open": [10.0, 10.2],
                    "high": [10.5, 10.4],
                    "low": [9.9, 10.0],
                    "close": [10.2, 10.1],
                    "pre_close": [10.0, 10.2],
                    "vol": [1000.0, 2000.0],  # 手
                    "amount": [5000.0, 8000.0],  # 千元
                    "pct_chg": [2.0, -0.98],
                }
            )

    source = TushareSource()
    source._pro = FakePro()
    frame = source.fetch_daily(parse("600519"), "2024-01-02", "2024-01-03")

    assert list(frame.columns) == CANONICAL_COLUMNS
    assert frame["volume"].tolist() == [100_000.0, 200_000.0]
    assert frame["amount"].tolist() == [5_000_000.0, 8_000_000.0]
    assert frame["preclose"].tolist() == [10.0, 10.2]
    assert frame["source"].iloc[0] == "tushare"
    assert frame["date"].tolist() == ["2024-01-02", "2024-01-03"]


def test_missing_token_marks_source_unusable(monkeypatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr("astock_af.data.sources._TOKEN_FILE", Path("/nonexistent/tushare-token"))
    usable, reason = TushareSource().available()
    assert usable is False
    assert "TUSHARE_TOKEN" in reason


def test_available_sources_reports_every_registry_entry() -> None:
    report = available_sources()
    assert {"baostock", "akshare", "tushare"} <= set(report)
    assert all(reason.startswith(("usable", "unusable")) for reason in report.values())
