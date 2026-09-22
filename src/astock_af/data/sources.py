"""Pluggable daily-bar data sources for A-shares.

Every vendor is normalized to one schema so the panel builder does not care
where the rows came from::

    date  ts_code  name  open  high  low  close  preclose  volume  amount
    turn  pct_chg  pe_ttm  pb_mrq  tradestatus  source

Unit normalization (all vendors -> CNY and shares):

=========  ==================  ===================
Vendor     raw volume / amount  normalized
=========  ==================  ===================
baostock   股 / 元              as-is
akshare    股 / 元              as-is
tushare    手 / 千元            x100 / x1000
=========  ==================  ===================

``turn`` is normalized to **percent of free float traded** and ``tradestatus``
to ``1`` (trading) / ``0`` (suspended).
"""

from __future__ import annotations

import abc
import os
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .symbols import Symbol

CANONICAL_COLUMNS = [
    "date",
    "ts_code",
    "name",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "turn",
    "pct_chg",
    "pe_ttm",
    "pb_mrq",
    "tradestatus",
    "source",
]

_ADJUST_FLAG = {"hfq": "1", "qfq": "2", "none": "3"}
_TOKEN_FILE = Path("~/.tushare/token").expanduser()


class DataSourceError(RuntimeError):
    """Raised when a vendor is unavailable or returns an error payload."""


def _coerce(df: pd.DataFrame, numeric: Sequence[str]) -> pd.DataFrame:
    """Vendor payloads arrive as strings with '' for suspended days."""
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _finalize(df: pd.DataFrame, symbol: Symbol, source: str, venue_adjust: str | None) -> pd.DataFrame:
    """Sort, de-duplicate and pin the canonical column set."""
    if df.empty:
        raise DataSourceError(f"{source}: empty payload for {symbol.ts_code}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["ts_code"] = symbol.ts_code
    df["name"] = symbol.name or symbol.ts_code
    df["source"] = source
    if venue_adjust is not None:
        df["venue_adjust"] = venue_adjust
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[CANONICAL_COLUMNS]


class DataSource(abc.ABC):
    """Interface every vendor adapter implements."""

    name: str = "base"
    requires_token: bool = False

    def available(self) -> tuple[bool, str]:
        """Return ``(usable, reason)`` without touching the network."""
        return True, "ok"

    @abc.abstractmethod
    def fetch_daily(self, symbol: Symbol, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
        """Fetch daily bars in the canonical schema (units: CNY, shares, percent)."""

    def close(self) -> None:
        """Release vendor sessions (no-op by default)."""


class BaostockSource(DataSource):
    """Free, token-less A-share vendor. Includes pre-adjusted prices and PE/PB.

    Baostock needs an anonymous ``login()`` and is comparatively slow
    (~15-20s per symbol for a 7-year daily history), but it is the most stable
    free endpoint reachable from mainland networks.
    """

    name = "baostock"
    _FIELDS = "date,open,high,low,close,preclose,volume,amount,turn,pctChg,tradestatus,peTTM,pbMRQ,isST"

    def __init__(self) -> None:
        self._bs = None

    def _client(self):
        if self._bs is None:
            import baostock as bs

            result = bs.login()
            if result.error_code != "0":
                raise DataSourceError(f"baostock login failed: {result.error_msg}")
            self._bs = bs
        return self._bs

    def close(self) -> None:
        if self._bs is not None:
            self._bs.logout()
            self._bs = None

    def fetch_daily(self, symbol: Symbol, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
        bs = self._client()
        flag = "3" if symbol.kind == "index" else _ADJUST_FLAG.get(adjust, "2")
        rs = bs.query_history_k_data_plus(
            symbol.baostock,
            self._FIELDS,
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag=flag,
        )
        if rs.error_code != "0":
            raise DataSourceError(f"baostock {symbol.baostock}: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        df = _coerce(
            df,
            ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg", "peTTM", "pbMRQ"],
        )
        df = df.rename(columns={"pctChg": "pct_chg", "peTTM": "pe_ttm", "pbMRQ": "pb_mrq"})
        if "tradestatus" in df.columns:
            df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce").fillna(0).astype(int)
        return _finalize(df, symbol, self.name, flag)


class AkshareSource(DataSource):
    """Token-less scraper. Uses the Sina endpoint, which is far faster than Baostock.

    The Eastmoney endpoints (``stock_zh_a_hist`` / ``index_zh_a_hist``) are
    rejected from some networks, so the Sina route is used explicitly.
    """

    name = "akshare"

    def fetch_daily(self, symbol: Symbol, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
        import akshare as ak

        s, e = start.replace("-", ""), end.replace("-", "")
        adj = adjust if adjust != "none" else ""
        if symbol.kind == "index":
            df = ak.stock_zh_index_daily(symbol=symbol.sina)
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start) & (df["date"] <= end)]
        else:
            df = ak.stock_zh_a_daily(symbol=symbol.sina, start_date=s, end_date=e, adjust=adj)
        df = _coerce(df, ["open", "high", "low", "close", "volume", "amount", "outstanding_share", "turnover"])
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["preclose"] = df["close"].shift(1)
        # Sina reports turnover as a ratio; recompute from shares so both
        # vendors land on the same "percent of tradable shares" scale.
        if "outstanding_share" in df.columns:
            df["turn"] = df["volume"] / df["outstanding_share"] * 100.0
        df["tradestatus"] = 1
        df["pct_chg"] = df["close"].pct_change() * 100.0
        return _finalize(df, symbol, self.name, adj or "none")


class TushareSource(DataSource):
    """Optional adapter kept for parity: needs ``TUSHARE_TOKEN``.

    At the free 120-credit tier only ``daily`` (unadjusted OHLCV) is reachable,
    so this adapter deliberately exposes no PE/PB/turnover and ignores
    ``adjust``. It exists so the pipeline can switch vendor with a flag.
    """

    name = "tushare"
    requires_token = True

    def __init__(self) -> None:
        self._pro = None

    def token(self) -> str | None:
        token = os.environ.get("TUSHARE_TOKEN", "").strip()
        if token:
            return token
        if _TOKEN_FILE.is_file():
            return _TOKEN_FILE.read_text().strip() or None
        return None

    def available(self) -> tuple[bool, str]:
        if self.token():
            return True, "token found"
        return False, "TUSHARE_TOKEN unset and ~/.tushare/token missing"

    def _client(self):
        if self._pro is None:
            import tushare as ts

            self._pro = ts.pro_api(self.token())
        return self._pro

    def fetch_daily(self, symbol: Symbol, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
        pro = self._client()
        df = pro.daily(ts_code=symbol.ts_code, start_date=start.replace("-", ""), end_date=end.replace("-", ""))
        df = _coerce(df, ["open", "high", "low", "close", "pre_close", "vol", "amount", "pct_chg"])
        df = df.rename(columns={"trade_date": "date", "pre_close": "preclose"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["volume"] = df["vol"] * 100.0  # 手 -> 股
        df["amount"] = df["amount"] * 1000.0  # 千元 -> 元
        df["tradestatus"] = 1
        return _finalize(df, symbol, self.name, "none")


_REGISTRY: dict[str, type[DataSource]] = {
    BaostockSource.name: BaostockSource,
    AkshareSource.name: AkshareSource,
    TushareSource.name: TushareSource,
}


def available_sources() -> dict[str, str]:
    """Map every registered source to a short availability reason."""
    report: dict[str, str] = {}
    for name, cls in _REGISTRY.items():
        instance = cls()
        ok, reason = instance.available()
        instance.close()
        report[name] = ("usable: " if ok else "unusable: ") + reason
    return report


def build_source(name: str) -> DataSource:
    """Instantiate a source by name."""
    if name not in _REGISTRY:
        raise DataSourceError(f"unknown source {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def resolve_chain(names: Iterable[str], verbose: bool = True) -> list[DataSource]:
    """Keep only usable sources, preserving the configured priority order."""
    chain: list[DataSource] = []
    for name in names:
        source = build_source(name)
        ok, reason = source.available()
        if verbose:
            print(f"  source {name:9} {'OK ' if ok else 'SKIP'}  {reason}")
        if ok:
            chain.append(source)
        else:
            source.close()
    if not chain:
        raise DataSourceError(f"no usable data source among {list(names)}")
    return chain
