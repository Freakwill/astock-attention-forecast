"""News collection for the LLM overlay - no token required.

Sources (all verified reachable from this machine, all free):

* Eastmoney per-symbol search API - company news (direct HTTP, see below)
* ``stock_notice_report``        - exchange filings (one fetch per calendar day,
  shared across every symbol because the endpoint returns all announcements)
* ``stock_info_global_cls``      - CLS newswire (market-wide)
* ``news_cctv``                  - CCTV news broadcast (policy / macro)

Two implementation notes that cost real debugging time:

1. Company news is fetched over HTTP directly rather than through
   ``akshare.stock_news_em``. On pandas 3.x every string column is
   PyArrow-backed and akshare's ``.str.replace(r"\\u3000", "", regex=True)``
   raises ``ArrowInvalid: invalid escape sequence: \\u`` because RE2 does not
   accept ``\\u`` escapes. Cleaning the strings in plain Python keeps this
   module working across pandas majors. ``akshare`` is kept as a fallback.
2. ``akshare.stock_notice_report(symbol=...)`` expects a *report category*
   ("全部", "重大事项", ...), not a stock code - passing a code raises
   ``KeyError``. Announcements are therefore pulled per day and filtered locally,
   with a per-day cache because the payload covers the whole market.

These endpoints only serve *recent* items, which is why the LLM overlay is a
forward-looking module rather than a historical backtest: there is no free
archive to replay last quarter's news.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from ..data.symbols import Symbol

# akshare drives tqdm internally; without this the progress bars swamp the CLI.
os.environ.setdefault("TQDM_DISABLE", "1")

EASTMONEY_SEARCH_URL = "https://search-api-web.eastmoney.com/search/jsonp"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
)

_NOTICE_CACHE: dict[str, pd.DataFrame] = {}


@dataclass(slots=True)
class NewsItem:
    """One normalized headline."""

    ts_code: str
    title: str
    content: str
    published: str
    source: str

    def as_dict(self) -> dict:
        return {
            "ts_code": self.ts_code,
            "title": self.title,
            "content": self.content,
            "published": self.published,
            "source": self.source,
        }


@dataclass(slots=True)
class NewsBundle:
    """Per-symbol headlines plus market-wide context."""

    per_symbol: dict[str, list[NewsItem]] = field(default_factory=dict)
    market: list[NewsItem] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "per_symbol": {code: [item.as_dict() for item in items] for code, items in self.per_symbol.items()},
            "market": [item.as_dict() for item in self.market],
            "failures": self.failures,
        }


def _clip(text: object, limit: int = 400) -> str:
    """Single-line, length-capped text; also strips the search-highlight tags."""
    cleaned = str(text).replace("<em>", "").replace("</em>", "").replace("\u3000", " ")
    return " ".join(cleaned.split())[:limit]


def eastmoney_news(symbol: Symbol, limit: int = 8, timeout: int = 20) -> list[NewsItem]:
    """Company headlines from Eastmoney's news search (direct HTTP, dtype-safe)."""
    inner_param = {
        "uid": "",
        "keyword": symbol.code,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": max(limit, 10),
                "preTag": "<em>",
                "postTag": "</em>",
            }
        },
    }
    params = {
        "cb": "jQuery1124000000000000000_1700000000000",
        "param": json.dumps(inner_param, ensure_ascii=False),
        "_": str(int(time.time() * 1000)),
    }
    headers = {
        "user-agent": USER_AGENT,
        "referer": f"https://so.eastmoney.com/news/s?keyword={symbol.code}",
        "accept": "*/*",
    }
    response = requests.get(EASTMONEY_SEARCH_URL, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    text = response.text
    start, end = text.find("("), text.rfind(")")
    if start < 0 or end <= start:
        raise ValueError("unexpected JSONP payload from Eastmoney news search")
    articles = json.loads(text[start + 1 : end]).get("result", {}).get("cmsArticleWebOld", []) or []

    items: list[NewsItem] = []
    for article in articles[:limit]:
        items.append(
            NewsItem(
                ts_code=symbol.ts_code,
                title=_clip(article.get("title", ""), 160),
                content=_clip(article.get("content", ""), 400),
                published=str(article.get("date", "")),
                source=f"eastmoney/{article.get('mediaName', 'news')}",
            )
        )
    return items


def akshare_news(symbol: Symbol, limit: int = 8) -> list[NewsItem]:
    """Fallback path via akshare (breaks on pandas 3.x, see module docstring)."""
    import akshare as ak

    frame = ak.stock_news_em(symbol=symbol.code)
    return [
        NewsItem(
            ts_code=symbol.ts_code,
            title=_clip(row.get("新闻标题", ""), 160),
            content=_clip(row.get("新闻内容", ""), 400),
            published=str(row.get("发布时间", "")),
            source=f"eastmoney/{row.get('文章来源', 'news')}",
        )
        for _, row in frame.head(limit).iterrows()
    ]


def symbol_news(symbol: Symbol, limit: int = 8) -> list[NewsItem]:
    """Company news: direct HTTP first, akshare as a fallback."""
    try:
        items = eastmoney_news(symbol, limit)
        if items:
            return items
    except Exception as exc:  # noqa: BLE001 - fall through to akshare
        first_error = exc
    else:
        first_error = ValueError("empty payload")
    try:
        return akshare_news(symbol, limit)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"both news paths failed: {first_error}; akshare: {exc}") from exc


def _notice_frame(day: str) -> pd.DataFrame:
    """Announcements for one calendar day, cached (payload covers all symbols)."""
    if day not in _NOTICE_CACHE:
        import akshare as ak

        _NOTICE_CACHE[day] = ak.stock_notice_report(symbol="全部", date=day)
    return _NOTICE_CACHE[day]


def symbol_notices(symbol: Symbol, lookback_days: int = 6, limit: int = 5) -> list[NewsItem]:
    """Most recent exchange filings for a symbol, scanning back over calendar days."""
    for offset in range(lookback_days):
        day = (date.today() - timedelta(days=offset)).strftime("%Y%m%d")
        try:
            frame = _notice_frame(day)
        except Exception:  # noqa: BLE001 - non-trading days simply have no payload
            continue
        if frame is None or frame.empty:
            continue
        hit = frame[frame["代码"].astype(str).str.zfill(6) == symbol.code]
        if hit.empty:
            continue
        return [
            NewsItem(
                ts_code=symbol.ts_code,
                title=_clip(row.get("公告标题", ""), 160),
                content=_clip(row.get("公告类型", ""), 120),
                published=str(row.get("公告日期", day)),
                source="exchange-filing",
            )
            for _, row in hit.head(limit).iterrows()
        ]
    return []


def market_news(limit: int = 12) -> list[NewsItem]:
    """Market-wide newswire: CLS telegraph plus the CCTV broadcast."""
    import akshare as ak

    items: list[NewsItem] = []
    try:
        frame = ak.stock_info_global_cls()
        for _, row in frame.head(limit).iterrows():
            items.append(
                NewsItem(
                    ts_code="MARKET",
                    title=_clip(row.get("标题") or row.get("内容", ""), 160),
                    content=_clip(row.get("内容", ""), 300),
                    published=f"{row.get('发布日期', '')} {row.get('发布时间', '')}",
                    source="cls-telegraph",
                )
            )
    except Exception as exc:  # noqa: BLE001
        items.append(NewsItem("MARKET", f"cls unavailable: {type(exc).__name__}", "", "", "error"))

    for offset in range(1, 5):
        day = (date.today() - timedelta(days=offset)).strftime("%Y%m%d")
        try:
            frame = ak.news_cctv(date=day)
        except Exception:  # noqa: BLE001
            continue
        if frame is None or frame.empty:
            continue
        for _, row in frame.head(4).iterrows():
            items.append(
                NewsItem(
                    ts_code="MARKET",
                    title=_clip(row.get("title", ""), 160),
                    content=_clip(row.get("content", ""), 300),
                    published=day,
                    source="cctv-news",
                )
            )
        break
    return items


def collect(
    symbols: list[Symbol],
    per_symbol_limit: int = 8,
    market_limit: int = 12,
    include_notices: bool = True,
) -> NewsBundle:
    """Gather company and market news, recording which endpoints failed."""
    bundle = NewsBundle()
    for symbol in symbols:
        collected: list[NewsItem] = []
        fetchers = [("news", lambda s=symbol: symbol_news(s, per_symbol_limit))]
        if include_notices:
            fetchers.append(("notices", lambda s=symbol: symbol_notices(s)))
        for label, fetcher in fetchers:
            try:
                collected.extend(fetcher())
            except Exception as exc:  # noqa: BLE001 - degrade, do not abort
                bundle.failures.append(f"{symbol.ts_code}:{label}:{type(exc).__name__}")
        bundle.per_symbol[symbol.ts_code] = collected
    try:
        bundle.market = market_news(market_limit)
    except Exception as exc:  # noqa: BLE001
        bundle.failures.append(f"market:{type(exc).__name__}")
    return bundle


def bundle_frame(bundle: NewsBundle) -> pd.DataFrame:
    """Flat view for eyeballing / saving the raw context."""
    rows = [item.as_dict() for item in bundle.market]
    for items in bundle.per_symbol.values():
        rows.extend(item.as_dict() for item in items)
    return pd.DataFrame(rows)


def save_bundle(bundle: NewsBundle, path: str | Path) -> Path:
    """Persist the raw news context next to the overlay report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(bundle.as_dict(), indent=2, ensure_ascii=False))
    return target
