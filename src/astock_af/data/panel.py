"""Panel construction: fetch symbols from a source chain, align, cache."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .sources import DataSource, DataSourceError
from .symbols import Symbol

PANEL_NAME = "panel"


@dataclass(slots=True)
class PanelMeta:
    """Provenance record stored next to every cached panel."""

    name: str
    start: str
    end: str
    adjust: str
    align: str
    symbols: list[str] = field(default_factory=list)
    per_source: dict[str, list[str]] = field(default_factory=dict)
    per_symbol_source: dict[str, str] = field(default_factory=dict)
    rows: int = 0
    first_date: str = ""
    last_date: str = ""
    fetched_at: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def cache_paths(cache_dir: str | Path, name: str = PANEL_NAME) -> tuple[Path, Path]:
    """Return ``(parquet_path, meta_path)`` for a panel name."""
    directory = Path(cache_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.parquet", directory / f"{name}.meta.json"


def fetch_symbols(
    symbols: list[Symbol],
    chain: list[DataSource],
    start: str,
    end: str,
    adjust: str = "qfq",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch every symbol, falling back through ``chain`` on failure."""
    frames: list[pd.DataFrame] = []
    served: dict[str, str] = {}
    for symbol in symbols:
        last_error: Exception | None = None
        for source in chain:
            try:
                df = source.fetch_daily(symbol, start, end, adjust=adjust)
                frames.append(df)
                served[symbol.ts_code] = source.name
                if verbose:
                    print(
                        f"  {symbol.ts_code:11} {symbol.name[:14]:14} rows={len(df):5} "
                        f"{df['date'].iloc[0]} -> {df['date'].iloc[-1]}  [{source.name}]"
                    )
                break
            except Exception as exc:  # noqa: BLE001 - try the next vendor
                last_error = exc
                if verbose:
                    print(f"  {symbol.ts_code:11} {source.name} failed: {type(exc).__name__}: {str(exc)[:80]}")
        else:
            raise DataSourceError(f"all sources failed for {symbol.ts_code}: {last_error}")
    return pd.concat(frames, ignore_index=True), served


def align_panel(panel: pd.DataFrame, how: str = "intersection", verbose: bool = True) -> pd.DataFrame:
    """Keep dates on which every symbol actually traded (intersection) or any (outer).

    A symbol counts as trading on a date only when ``tradestatus == 1``, so
    suspension days (Baostock reports them with empty volume/amount/turn, e.g.
    SMIC 2025-09-01..2025-09-08) are dropped instead of silently filled with
    zeros, which would fabricate zero-volume observations.
    """
    if how not in {"intersection", "outer"}:
        raise ValueError(f"align must be 'intersection' or 'outer', got {how!r}")
    panel = panel.sort_values(["date", "ts_code"]).reset_index(drop=True)
    if how == "outer":
        return panel
    traded = panel[panel["tradestatus"] == 1] if "tradestatus" in panel else panel
    n_symbols = panel["ts_code"].nunique()
    counts = traded.groupby("date")["ts_code"].nunique()
    keep = set(counts[counts == n_symbols].index)
    dropped = sorted(set(panel["date"]) - keep)
    if verbose and dropped:
        print(f"  align: dropped {len(dropped)} date(s) without full participation ({dropped[0]} .. {dropped[-1]})")
    return panel[panel["date"].isin(keep)].reset_index(drop=True)


def build_panel(
    symbols: list[Symbol],
    chain: list[DataSource],
    start: str,
    end: str | None = None,
    adjust: str = "qfq",
    align: str = "intersection",
    name: str = PANEL_NAME,
    cache_dir: str | Path = "data/cache",
    verbose: bool = True,
) -> tuple[pd.DataFrame, PanelMeta]:
    """Fetch, align and persist a multi-asset daily panel."""
    end = end or datetime.now().strftime("%Y-%m-%d")
    raw, served = fetch_symbols(symbols, chain, start, end, adjust=adjust, verbose=verbose)
    panel = align_panel(raw, how=align)
    if panel.empty:
        raise DataSourceError("panel is empty after alignment - check the symbol/date configuration")
    per_source: dict[str, list[str]] = {}
    for ts_code, source_name in served.items():
        per_source.setdefault(source_name, []).append(ts_code)
    meta = PanelMeta(
        name=name,
        start=start,
        end=end,
        adjust=adjust,
        align=align,
        symbols=list(served),
        per_source=per_source,
        per_symbol_source=served,
        rows=len(panel),
        first_date=str(panel["date"].min()),
        last_date=str(panel["date"].max()),
        fetched_at=datetime.now().isoformat(timespec="seconds"),
    )
    parquet, meta_file = cache_paths(cache_dir, name)
    panel.to_parquet(parquet, index=False)
    meta_file.write_text(meta.to_json())
    if verbose:
        print(f"  panel -> {parquet} ({len(panel)} rows, {meta.first_date} .. {meta.last_date})")
    return panel, meta


def load_panel(cache_dir: str | Path = "data/cache", name: str = PANEL_NAME) -> tuple[pd.DataFrame, PanelMeta]:
    """Read a cached panel plus its provenance record."""
    parquet, meta_file = cache_paths(cache_dir, name)
    if not parquet.is_file():
        raise FileNotFoundError(f"no cached panel at {parquet}; run `aaf fetch` first")
    return pd.read_parquet(parquet), PanelMeta(**json.loads(meta_file.read_text()))


def coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol row counts and NaN counts on the core price columns."""
    core = ["open", "high", "low", "close", "volume", "amount", "turn", "pe_ttm", "pb_mrq"]
    rows = []
    for ts_code, group in panel.groupby("ts_code"):
        record = {
            "ts_code": ts_code,
            "name": group["name"].iloc[0],
            "source": group["source"].iloc[0],
            "rows": len(group),
            "first": group["date"].min(),
            "last": group["date"].max(),
        }
        record.update({f"nan_{col}": int(group[col].isna().sum()) for col in core})
        rows.append(record)
    return pd.DataFrame(rows)
