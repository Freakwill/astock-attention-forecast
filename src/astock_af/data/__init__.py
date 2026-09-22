"""Data layer: token-less vendor adapters, panel builder, feature tensors."""

from .panel import (
    PanelMeta,
    align_panel,
    build_panel,
    cache_paths,
    coverage_report,
    load_panel,
)
from .sources import (
    AkshareSource,
    BaostockSource,
    DataSource,
    DataSourceError,
    TushareSource,
    available_sources,
    build_source,
    resolve_chain,
)
from .symbols import Symbol, SymbolError, parse, parse_many

__all__ = [
    "AkshareSource",
    "BaostockSource",
    "DataSource",
    "DataSourceError",
    "PanelMeta",
    "Symbol",
    "SymbolError",
    "TushareSource",
    "align_panel",
    "available_sources",
    "build_panel",
    "build_source",
    "cache_paths",
    "coverage_report",
    "load_panel",
    "parse",
    "parse_many",
    "resolve_chain",
]
