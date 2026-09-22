"""Symbol parsing / normalization across data vendors.

The canonical internal form is Tushare style ``600519.SH``; every vendor gets
its own rendering through :class:`Symbol` properties (``sh.600519`` for
Baostock, ``sh600519`` for Sina/Akshare).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_EXCHANGES = ("SH", "SZ", "BJ")
_SH_RE = re.compile(r"^6\d{5}$")
_SZ_RE = re.compile(r"^(0|3)\d{5}$")
_BJ_RE = re.compile(r"^(4|8|9)\d{5}$")
_VENDOR_RE = re.compile(r"^(sh|sz|bj)[.\-]?(\d{6})$")
_BARE_RE = re.compile(r"^(\d{6})(?:[.\-](sh|sz|bj))?$")

# Six-digit codes that exist as a Shanghai *index* while looking like a
# Shenzhen *stock* (000001 = Ping An Bank / SSE Composite). A bare code is
# rejected for these so the dataset is never silently built from the wrong
# instrument; pass an explicit suffix (e.g. 000300.SH + kind="index").
AMBIGUOUS_CODES = frozenset({"000001", "000016", "000300", "000688", "000905", "000852", "000985", "000015"})


class SymbolError(ValueError):
    """Raised when a symbol string cannot be interpreted unambiguously."""


@dataclass(frozen=True, slots=True)
class Symbol:
    """A tradable A-share / index identifier.

    Attributes:
        code: Six-digit numeric code, e.g. ``600519``.
        exchange: ``SH``, ``SZ`` or ``BJ``.
        name: Human readable name (optional, used for plotting only).
        kind: ``stock`` or ``index`` - indexes are not price-adjusted.
    """

    code: str
    exchange: str
    name: str = ""
    kind: str = "stock"

    @property
    def ts_code(self) -> str:
        """Tushare style, e.g. ``600519.SH``."""
        return f"{self.code}.{self.exchange}"

    @property
    def baostock(self) -> str:
        """Baostock style, e.g. ``sh.600519``."""
        return f"{self.exchange.lower()}.{self.code}"

    @property
    def sina(self) -> str:
        """Sina / Akshare style, e.g. ``sh600519``."""
        return f"{self.exchange.lower()}{self.code}"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.ts_code


def _infer_exchange(code: str) -> str:
    if _SH_RE.match(code):
        return "SH"
    if _SZ_RE.match(code):
        return "SZ"
    if _BJ_RE.match(code):
        return "BJ"
    raise SymbolError(f"cannot infer exchange for code {code!r}; pass an explicit suffix like 000300.SH")


def parse(raw: str, name: str = "", kind: str = "stock") -> Symbol:
    """Parse ``600519``, ``600519.SH``, ``sh.600519`` or ``sh600519``.

    Codes in :data:`AMBIGUOUS_CODES` must carry an explicit exchange suffix
    because they name both a Shenzhen stock and a Shanghai index.
    """
    text = str(raw).strip().lower().replace("_", ".")
    vendor = _VENDOR_RE.match(text)
    if vendor:
        exchange, code = vendor.group(1).upper(), vendor.group(2)
    else:
        bare = _BARE_RE.match(text)
        if not bare:
            raise SymbolError(f"cannot parse symbol: {raw!r}")
        code = bare.group(1)
        suffix = bare.group(2)
        if suffix:
            exchange = suffix.upper()
        else:
            if code in AMBIGUOUS_CODES:
                raise SymbolError(
                    f"{code!r} is ambiguous (Shenzhen stock vs Shanghai index); "
                    f"pass an explicit suffix, e.g. {code}.SZ or {code}.SH"
                )
            exchange = _infer_exchange(code)
    if exchange not in _EXCHANGES:
        raise SymbolError(f"unknown exchange {exchange!r} for {raw!r}")
    return Symbol(code=code, exchange=exchange, name=name, kind=kind)


def parse_many(items: Iterable[dict | str]) -> list[Symbol]:
    """Parse a config-style list of ``{code, name, kind}`` dicts or plain strings."""
    out: list[Symbol] = []
    for item in items:
        if isinstance(item, str):
            out.append(parse(item))
        else:
            out.append(
                parse(
                    item["code"],
                    name=item.get("name", ""),
                    kind=item.get("kind", "stock"),
                )
            )
    return out
