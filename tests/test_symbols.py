"""Symbol parsing / normalization tests."""

from __future__ import annotations

import pytest

from astock_af.data.symbols import SymbolError, parse, parse_many


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("600519.SH", "600519.SH"),
        ("600519.SS", None),  # not a supported suffix -> handled below
    ],
)
def test_suffix_styles(raw: str, expected: str | None) -> None:
    if expected is None:
        with pytest.raises(SymbolError):
            parse(raw)
    else:
        assert parse(raw).ts_code == expected


@pytest.mark.parametrize("raw", ["sh.600519", "sh600519", "SH.600519", "600519.SH"])
def test_vendor_styles_normalize(raw: str) -> None:
    symbol = parse(raw)
    assert symbol.ts_code == "600519.SH"
    assert symbol.baostock == "sh.600519"
    assert symbol.sina == "sh600519"


def test_infer_exchange_from_prefix() -> None:
    assert parse("300750").exchange == "SZ"
    assert parse("688981").exchange == "SH"
    assert parse("830799").exchange == "BJ"


@pytest.mark.parametrize("raw", ["000001", "000300", "000905"])
def test_ambiguous_code_requires_suffix(raw: str) -> None:
    """Codes that exist on both exchanges must not be silently guessed.

    ``000001`` is Ping An Bank (SZ) and the SSE Composite (SH); ``000300`` is the
    CSI300 index (SH). Guessing wrong would silently build the dataset from the
    wrong instrument.
    """
    with pytest.raises(SymbolError):
        parse(raw)


def test_unparseable_code_is_rejected() -> None:
    for raw in ("12345", "abcdef", "60051", "6005190"):
        with pytest.raises(SymbolError):
            parse(raw)


def test_index_kind_is_preserved() -> None:
    symbol = parse("000300.SH", name="CSI300", kind="index")
    assert symbol.kind == "index"
    assert symbol.ts_code == "000300.SH"


def test_parse_many_mixes_dicts_and_strings() -> None:
    symbols = parse_many([{"code": "600519.SH", "name": "Moutai"}, "300750.SZ"])
    assert [s.ts_code for s in symbols] == ["600519.SH", "300750.SZ"]
    assert symbols[0].name == "Moutai"
