"""LLM overlay: news-driven directional view, compared against the model output.

Two honest limits are encoded here:

1. The free news endpoints only serve recent items, so the overlay is a
   *forward-looking* module - it cannot be replayed over last quarter's news.
2. Every call is written to disk with its ``as_of`` date, and
   :func:`score_previous` grades older calls once their horizon has elapsed, so
   the module accumulates a track record instead of asserting one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..config import load_data_config
from ..data import load_panel
from ..data.symbols import Symbol
from .client import ChatClient
from .news import NewsBundle, collect, save_bundle

SYSTEM_PROMPT = (
    "You are a sell-side equity analyst covering Chinese A-shares. You are given recent price action, "
    "company news, exchange filings and market-wide headlines. Judge the directional bias of the next "
    f"trading window. Be calibrated: say 'flat' when the evidence is genuinely mixed. "
    "Write every rationale string in English. Reply with JSON only."
)


@dataclass(slots=True)
class OverlayCall:
    """One LLM judgement for one symbol."""

    ts_code: str
    name: str
    as_of: str
    direction: str = "unknown"
    confidence: float = 0.0
    expected_return_pct: float = 0.0
    horizon_days: int = 5
    drivers: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    raw: str = ""
    error: str = ""


def load_llm_config(path: str | Path) -> dict[str, Any]:
    """Read the ``llm`` section plus the data config path it references."""
    raw = yaml.safe_load(Path(path).expanduser().read_text()) or {}
    cfg = raw.get("llm", {}) or {}
    cfg.setdefault("data_config", raw.get("data_config", "config/data.yaml"))
    cfg.setdefault("cache_dir", raw.get("cache_dir", "data/cache"))
    cfg.setdefault("panel_name", raw.get("panel_name", "panel"))
    return cfg


def price_context(panel: pd.DataFrame, ts_code: str, lookback: int = 10) -> dict[str, Any]:
    """Recent price action for one asset: last rows plus momentum/vol summary."""
    frame = panel[panel["ts_code"] == ts_code].sort_values("date")
    close = frame["close"].astype(float)
    recent = frame.tail(lookback)
    returns = close.pct_change()
    return {
        "last_date": str(frame["date"].iloc[-1]),
        "last_20d_return_pct": float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 21 else None,
        "last_60d_return_pct": float((close.iloc[-1] / close.iloc[-61] - 1) * 100) if len(close) > 61 else None,
        "realized_vol_20d_pct": float(returns.tail(20).std() * 100),
        "recent_sessions": [
            {
                "date": str(row["date"]),
                "close": round(float(row["close"]), 2),
                "pct_chg": round(float(row["pct_chg"]), 2) if pd.notna(row["pct_chg"]) else None,
                "turnover_pct": round(float(row["turn"]), 3) if pd.notna(row["turn"]) else None,
            }
            for _, row in recent.iterrows()
        ],
    }


def build_messages(
    symbol: Symbol,
    context: dict[str, Any],
    company_news: list[dict],
    filings: list[dict],
    market_news: list[dict],
    horizon: int,
) -> list[dict]:
    """Assemble the prompt: price action first, then evidence, then the JSON contract."""
    news_lines = "\n".join(f"- [{item['published']}] {item['title']} :: {item['content'][:180]}" for item in company_news)
    filing_lines = "\n".join(f"- {item['title']} ({item['published']})" for item in filings)
    market_lines = "\n".join(
        f"- [{item['published']}] {item['title']} :: {item['content'][:150]}" for item in market_news[:10]
    )
    payload = (
        f"Symbol: {symbol.ts_code} ({symbol.name})\n"
        f"Data as of: {context['last_date']}\n"
        f"20-day return: {context['last_20d_return_pct']}%\n"
        f"60-day return: {context['last_60d_return_pct']}%\n"
        f"20-day realized volatility: {context['realized_vol_20d_pct']}%\n"
        f"Last sessions: {json.dumps(context['recent_sessions'], ensure_ascii=False)}\n\n"
        f"Company news:\n{news_lines or '- none available'}\n\n"
        f"Exchange filings:\n{filing_lines or '- none available'}\n\n"
        f"Market-wide headlines:\n{market_lines or '- none available'}\n\n"
        f"Task: forecast the {horizon}-trading-day directional bias.\n"
        "Respond with JSON: "
        '{"direction": "up"|"down"|"flat", "confidence": 0.0-1.0, "expected_return_pct": float, '
        '"drivers": ["..."], "risks": ["..."]}'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": payload},
    ]


def _parse_call(symbol: Symbol, as_of: str, horizon: int, reply: str | dict) -> OverlayCall:
    """Normalize the model reply into an :class:`OverlayCall`."""
    call = OverlayCall(ts_code=symbol.ts_code, name=symbol.name, as_of=as_of, horizon_days=horizon)
    if isinstance(reply, str):
        call.raw = reply[:2000]
        from .client import parse_json_reply

        try:
            data = parse_json_reply(reply)
        except Exception as exc:  # noqa: BLE001
            call.error = f"{type(exc).__name__}: {str(exc)[:120]}"
            return call
    else:
        data = reply
        call.raw = json.dumps(reply, ensure_ascii=False)[:2000]

    direction = str(data.get("direction", "unknown")).lower().strip()
    call.direction = direction if direction in {"up", "down", "flat"} else "unknown"
    try:
        call.confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        call.confidence = 0.0
    try:
        call.expected_return_pct = float(data.get("expected_return_pct", 0.0))
    except (TypeError, ValueError):
        call.expected_return_pct = 0.0
    call.drivers = [str(item) for item in (data.get("drivers") or [])][:5]
    call.risks = [str(item) for item in (data.get("risks") or [])][:5]
    return call


def latest_forecast(run_dir: str | Path, tensors_path: str | Path = "data/cache/tensors.npz") -> dict[str, list[float]]:
    """Run a trained model on the most recent window (no target required).

    Works for the torch architectures (checkpoint in ``model.pt``); for the
    ridge VAR baseline the coefficients are refitted on every origin whose
    outcome is already known.
    """
    from ..data.features import PanelTensors
    from ..models import build_forecaster
    from ..models.baselines import RidgeVAR

    run_dir = Path(run_dir)
    models_dir = run_dir
    tensors = PanelTensors.load(tensors_path)
    checkpoint_path = models_dir / "model.pt"

    if checkpoint_path.is_file():
        import torch

        from ..config import ModelConfig

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        architecture = checkpoint["architecture"]
        model_cfg = ModelConfig()
        stored = checkpoint.get("model_config") or {}
        for key, value in {**stored, **(checkpoint.get("hyperparameters") or {})}.items():
            if hasattr(model_cfg, key):
                setattr(model_cfg, key, value)
        model = build_forecaster(architecture, tensors, model_cfg)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        window = torch.from_numpy(tensors.x[-tensors.lookback :][None].astype("float32"))
        with torch.no_grad():
            forecast, _ = model(window)
        prediction = forecast.numpy()[0]
    else:
        var_path = run_dir / "var_model.pkl"
        if var_path.is_file():
            import pickle

            model = pickle.loads(var_path.read_bytes())
        else:
            raise FileNotFoundError(f"no model.pt or var_model.pkl in {run_dir}")
        x, y, origins = _known_targets(tensors)
        model.fit(x, y, feature_names=tensors.features)
        prediction = model.predict(tensors.x[-tensors.lookback :][None])
        prediction = np.asarray(prediction)[0]

    return {
        symbol: [round(float(value) * 100.0, 3) for value in prediction[index]]
        for index, symbol in enumerate(tensors.symbols)
    }


def _known_targets(tensors) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows whose forward returns are fully realized (used by the VAR refit)."""
    from ..models import make_windows

    x, y, origins = make_windows(tensors)
    mask = ~np.isnan(y).any(axis=(1, 2))
    return x[mask], y[mask], origins[mask]


def score_previous(out_path: str | Path, panel: pd.DataFrame) -> list[dict]:
    """Grade earlier calls whose horizon has elapsed, using realized closes."""
    out_path = Path(out_path)
    if not out_path.is_file():
        return []
    history = json.loads(out_path.read_text())
    previous = history.get("history", []) if isinstance(history, dict) else []
    scored: list[dict] = []
    for entry in previous:
        call_date = entry.get("as_of")
        if not call_date:
            continue
        for call in entry.get("calls", []):
            frame = panel[panel["ts_code"] == call["ts_code"]].sort_values("date")
            future = frame[frame["date"] > call_date].head(call.get("horizon_days", 5))
            if len(future) < call.get("horizon_days", 5):
                continue
            base = frame[frame["date"] <= call_date]["close"]
            if base.empty:
                continue
            realized = float(future["close"].iloc[-1] / base.iloc[-1] - 1) * 100
            sign = "up" if realized > 0.5 else "down" if realized < -0.5 else "flat"
            scored.append(
                {
                    "as_of": call_date,
                    "ts_code": call["ts_code"],
                    "called": call["direction"],
                    "confidence": call.get("confidence"),
                    "realized_return_pct": round(realized, 3),
                    "realized_direction": sign,
                    "correct": call["direction"] == sign,
                }
            )
    return scored


def run_overlay(
    config_path: str | Path = "config/llm.yaml",
    run_dir: str | Path | None = None,
    days: int = 7,
    symbols: str | None = None,
    out: str | Path = "reports/llm_overlay.json",
) -> dict:
    """Collect news, ask the LLM per symbol, compare with the model, persist results."""
    cfg = load_llm_config(config_path)
    data_cfg = load_data_config(cfg["data_config"])
    universe = data_cfg.symbols
    if symbols:
        wanted = {item.strip() for item in symbols.split(",")}
        universe = [s for s in universe if s.ts_code in wanted or s.code in wanted]
    horizon = int(cfg.get("horizon_days", 5))

    panel, meta = load_panel(cfg["cache_dir"], cfg["panel_name"])
    as_of = str(panel["date"].max())
    print(f"[llm] as_of={as_of} universe={[s.ts_code for s in universe]} horizon={horizon}")

    bundle: NewsBundle = collect(universe, per_symbol_limit=int(cfg.get("max_news_per_symbol", 8)))
    if bundle.failures:
        print(f"[llm] news endpoint failures: {bundle.failures}")
    save_bundle(bundle, Path(out).with_suffix(".news.json"))

    client = ChatClient.from_config(cfg)
    calls: list[OverlayCall] = []
    for symbol in universe:
        items = bundle.per_symbol.get(symbol.ts_code, [])
        company_news = [item.as_dict() for item in items if item.source.startswith("eastmoney")]
        filings = [item.as_dict() for item in items if item.source == "exchange-filing"]
        context = price_context(panel, symbol.ts_code)
        messages = build_messages(
            symbol,
            context,
            company_news,
            filings,
            [item.as_dict() for item in bundle.market],
            horizon,
        )
        try:
            reply = client.chat(messages, json_mode=True)
            call = _parse_call(symbol, as_of, horizon, reply)
        except Exception as exc:  # noqa: BLE001 - one failed symbol must not kill the run
            call = _parse_call(symbol, as_of, horizon, "")
            call.error = f"{type(exc).__name__}: {str(exc)[:160]}"
        print(
            f"  {call.ts_code:11} {call.direction:7} conf={call.confidence:.2f} "
            f"expected={call.expected_return_pct:+.2f}% drivers={len(call.drivers)}{' ERROR=' + call.error if call.error else ''}"
        )
        calls.append(call)

    model_view: dict[str, dict] = {}
    agreement = {"compared": 0, "agree": 0, "assets": []}
    if run_dir:
        try:
            forecasts = latest_forecast(run_dir)
            model_view = {code: {"per_horizon_pct": values, "h1_sign": np.sign(values[0])} for code, values in forecasts.items()}
            for call in calls:
                view = model_view.get(call.ts_code)
                if not view:
                    continue
                model_sign = "up" if view["h1_sign"] > 0 else "down" if view["h1_sign"] < 0 else "flat"
                agree = model_sign == call.direction
                agreement["compared"] += 1
                agreement["agree"] += int(agree)
                agreement["assets"].append(
                    {
                        "ts_code": call.ts_code,
                        "llm": call.direction,
                        "model": model_sign,
                        "model_h1_pct": view["per_horizon_pct"][0],
                        "agree": agree,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[llm] model comparison skipped: {type(exc).__name__}: {exc}")

    scored = score_previous(out, panel)
    payload = {
        "as_of": as_of,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": client.model,
        "horizon_days": horizon,
        "panel": {"rows": int(meta.rows), "first": meta.first_date, "last": meta.last_date, "source": meta.per_source},
        "calls": [asdict(call) for call in calls],
        "model_view": model_view,
        "agreement": agreement,
        "news_failures": bundle.failures,
    }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(out_path.read_text()).get("history", []) if out_path.is_file() else []
    history.append(payload)
    out_path.write_text(json.dumps({"history": history}, indent=2, ensure_ascii=False))
    write_overlay_markdown(payload, scored, out_path.with_suffix(".md"))
    if scored:
        hits = sum(1 for row in scored if row["correct"])
        print(f"[llm] scored {len(scored)} earlier call(s): direction accuracy {hits / len(scored):.2f}")
    else:
        print("[llm] no earlier calls with an elapsed horizon yet (track record accumulates from this run)")
    print(f"[llm] -> {out_path} and {out_path.with_suffix('.md')}")
    return payload


def write_overlay_markdown(payload: dict, scored: list[dict], path: Path) -> Path:
    """Human-readable summary of the overlay run."""
    lines = [
        f"# LLM overlay - as of {payload['as_of']}",
        "",
        f"- model: `{payload['model']}`, horizon: {payload['horizon_days']} trading days",
        f"- generated: {payload['generated_at']}",
        f"- news endpoint failures: {payload['news_failures'] or 'none'}",
        "",
        "| asset | LLM view | confidence | expected % | model h1 | agree |",
        "|---|---|---|---|---|---|",
    ]
    model_assets = {row["ts_code"]: row for row in payload["agreement"]["assets"]}
    for call in payload["calls"]:
        compare = model_assets.get(call["ts_code"], {})
        lines.append(
            f"| {call['ts_code']} | {call['direction']} | {call['confidence']:.2f} | "
            f"{call['expected_return_pct']:+.2f} | {compare.get('model', '-')} ({compare.get('model_h1_pct', '-')}%) | "
            f"{compare.get('agree', '-')} |"
        )
    lines += ["", "## Rationale", ""]
    for call in payload["calls"]:
        lines.append(f"### {call['ts_code']} - {call['name']}")
        lines.append(f"- drivers: {'; '.join(call['drivers']) or 'n/a'}")
        lines.append(f"- risks: {'; '.join(call['risks']) or 'n/a'}")
        if call.get("error"):
            lines.append(f"- error: {call['error']}")
        lines.append("")
    if scored:
        hits = sum(1 for row in scored if row["correct"])
        lines += [
            "## Track record (calls whose horizon has elapsed)",
            "",
            f"direction accuracy: {hits}/{len(scored)} = {hits / len(scored):.2f}",
            "",
            "| as_of | asset | called | realized % | realized | correct |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {row['as_of']} | {row['ts_code']} | {row['called']} | {row['realized_return_pct']:+.2f} | "
            f"{row['realized_direction']} | {row['correct']} |"
            for row in scored
        ]
    path.write_text("\n".join(lines) + "\n")
    return path
