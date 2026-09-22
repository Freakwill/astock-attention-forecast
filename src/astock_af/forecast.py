"""Forward forecast for the trading days that have not happened yet.

`train` and `benchmark` answer "how good is the model on a holdout". This module
answers the only question that a forecast can ultimately be graded on once its
horizon elapses: what does the model say about the *next* H days, starting from
the last date in the panel?

The raw output of a return model is not directly usable, because the residue of
the level it failed to learn sits on every prediction (see the project README:
a -0.96 pp/day level bias after point-initialisation). Two things are therefore
reported side by side:

* the **raw** forecast, exactly as the network emits it;
* a **bias-corrected** forecast, where the mean error is estimated on the
  *validation* slice - information that genuinely precedes the forecast date -
  and subtracted. Estimating that correction on the test slice would be
  look-ahead, so it is never done here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def next_trading_days(after: str, count: int = 5) -> tuple[list[str], str]:
    """Trading dates strictly after ``after``, plus the source that supplied them.

    Prefers the exchange calendar (akshare's Sina mirror, token-less). If that
    endpoint is unreachable the function falls back to weekdays and *says so* in
    the returned source string rather than silently inventing market holidays.
    """
    try:
        import akshare as ak

        frame = ak.tool_trade_date_hist_sina()
        column = "trade_date" if "trade_date" in frame.columns else frame.columns[0]
        dates = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
        future = [day for day in dates if day > after]
        if len(future) >= count:
            return future[:count], "exchange calendar (akshare/sina)"
    except Exception:  # noqa: BLE001 - any failure degrades to the labelled fallback
        pass

    start = pd.Timestamp(after) + pd.Timedelta(days=1)
    weekdays = [day.strftime("%Y-%m-%d") for day in pd.bdate_range(start=start, periods=count)]
    return weekdays, "weekday fallback (exchange calendar unavailable)"


def validation_bias(run_dir: str | Path) -> np.ndarray | None:
    """Mean (prediction - realised) per (asset, horizon) on the validation slice, in pp.

    Returns ``None`` when the run has no stored validation predictions (the
    fitted VAR baseline keeps coefficients instead of a checkpoint).
    """
    path = Path(run_dir) / "predictions.npz"
    if not path.is_file():
        return None
    arrays = np.load(path, allow_pickle=False)
    if "val" not in arrays or "y_val" not in arrays:
        return None
    return (arrays["val"] - arrays["y_val"]).mean(axis=0) * 100.0


def forward_forecast(
    run_dir: str | Path,
    tensors_path: str | Path = "data/cache/tensors.npz",
    horizon_days: int = 5,
) -> dict[str, Any]:
    """Forecast the next ``horizon_days`` from the last window in the panel.

    Returns a dict with the as-of date, the target trading dates, the per-asset
    ``raw`` matrix, the validation-estimated ``bias`` and the ``corrected``
    matrix (``raw - bias``), all in percent log return.
    """
    from .data.features import PanelTensors
    from .llm.overlay import latest_forecast

    tensors = PanelTensors.load(tensors_path)
    raw_by_symbol = latest_forecast(run_dir, tensors_path)
    names = list(tensors.names)
    symbols = list(tensors.symbols)

    raw = np.array([raw_by_symbol[symbol] for symbol in symbols], dtype=float)
    bias = validation_bias(run_dir)
    as_of = str(tensors.dates[-1])
    targets, calendar_source = next_trading_days(as_of, horizon_days)

    return {
        "run_dir": str(run_dir),
        "as_of": as_of,
        "targets": targets,
        "calendar_source": calendar_source,
        "symbols": symbols,
        "names": names,
        "raw": raw,
        "bias": bias,
        "corrected": None if bias is None else raw - bias,
    }


def format_forecast(forecast: dict[str, Any]) -> str:
    """Render a forecast as markdown: the raw view, the bias estimate, the correction."""
    rows = ["| asset | " + " | ".join(f"t+{i + 1}" for i in range(forecast["raw"].shape[1])) + " |"]
    rows.append("|---" * (forecast["raw"].shape[1] + 1) + "|")

    def table(matrix: np.ndarray) -> list[str]:
        out = []
        for index, (symbol, name) in enumerate(zip(forecast["symbols"], forecast["names"])):
            cells = " | ".join(f"{value:+.2f}" for value in matrix[index])
            out.append(f"| {name} ({symbol}) | {cells} |")
        return out

    lines = [
        f"# Forward forecast - as of {forecast['as_of']}",
        "",
        f"Target trading days: {', '.join(forecast['targets'])}  "
        f"({forecast['calendar_source']})",
        "",
        "Values are forward log returns in %, relative to the as-of close.",
        "",
        "## Raw model output",
        "",
        *rows,
        *table(forecast["raw"]),
    ]

    if forecast["bias"] is None:
        lines += ["", "_No validation predictions stored for this run, so no bias estimate._"]
    else:
        lines += [
            "",
            "## Validation-slice bias (mean prediction - realised, pp)",
            "",
            *rows,
            *table(forecast["bias"]),
            "",
            "## Bias-corrected forecast (raw minus validation bias)",
            "",
            *rows,
            *table(forecast["corrected"]),
            "",
            "The correction is estimated on the validation slice, which precedes the as-of date, so it",
            "is not look-ahead. It is still an estimate: if the current regime differs from validation,",
            "the correction is wrong by that difference.",
        ]
    return "\n".join(lines) + "\n"
