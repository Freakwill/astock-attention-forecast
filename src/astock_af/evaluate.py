"""Evaluation: metric tables, benchmark comparison and report figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data.features import PanelTensors
from .models import TARGET_SCALE
from .train import regression_metrics

METRIC_COLUMNS = ["mae_pct", "rmse_pct", "direction_accuracy", "bias_pct"]


def metrics_table(predictions: np.ndarray, target: np.ndarray, names: list[str], horizon: int) -> pd.DataFrame:
    """Long-format table: one row per (model, asset, horizon) plus an ``ALL`` row."""
    rows = []
    for index, name in enumerate(names):
        for h in range(horizon):
            metrics = regression_metrics(predictions[:, index, h], target[:, index, h])
            rows.append({"asset": name, "horizon": h + 1, **metrics})
    rows.append({"asset": "ALL", "horizon": 0, **regression_metrics(predictions, target)})
    return pd.DataFrame(rows)


def load_run(run_dir: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read ``metrics.json`` and ``predictions.npz`` for a finished run."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    data = np.load(run_dir / "predictions.npz", allow_pickle=False)
    arrays = {key: data[key] for key in ("test", "y_test", "val", "y_val")}
    arrays["names"] = [str(n) for n in data["names"]]
    arrays["test_dates"] = [str(d) for d in data["test_dates"]]
    return metrics, arrays


def benchmark_runs(run_dirs: list[str | Path]) -> pd.DataFrame:
    """Aggregate finished runs into a model comparison table."""
    rows = []
    for run_dir in run_dirs:
        metrics, arrays = load_run(run_dir)
        overall = regression_metrics(arrays["test"], arrays["y_test"])
        row = {
            "model": metrics["architecture"],
            "run": Path(run_dir).name,
            "epochs": metrics.get("epochs_run"),
            "sec": metrics.get("elapsed_sec"),
            "val_loss": metrics.get("best_val_loss"),
            **overall,
        }
        for h in range(arrays["test"].shape[2]):
            row[f"mae_h{h + 1}"] = regression_metrics(arrays["test"][:, :, h], arrays["y_test"][:, :, h])["mae_pct"]
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("mae_pct").reset_index(drop=True)
    return frame


def format_benchmark(frame: pd.DataFrame) -> str:
    """Markdown table for the README (no external table formatting deps)."""
    columns = [c for c in frame.columns if c not in {"run"}]
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, divider]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_price_panel(
    panel: pd.DataFrame,
    tensors: PanelTensors | None = None,
    run_dir: str | Path | None = None,
    out: str | Path = "reports/figures/price_panel.png",
    asset: int = 0,
) -> Path:
    """Three-panel overview: the dataset, then the model's output drawn on it.

    Top panel: forward-adjusted closes indexed to 100 at the first date, with the
    train / validation / test regions shaded (the unfilled gaps are the purge
    bands that keep a window's 5-day target out of the next split). The middle and
    bottom panels show what the model actually emits on the out-of-sample period -
    per-day and cumulative next-day return. That is how a persistent level bias
    becomes visible, something the MAE table alone hides.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    figure, (axis_price, axis_daily, axis_return) = plt.subplots(
        3, 1, figsize=(13, 11.5), height_ratios=[2.0, 1.0, 1.0], constrained_layout=True
    )

    benchmark_codes = [code for code in panel["ts_code"].unique() if code.endswith("000300.SH")]
    plot_codes = [code for code in panel["ts_code"].unique() if code not in benchmark_codes]

    for code in plot_codes:
        frame = panel[panel["ts_code"] == code].sort_values("date")
        dates = pd.to_datetime(frame["date"])
        indexed = frame["close"].astype(float) / float(frame["close"].iloc[0]) * 100.0
        label = f"{frame['name'].iloc[0]} ({code})"
        axis_price.plot(dates, indexed, linewidth=1.4, label=label)

    for code in benchmark_codes:
        frame = panel[panel["ts_code"] == code].sort_values("date")
        axis_price.plot(
            pd.to_datetime(frame["date"]),
            frame["close"].astype(float) / float(frame["close"].iloc[0]) * 100.0,
            linewidth=1.1,
            linestyle="--",
            color="grey",
            label=f"{frame['name'].iloc[0]} ({code})",
        )

    if tensors is not None:
        bounds = tensors.split_bounds(purge=tensors.horizon)
        shaded = {"train": "#cfe2f3", "val": "#fde8cd", "test": "#d6f0d6"}
        for part, (low, high) in bounds.items():
            if high - low <= 0:
                continue
            span = (pd.to_datetime(tensors.dates[low]), pd.to_datetime(tensors.dates[high - 1]))
            axis_price.axvspan(*span, color=shaded[part], alpha=0.55, zorder=0)
            centre = span[0] + (span[1] - span[0]) / 2
            axis_price.text(
                centre,
                0.97,
                f"{part}\n{high - low} days",
                transform=axis_price.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#333333",
            )
        for part in ("train", "val"):
            boundary = pd.to_datetime(tensors.dates[bounds[part][1]])
            axis_price.annotate(
                f"purge {tensors.horizon}d",
                xy=(boundary, 0.06),
                xycoords=("data", "axes fraction"),
                fontsize=7,
                color="#8a6d3b",
                rotation=90,
                ha="right",
                va="bottom",
            )

    axis_price.set_ylabel("adjusted close, indexed (=100 at first date)")
    axis_price.set_title(
        "A-share panel built by `aaf fetch` -> `aaf tensors`  |  "
        f"{len(plot_codes)} assets, {panel['date'].min()} .. {panel['date'].max()}, "
        f"{len(panel) // max(len(plot_codes), 1)} trading days each"
    )
    axis_price.legend(loc="upper left", fontsize=8, ncols=2, framealpha=0.9)
    axis_price.grid(alpha=0.25)
    axis_price.xaxis.set_major_locator(mdates.YearLocator())
    axis_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    if run_dir is not None:
        metrics, arrays = load_run(run_dir)
        test, y_test = arrays["test"], arrays["y_test"]
        asset = min(asset, test.shape[1] - 1)
        name = arrays["names"][asset]
        dates = pd.to_datetime(arrays["test_dates"])
        actual_daily = y_test[:, asset, 0] * TARGET_SCALE
        predicted_daily = test[:, asset, 0] * TARGET_SCALE
        mean_prediction = float(predicted_daily.mean())
        bias = float((predicted_daily - actual_daily).mean())

        axis_daily.plot(dates, actual_daily, color="#222222", linewidth=1.1, label="actual next-day return")
        axis_daily.plot(dates, predicted_daily, color="#c0392b", linewidth=1.1, label="model forecast")
        axis_daily.axhline(0, color="grey", linewidth=0.7)
        axis_daily.axhline(
            mean_prediction,
            color="#c0392b",
            linewidth=1.0,
            linestyle=":",
            label=f"mean forecast {mean_prediction:+.2f} pp/day",
        )
        axis_daily.set_ylabel("next-day return (%)")
        axis_daily.set_title(
            f"{metrics['architecture']} out-of-sample output - {name}, next-day log return "
            f"(level bias {bias:+.2f} pp/day)"
        )
        axis_daily.legend(loc="lower left", fontsize=8, ncols=3)
        axis_daily.grid(alpha=0.25)

        axis_return.plot(dates, np.cumsum(actual_daily), color="#222222", linewidth=1.5, label="actual (buy & hold)")
        axis_return.plot(
            dates, np.cumsum(predicted_daily), color="#c0392b", linewidth=1.5, label="model forecast"
        )
        axis_return.axhline(0, color="grey", linewidth=0.7)
        axis_return.set_ylabel("cumulative return (%)")
        axis_return.set_title("cumulative - a persistent level bias compounds into a meaningless curve")
        axis_return.legend(loc="lower left", fontsize=8)
        axis_return.grid(alpha=0.25)
        for axis in (axis_daily, axis_return):
            axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        overall = regression_metrics(test, y_test)
        axis_return.text(
            0.99,
            0.04,
            f"all assets/horizons: MAE {overall['mae_pct']:.3f}%   RMSE {overall['rmse_pct']:.3f}%   "
            f"direction {overall['direction_accuracy']:.3f}   bias {overall['bias_pct']:+.3f} pp\n"
            f"{name}: MAE {regression_metrics(test[:, asset, 0], y_test[:, asset, 0])['mae_pct']:.3f}%   "
            f"cumulative forecast {float(np.cumsum(predicted_daily)[-1]):+.1f}% vs actual "
            f"{float(np.cumsum(actual_daily)[-1]):+.1f}%",
            transform=axis_return.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
        )
    else:
        for axis in (axis_daily, axis_return):
            axis.axis("off")
        axis_daily.text(
            0.5,
            0.5,
            "pass --run reports/runs/transformer_<stamp> to overlay the model's out-of-sample output",
            ha="center",
            va="center",
            fontsize=10,
            color="#666666",
        )

    figure.savefig(out, dpi=150)
    plt.close(figure)
    return out


def plot_run(run_dir: str | Path, out: str | Path | None = None) -> Path:
    """Three-panel figure: 1-step forecast vs actual, error by horizon, attention map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    metrics, arrays = load_run(run_dir)
    test, y_test = arrays["test"], arrays["y_test"]
    names = arrays["names"]
    horizon = test.shape[2]
    out = Path(out or run_dir / "report.png")

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    dates = pd.to_datetime(arrays["test_dates"])

    axes[0].plot(dates, (y_test[:, 0, 0] * TARGET_SCALE), label="actual", linewidth=1.2)
    axes[0].plot(dates, (test[:, 0, 0] * TARGET_SCALE), label="predicted", linewidth=1.0, alpha=0.8)
    axes[0].set_title(f"{names[0]} - next-day log return (%)")
    axes[0].legend(fontsize=8)
    axes[0].axhline(0, color="grey", linewidth=0.6)

    widths = [regression_metrics(test[:, :, h], y_test[:, :, h])["mae_pct"] for h in range(horizon)]
    axes[1].bar(range(1, horizon + 1), widths, color="#3b6ea5")
    axes[1].set_title("MAE by forecast horizon")
    axes[1].set_xlabel("horizon (trading days)")
    axes[1].set_ylabel("MAE (%)")

    attention_path = run_dir / "attention.npz"
    if attention_path.is_file():
        weights = np.load(attention_path)["weights"]
        axes[2].imshow(weights[-1, 0], aspect="auto", cmap="viridis")
        axes[2].set_title(f"attention: {names[0]} (last sample)")
        axes[2].set_xlabel("lookback position")
        axes[2].set_ylabel("horizon")
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "no attention weights", ha="center")

    figure.suptitle(f"{metrics['architecture']} | test MAE {regression_metrics(test, y_test)['mae_pct']:.3f}%")
    figure.tight_layout()
    figure.savefig(out, dpi=150)
    plt.close(figure)
    return out


def attention_summary(run_dir: str | Path, tensors: PanelTensors, top_k: int = 5) -> pd.DataFrame:
    """Average cross-attention over the lookback window, per asset and horizon.

    A concentration on the most recent days is the expected signature (short
    memory); a spread-out profile means the model uses longer context.
    """
    run_dir = Path(run_dir)
    weights = np.load(run_dir / "attention.npz")["weights"]  # (S, N, H, L)
    mean = weights.mean(axis=0)  # (N, H, L)
    positions = np.arange(tensors.lookback)
    lag_days = tensors.lookback - 1 - positions  # 0 == most recent day
    rows = []
    for index, name in enumerate(tensors.names):
        for h in range(tensors.horizon):
            profile = mean[index, h]
            top = np.argsort(profile)[::-1][:top_k]
            rows.append(
                {
                    "asset": name,
                    "horizon": h + 1,
                    "top_lags_days": ", ".join(str(int(lag_days[i])) for i in top),
                    "top_weight": float(profile[top[0]]),
                    "weight_last_5d": float(profile[-5:].sum()),
                    "uniform_weight": 1.0 / tensors.lookback,
                }
            )
    return pd.DataFrame(rows)
