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
