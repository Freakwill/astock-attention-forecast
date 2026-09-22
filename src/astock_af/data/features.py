"""Feature engineering and tensor assembly.

The model consumes a ``(time, assets, features)`` tensor plus a
``(time, assets, horizon)`` target of forward log returns, so every feature is
computed per asset from that asset's own adjusted series and then pivoted onto
a shared trading-date axis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FeatureConfig

PRICE_COLUMNS = ["open", "high", "low", "close", "preclose"]


class TensorError(RuntimeError):
    """Raised when a panel cannot be turned into a model-ready tensor."""


def _asset_features(group: pd.DataFrame) -> pd.DataFrame:
    """Per-asset feature block, computed on the asset's own adjusted series."""
    close = group["close"].astype(float)
    prev = group["preclose"].astype(float).where(group["preclose"].notna(), close.shift(1))
    log_return = np.log(close / prev)
    out = pd.DataFrame(index=group.index)
    out["log_return"] = log_return
    out["intraday_range"] = (group["high"].astype(float) - group["low"].astype(float)) / prev
    out["log_volume"] = np.log1p(group["volume"].astype(float).clip(lower=0))
    out["log_amount"] = np.log1p(group["amount"].astype(float).clip(lower=0))
    out["turn"] = group["turn"].astype(float)
    out["mom_5"] = np.log(close / close.shift(5))
    out["mom_20"] = np.log(close / close.shift(20))
    out["vol_20"] = log_return.rolling(20, min_periods=10).std()
    return out


def feature_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Stack per-asset features into a long frame with the original keys."""
    blocks = []
    for ts_code, group in panel.sort_values(["ts_code", "date"]).groupby("ts_code", sort=False):
        block = _asset_features(group.reset_index(drop=True))
        block["ts_code"] = ts_code
        block["date"] = group["date"].to_numpy()
        block["close"] = group["close"].to_numpy()
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


@dataclass(slots=True)
class PanelTensors:
    """Model-ready arrays plus the split bookkeeping needed to avoid leakage."""

    x: np.ndarray  # (T, N, F) standardized features
    y: np.ndarray  # (T, N, H) forward log returns, raw scale
    dates: list[str]  # (T,)
    symbols: list[str]  # (N,)
    names: list[str]  # (N,)
    features: list[str]  # (F,)
    mean: np.ndarray  # (N, F) train-slice statistics
    std: np.ndarray  # (N, F)
    n_train: int
    n_val: int
    horizon: int
    lookback: int
    meta: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return self.x.shape[2]

    @property
    def n_assets(self) -> int:
        return self.x.shape[1]

    def split_bounds(self, purge: int = 0) -> dict[str, tuple[int, int]]:
        """Chronological ``timestep`` ranges for train/val/test with a purge gap.

        ``purge`` drops the last ``purge`` timesteps of a split so the target of
        a training window cannot overlap the following split.
        """
        a = self.n_train - purge
        b = self.n_val - purge
        return {
            "train": (0, max(a, 1)),
            "val": (max(a + purge, 1), max(b, a + purge + 1)),
            "test": (max(b + purge, 1), len(self.dates)),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=self.x,
            y=self.y,
            mean=self.mean,
            std=self.std,
            dates=np.array(self.dates),
            symbols=np.array(self.symbols),
            names=np.array(self.names),
            features=np.array(self.features),
            scalars=np.array([self.n_train, self.n_val, self.horizon, self.lookback]),
            meta=np.array([json.dumps(self.meta)]),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> PanelTensors:
        data = np.load(path, allow_pickle=False)
        n_train, n_val, horizon, lookback = (int(v) for v in data["scalars"])
        return cls(
            x=data["x"],
            y=data["y"],
            dates=[str(d) for d in data["dates"]],
            symbols=[str(s) for s in data["symbols"]],
            names=[str(s) for s in data["names"]],
            features=[str(f) for f in data["features"]],
            mean=data["mean"],
            std=data["std"],
            n_train=n_train,
            n_val=n_val,
            horizon=horizon,
            lookback=lookback,
            meta=json.loads(str(data["meta"][0])),
        )


def build_tensors(panel: pd.DataFrame, cfg: FeatureConfig, benchmark: pd.DataFrame | None = None) -> PanelTensors:
    """Pivot features to ``(T, N, F)``, add forward-return targets, standardize.

    Standardization statistics are computed on the training slice only, so the
    validation/test sections keep their real scale.
    """
    feats = feature_frame(panel)
    symbols = list(dict.fromkeys(panel.sort_values("date")["ts_code"]))
    names = [panel.loc[panel["ts_code"] == code, "name"].iloc[0] for code in symbols]
    dates = sorted(panel["date"].unique())

    columns: dict[str, pd.DataFrame] = {}
    for feature in cfg.features:
        if feature not in feats.columns:
            raise TensorError(f"feature {feature!r} is not produced by the feature frame")
        columns[feature] = feats.pivot(index="date", columns="ts_code", values=feature).reindex(dates)[symbols]

    close = feats.pivot(index="date", columns="ts_code", values="close").reindex(dates)[symbols]
    if benchmark is not None and cfg.include_benchmark:
        bench = benchmark.set_index("date")["close"].reindex(dates)
        bench_ret = np.log(bench / bench.shift(1))
        columns["bench_return"] = pd.DataFrame(
            np.repeat(bench_ret.to_numpy()[:, None], len(symbols), axis=1), index=dates, columns=symbols
        )
        for feature in ("log_return", "mom_5", "mom_20"):
            if feature in columns:
                columns[f"rel_{feature}"] = columns[feature].sub(columns["bench_return"], axis=0)

    feature_names = list(columns)
    x = np.stack([columns[f].to_numpy(dtype=np.float64) for f in feature_names], axis=-1)  # (T, N, F)

    close_np = close.to_numpy(dtype=np.float64)
    horizon = cfg.horizon
    t, n = close_np.shape
    y = np.full((t, n, horizon), np.nan)
    for h in range(1, horizon + 1):
        y[: t - h, :, h - 1] = np.log(close_np[h:] / close_np[: t - h])

    valid = ~np.isnan(x).any(axis=(1, 2))
    if not valid.all():
        dropped = int((~valid).sum())
        x = x[valid]
        y = y[valid]
        dates = [d for d, keep in zip(dates, valid) if keep]
        print(f"  tensors: dropped {dropped} timestep(s) with incomplete features")

    n_train = int(len(dates) * cfg.train_ratio)
    n_val = n_train + int(len(dates) * cfg.val_ratio)

    train_slice = x[: max(n_train, 1)]
    mean = np.nanmean(train_slice, axis=0)
    std = np.nanstd(train_slice, axis=0)
    std[std == 0] = 1.0
    x_std = (x - mean) / std

    if np.isnan(x_std).any():
        raise TensorError(
            "NaNs remain after standardization; widen the date range or shorten the lookback "
            "(early rows lack the 20-day rolling features)"
        )

    return PanelTensors(
        x=x_std.astype(np.float32),
        y=y.astype(np.float32),
        dates=dates,
        symbols=symbols,
        names=names,
        features=feature_names,
        mean=mean,
        std=std,
        n_train=n_train,
        n_val=n_val,
        horizon=horizon,
        lookback=cfg.lookback,
        meta={
            "train_ratio": cfg.train_ratio,
            "val_ratio": cfg.val_ratio,
            "start": dates[0],
            "end": dates[-1],
            "rows": len(dates),
            "assets": len(symbols),
            "features_raw": cfg.features,
        },
    )


def tensor_summary(t: PanelTensors) -> str:
    """One-screen description used by the CLI and the README."""
    bounds = t.split_bounds(purge=t.horizon)
    lines = [
        f"tensor      x={t.x.shape} y={t.y.shape}  ({t.n_assets} assets x {t.n_features} features)",
        f"dates       {t.dates[0]} .. {t.dates[-1]}  ({len(t.dates)} trading days)",
        f"target      forward log return, horizon H={t.horizon}, lookback L={t.lookback}",
    ]
    for part, (lo, hi) in bounds.items():
        lines.append(f"  {part:5} timesteps {lo:5} .. {hi:5}  ({t.dates[lo]} .. {t.dates[hi - 1]})")
    lines.append(f"features    {', '.join(t.features)}")
    return "\n".join(lines)


def tensors_meta_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".meta.json")


def write_sidecar(meta: dict, path: str | Path) -> Path:
    """Human-readable sidecar so cached tensors are self-describing."""
    target = tensors_meta_path(path)
    target.write_text(json.dumps(asdict(meta) if not isinstance(meta, dict) else meta, indent=2, ensure_ascii=False))
    return target
