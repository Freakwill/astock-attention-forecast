"""Forecasters: self-attention model plus reference baselines."""

from __future__ import annotations

from ..config import ModelConfig
from ..data.features import PanelTensors
from .attention import AttentionForecaster
from .baselines import LSTMForecaster, LinearForecaster, RidgeVAR, build_baseline
from .dataset import (
    TARGET_SCALE,
    WindowedSplits,
    make_loader,
    make_windows,
    resolve_device,
    split_windows,
)

ARCHITECTURES = ("transformer", "lstm", "linear", "var")


def build_forecaster(name: str, tensors: PanelTensors, cfg: ModelConfig):
    """Instantiate an architecture from the registry.

    ``transformer`` is the project's self-attention model; ``lstm`` / ``linear`` /
    ``var`` are baselines evaluated with the same split and metrics.
    """
    kwargs = {
        "n_assets": tensors.n_assets,
        "n_features": tensors.n_features,
        "lookback": tensors.lookback,
        "horizon": tensors.horizon,
    }
    if name == "transformer":
        return AttentionForecaster(
            **kwargs,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )
    if name in {"lstm", "linear"}:
        return build_baseline(name, **kwargs, dropout=cfg.dropout)
    if name == "var":
        return build_baseline(name, **kwargs)
    raise ValueError(f"unknown architecture {name!r}; expected one of {ARCHITECTURES}")


__all__ = [
    "ARCHITECTURES",
    "AttentionForecaster",
    "LSTMForecaster",
    "LinearForecaster",
    "RidgeVAR",
    "TARGET_SCALE",
    "WindowedSplits",
    "build_forecaster",
    "make_loader",
    "make_windows",
    "resolve_device",
    "split_windows",
]
