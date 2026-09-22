"""Reference baselines: linear, LSTM and a ridge VAR.

Every baseline shares the ``(S, L, N, F) -> (S, N, H)`` contract so the
benchmark table is apples-to-apples. ``RidgeVAR`` is the classical econometric
benchmark: per-asset returns regressed on the last ``p`` lags of *all* assets.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge


class LinearForecaster(nn.Module):
    """Flatten the whole window and regress directly on the H-step targets."""

    def __init__(self, n_assets: int, n_features: int, lookback: int, horizon: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.horizon = horizon
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(lookback * n_assets * n_features, n_assets * horizon)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch = x.shape[0]
        flat = x.reshape(batch, -1)
        out = self.proj(self.dropout(flat)).reshape(batch, self.n_assets, self.horizon)
        return out, None


class LSTMForecaster(nn.Module):
    """Sequence baseline: LSTM over timesteps, last hidden state drives all horizons."""

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        lookback: int,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=n_assets * n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, n_assets * horizon))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch, steps, assets, features = x.shape
        sequence = x.reshape(batch, steps, assets * features)
        output, _ = self.lstm(sequence)
        out = self.head(output[:, -1]).reshape(batch, self.n_assets, self.horizon)
        return out, None


class RidgeVAR:
    """Vector-autoregression style baseline fitted with ridge regression.

    Uses only the ``log_return`` channel (index configurable), lags over every
    asset: ``y[t, :, :] ~ [r[t-1], ..., r[t-p]]`` flattened across assets.
    """

    def __init__(self, n_assets: int, horizon: int, lag: int = 5, feature_index: int = 0, alpha: float = 1.0) -> None:
        self.n_assets = n_assets
        self.horizon = horizon
        self.lag = lag
        self.feature_index = feature_index
        self.alpha = alpha
        self.model: Ridge | None = None
        self._feature_names: list[str] = []

    def _design(self, x: np.ndarray) -> np.ndarray:
        window = x[:, -self.lag :, :, self.feature_index]  # (S, p, N)
        return window.reshape(window.shape[0], self.lag * self.n_assets)

    def fit(self, x: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> RidgeVAR:
        if feature_names:
            self._feature_names = feature_names
            if "log_return" in feature_names:
                self.feature_index = feature_names.index("log_return")
        design = self._design(x)
        target = y.reshape(y.shape[0], self.n_assets * self.horizon)
        self.model = Ridge(alpha=self.alpha).fit(design, target)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RidgeVAR must be fitted before predict")
        flat = self.model.predict(self._design(x))
        return flat.reshape(x.shape[0], self.n_assets, self.horizon)


def build_baseline(name: str, n_assets: int, n_features: int, lookback: int, horizon: int, **kwargs):
    """Factory for the non-attention baselines."""
    if name == "linear":
        return LinearForecaster(n_assets, n_features, lookback, horizon, **kwargs)
    if name == "lstm":
        return LSTMForecaster(n_assets, n_features, lookback, horizon, **kwargs)
    if name == "var":
        return RidgeVAR(n_assets, horizon, **kwargs)
    raise ValueError(f"unknown baseline {name!r}; expected linear | lstm | var")


def naive_predictions(mode: str, last_return: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """No-fit reference forecasts - the bar every learned model has to clear.

    Args:
        mode: ``zero`` | ``mean`` | ``momentum``.
        last_return: ``(S, N)`` **raw** log returns of the final lookback step.
            Callers must de-standardize first: feeding standardized values here
            silently inflates the momentum MAE by two orders of magnitude.
        y_train: ``(S_train, N, H)`` training targets, used by ``mean``.

    * ``zero``     - predict 0% for every asset/horizon; the unconditional mean of
      daily log returns is ~0, which makes this surprisingly hard to beat on MAE.
    * ``mean``     - predict each asset's average training-period return.
    * ``momentum`` - persist the last observed return across all horizons.
    """
    samples, n_assets = last_return.shape
    horizon = y_train.shape[2]
    if mode == "zero":
        return np.zeros((samples, n_assets, horizon), dtype=np.float32)
    if mode == "mean":
        average = y_train.mean(axis=0)  # (N, H)
        return np.repeat(average[None, :, :], samples, axis=0).astype(np.float32)
    if mode == "momentum":
        return np.repeat(last_return[:, :, None], horizon, axis=2).astype(np.float32)
    raise ValueError(f"unknown naive mode {mode!r}; expected zero | mean | momentum")
