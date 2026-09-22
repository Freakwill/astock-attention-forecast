"""Windowing and torch plumbing shared by every forecaster.

Shape conventions used across the project::

    X : (S, L, N, F)  S samples, L lookback, N assets, F features
    Y : (S, N, H)     forward log returns, H horizons

Targets stay in log-return units; the loss and every reported metric multiply
by ``TARGET_SCALE`` so numbers read as percent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..data.features import PanelTensors
from ..config import FeatureConfig

TARGET_SCALE = 100.0


def make_windows(t: PanelTensors) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a lookback window over the tensor, anchored on the last observed day.

    A sample with origin timestep ``t`` holds ``x[t - L + 1 : t + 1]`` and the
    target ``y[t]`` (forward returns measured from ``t``).
    """
    n_timesteps = len(t.dates)
    origins = np.arange(t.lookback - 1, n_timesteps - t.horizon)
    if len(origins) == 0:
        raise ValueError(
            f"no samples: {n_timesteps} timesteps with L={t.lookback} and H={t.horizon} leave no room"
        )
    x = np.stack([t.x[o - t.lookback + 1 : o + 1] for o in origins]).astype(np.float32)
    y = np.stack([t.y[o] for o in origins]).astype(np.float32)
    return x, y, origins


@dataclass(slots=True)
class WindowedSplits:
    """Tensor windows plus the origin timesteps each split covers."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_origins: np.ndarray
    test_dates: list[str]

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "train": self.x_train.shape,
            "val": self.x_val.shape,
            "test": self.x_test.shape,
        }


def split_windows(t: PanelTensors, cfg: FeatureConfig | None = None, purge: int | None = None) -> WindowedSplits:
    """Chronologically split windows, purging ``horizon`` steps between splits.

    The purge stops a training window's forward return from reaching into the
    first validation/test day (overlap would leak future information).
    """
    purge = t.horizon if purge is None else purge
    x, y, origins = make_windows(t)
    bounds = t.split_bounds(purge=purge)

    def take(part: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, hi = bounds[part]
        mask = (origins >= lo) & (origins < hi)
        return x[mask], y[mask], origins[mask]

    x_train, y_train, _ = take("train")
    x_val, y_val, _ = take("val")
    x_test, y_test, test_origins = take("test")
    return WindowedSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        test_origins=test_origins,
        test_dates=[t.dates[o] for o in test_origins],
    )


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap arrays in a DataLoader yielding float32 batches."""
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def resolve_device(preference: str = "auto") -> torch.device:
    """``auto`` picks Apple MPS when available, otherwise CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
