"""YAML configuration loading with typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .data.symbols import Symbol, parse_many


@dataclass(slots=True)
class DataConfig:
    """Everything the data layer needs to build a panel."""

    sources: list[str] = field(default_factory=lambda: ["baostock", "akshare", "tushare"])
    adjust: str = "qfq"
    start: str = "2019-01-01"
    end: str | None = None
    cache_dir: str = "data/cache"
    align: str = "intersection"
    symbols: list[Symbol] = field(default_factory=list)
    benchmark: Symbol | None = None
    same_sector_symbols: list[Symbol] = field(default_factory=list)

    @property
    def ts_codes(self) -> list[str]:
        return [s.ts_code for s in self.symbols]


@dataclass(slots=True)
class FeatureConfig:
    """Feature engineering / windowing options."""

    lookback: int = 30
    horizon: int = 5
    features: list[str] = field(
        default_factory=lambda: [
            "log_return",
            "intraday_range",
            "log_volume",
            "log_amount",
            "turn",
            "mom_5",
            "mom_20",
            "vol_20",
        ]
    )
    include_benchmark: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15


@dataclass(slots=True)
class ModelConfig:
    """Self-attention forecaster hyper-parameters."""

    architecture: str = "transformer"
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    dim_feedforward: int = 128


@dataclass(slots=True)
class TrainConfig:
    """Optimization loop settings."""

    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 12
    seed: int = 42
    device: str = "auto"  # auto | mps | cpu
    out_dir: str = "reports/runs"


@dataclass(slots=True)
class ExperimentConfig:
    """Root config object."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    text = Path(path).expanduser().read_text()
    return yaml.safe_load(text) or {}


def load_data_config(path: str | Path) -> DataConfig:
    """Build a :class:`DataConfig` from ``config/data.yaml``."""
    raw = _load_yaml(path).get("panel", {})
    return DataConfig(
        sources=list(raw.get("sources", ["baostock", "akshare", "tushare"])),
        adjust=raw.get("adjust", "qfq"),
        start=str(raw.get("start", "2019-01-01")),
        end=raw.get("end"),
        cache_dir=raw.get("cache_dir", "data/cache"),
        align=raw.get("align", "intersection"),
        symbols=parse_many(raw.get("symbols", [])),
        benchmark=parse_many([raw["benchmark"]])[0] if raw.get("benchmark") else None,
        same_sector_symbols=parse_many(raw.get("same_sector_symbols", [])),
    )


def load_feature_config(path: str | Path, section: str = "features") -> FeatureConfig:
    """Build a :class:`FeatureConfig` from a YAML section."""
    raw = _load_yaml(path).get(section, {}) or {}
    known = {f for f in FeatureConfig.__slots__}
    return FeatureConfig(**{k: v for k, v in raw.items() if k in known})


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a single YAML file that holds ``data``/``features``/``model``/``train`` sections."""
    raw = _load_yaml(path)
    cfg = ExperimentConfig()
    if "panel" in raw or "symbols" in raw:
        cfg.data = load_data_config(path)
    if "features" in raw:
        cfg.features = load_feature_config(path)
    if "model" in raw:
        known = set(ModelConfig.__slots__)
        cfg.model = ModelConfig(**{k: v for k, v in raw["model"].items() if k in known})
    if "train" in raw:
        known = set(TrainConfig.__slots__)
        cfg.train = TrainConfig(**{k: v for k, v in raw["train"].items() if k in known})
    return cfg


def dataset_config_path(config_dir: str | Path, name: str = "dataset.yaml") -> Path:
    """Resolve the dataset config inside a config directory."""
    return Path(config_dir).expanduser() / name
