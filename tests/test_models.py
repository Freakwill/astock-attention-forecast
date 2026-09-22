"""Model shape / sanity tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from astock_af.models import AttentionForecaster, LSTMForecaster, LinearForecaster
from astock_af.models.baselines import RidgeVAR, naive_predictions

SHAPES = {"n_assets": 4, "n_features": 6, "lookback": 12, "horizon": 5}


def _batch(batch: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch, SHAPES["lookback"], SHAPES["n_assets"], SHAPES["n_features"], generator=generator)


def test_attention_forecast_and_attention_shapes() -> None:
    model = AttentionForecaster(**SHAPES, d_model=32, n_heads=4, n_layers=2)
    model.eval()  # dropout on attention weights is disabled in eval mode
    forecast, attention = model(_batch(), return_attention=True)
    assert forecast.shape == (3, SHAPES["n_assets"], SHAPES["horizon"])
    assert attention.shape == (3, SHAPES["n_assets"], SHAPES["horizon"], SHAPES["lookback"])
    # Cross-attention rows are a distribution over the lookback window.
    assert torch.allclose(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)), atol=1e-4)
    assert (attention >= 0).all()


def test_training_mode_perturbs_attention_weights() -> None:
    """Dropout applies to attention weights while training, so sums drift from 1.

    Saved attention maps must therefore come from an ``eval()`` forward pass -
    otherwise the "which past days mattered" story reads noise.
    """
    model = AttentionForecaster(**SHAPES, d_model=32, dropout=0.5)
    model.train()
    _, attention = model(_batch(), return_attention=True)
    assert not torch.allclose(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)), atol=1e-3)


def test_attention_without_weights_returns_none() -> None:
    model = AttentionForecaster(**SHAPES, d_model=32)
    forecast, attention = model(_batch())
    assert forecast.shape[0] == 3
    assert attention is None


@pytest.mark.parametrize("cls", [LinearForecaster, LSTMForecaster])
def test_baseline_shapes(cls) -> None:
    model = cls(**SHAPES)
    forecast, attention = model(_batch())
    assert forecast.shape == (3, SHAPES["n_assets"], SHAPES["horizon"])
    assert attention is None


@pytest.mark.parametrize("cls", [AttentionForecaster, LinearForecaster, LSTMForecaster])
def test_point_initialisation_starts_at_the_trivial_forecast(cls) -> None:
    """Every learned model must start at the training-period mean, not at random noise.

    Otherwise the forecast level is learned from scratch too, and early stopping keeps
    whichever snapshot has the smallest - still large - level error, which scores worse
    than simply predicting the mean (an observed transformer sat at a -1.8 pp/day bias
    with a test MAE of 3.16% against 2.47% for predicting zero).
    """
    from astock_af.train import _zero_output_layer

    model = cls(**SHAPES)
    model.eval()  # dropout off, so a point-initialised model returns exact zeros
    _zero_output_layer(model, SHAPES["n_assets"] * SHAPES["horizon"])
    forecast, _ = model(_batch())
    assert forecast.abs().max() == 0, f"{cls.__name__} does not start at the trivial forecast"


def test_attention_is_permutation_sensitive() -> None:
    """Shuffling lookback steps must change the output, or the encoder is ignoring order."""
    model = AttentionForecaster(**SHAPES, d_model=32)
    model.eval()
    batch = _batch(1)
    with torch.no_grad():
        straight, _ = model(batch)
        shuffled, _ = model(batch.flip(1))
    assert not torch.allclose(straight, shuffled, atol=1e-5)


def test_ridge_var_shapes() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 12, 4, 6)).astype(np.float32)
    y = rng.normal(size=(50, 4, 5)).astype(np.float32)
    model = RidgeVAR(n_assets=4, horizon=5, lag=3).fit(x, y, feature_names=["log_return"] + [f"f{i}" for i in range(5)])
    prediction = model.predict(x)
    assert prediction.shape == (50, 4, 5)
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize("mode", ["zero", "mean", "momentum"])
def test_naive_predictions_shapes(mode: str) -> None:
    rng = np.random.default_rng(0)
    last_return = rng.normal(scale=0.02, size=(20, 4)).astype(np.float32)
    y_train = rng.normal(scale=0.02, size=(20, 4, 5)).astype(np.float32)
    out = naive_predictions(mode, last_return, y_train)
    assert out.shape == (20, 4, 5)
    if mode == "zero":
        assert np.allclose(out, 0.0)
    if mode == "momentum":
        assert np.allclose(out[0, :, 0], last_return[0])
        assert np.allclose(out[0, :, -1], last_return[0])
        assert np.abs(out).max() < 0.5  # raw log-return scale, not standardized


def test_unknown_naive_mode_raises() -> None:
    with pytest.raises(ValueError):
        naive_predictions("trend", np.zeros((2, 4), dtype=np.float32), np.zeros((2, 4, 5), dtype=np.float32))
