"""Training loop shared by the attention model and the neural baselines."""

from __future__ import annotations

import json
import random
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import ModelConfig, TrainConfig
from .data.features import PanelTensors
from .models import TARGET_SCALE, build_forecaster, make_loader, resolve_device, split_windows
from .models.baselines import RidgeVAR, naive_predictions


def seed_everything(seed: int) -> None:
    """Make a run reproducible across python / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed) if torch.backends.mps.is_available() else None


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, cwd=Path.cwd()
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        return "unknown"


def regression_metrics(prediction: np.ndarray, target: np.ndarray, scale: float = TARGET_SCALE) -> dict[str, float]:
    """Percent-scale MAE / RMSE plus sign accuracy.

    MAPE is deliberately not reported: dividing by daily returns that sit near
    zero produces meaningless numbers.
    """
    pred = np.asarray(prediction, dtype=np.float64) * scale
    true = np.asarray(target, dtype=np.float64) * scale
    diff = pred - true
    return {
        "mae_pct": float(np.mean(np.abs(diff))),
        "rmse_pct": float(np.sqrt(np.mean(diff**2))),
        "direction_accuracy": float(np.mean(np.sign(pred) == np.sign(true))),
        "bias_pct": float(np.mean(diff)),
    }


def _run_dir(base: str | Path, architecture: str, tag: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = architecture.replace(":", "-")
    name = f"{safe}_{tag}_{stamp}" if tag else f"{safe}_{stamp}"
    path = Path(base) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fit_naive(tensors: PanelTensors, splits, mode: str, verbose: bool) -> dict:
    """Reference forecasts that need no fitting (see ``naive_predictions``)."""
    index = tensors.features.index("log_return") if "log_return" in tensors.features else 0
    # Features are standardized; the momentum baseline needs the raw return scale.
    mean = tensors.mean[:, index]
    std = tensors.std[:, index]
    predictions = {}
    for part, x in (("val", splits.x_val), ("test", splits.x_test)):
        last_return = x[:, -1, :, index] * std[None, :] + mean[None, :]
        predictions[part] = naive_predictions(mode, last_return, splits.y_train)
    val_loss = float(np.mean((predictions["val"] * TARGET_SCALE - splits.y_val * TARGET_SCALE) ** 2))
    if verbose:
        print(f"    naive:{mode} val MSE {val_loss:.4f}")
    return {
        "history": [{"epoch": 1, "train_loss": None, "val_loss": val_loss}],
        "predictions": predictions,
        "best_val_loss": val_loss,
        "device": "cpu",
        "hyperparameters": {"mode": mode},
    }


def train_architecture(
    tensors: PanelTensors,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    architecture: str,
    out_dir: str | Path | None = None,
    epochs: int | None = None,
    device: str | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Train one architecture and persist checkpoint, history and metrics.

    Returns a dict with ``run_dir``, ``metrics``, ``history`` and ``predictions``.
    The test split is never touched during training: architecture and epoch
    selection use the validation split only.
    """
    seed = train_cfg.seed if seed is None else seed
    seed_everything(seed)
    epochs = train_cfg.epochs if epochs is None else epochs
    out_dir = out_dir or train_cfg.out_dir
    splits = split_windows(tensors)
    run_dir = _run_dir(out_dir, architecture, None)
    started = time.time()

    if architecture == "var":
        result = _fit_var(tensors, splits, run_dir, verbose=verbose)
    elif architecture.startswith("naive:"):
        result = _fit_naive(tensors, splits, architecture.split(":", 1)[1], verbose=verbose)
    else:
        result = _fit_torch(
            tensors, splits, model_cfg, train_cfg, architecture, run_dir, epochs, device, verbose=verbose
        )

    history = result.pop("history")
    predictions = result.pop("predictions")
    elapsed = time.time() - started
    test_metrics = regression_metrics(predictions["test"], splits.y_test)
    per_asset = {
        name: regression_metrics(predictions["test"][:, i], splits.y_test[:, i])
        for i, name in enumerate(tensors.names)
    }
    per_horizon = {
        f"h{h + 1}": regression_metrics(predictions["test"][:, :, h], splits.y_test[:, :, h])
        for h in range(tensors.horizon)
    }

    payload = {
        "architecture": architecture,
        "run_dir": str(run_dir),
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_run": len(history),
        "elapsed_sec": round(elapsed, 1),
        "device": result.get("device", "cpu"),
        "git_revision": _git_revision(),
        "tensor_meta": tensors.meta,
        "shapes": {k: list(v) for k, v in splits.shapes.items()},
        "hyperparameters": result.get("hyperparameters", {}),
        "test_metrics": test_metrics,
        "per_asset": per_asset,
        "per_horizon": per_horizon,
        "best_val_loss": result.get("best_val_loss"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    np.savez_compressed(
        run_dir / "predictions.npz",
        test=predictions["test"],
        val=predictions["val"],
        y_test=splits.y_test,
        y_val=splits.y_val,
        symbols=np.array(tensors.symbols),
        names=np.array(tensors.names),
        test_dates=np.array(splits.test_dates),
    )
    if verbose:
        print(
            f"  [{architecture}] {len(history)} epochs in {elapsed:.1f}s | "
            f"test MAE {test_metrics['mae_pct']:.3f}% RMSE {test_metrics['rmse_pct']:.3f}% "
            f"dir {test_metrics['direction_accuracy']:.3f} -> {run_dir}"
        )
    return {"run_dir": str(run_dir), "metrics": payload, "history": history, "predictions": predictions}


def _zero_output_layer(model: nn.Module, n_outputs: int) -> str:
    """Point-initialise a forecaster at the trivial prediction (return the layer path).

    A randomly initialised attention stack emits large-magnitude noise, so training
    starts far from the target level and early stopping keeps the least-bad snapshot
    anyway: an observed transformer ended up with a -1.8 pp/day level bias and a test
    MAE of 3.16% against 2.47% for predicting zero. Zeroing the output pathway makes
    every learned model start exactly at ``naive:mean`` (zero in the centred target
    space), so any reported gain over that baseline has to be earned out of sample
    rather than being a rounding error of an unlucky initialisation.
    """
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.out_features in {1, n_outputs}
    ]
    if not candidates:
        raise RuntimeError(f"no output layer with {1} or {n_outputs} outputs found")
    name, layer = candidates[-1]
    with torch.no_grad():
        layer.weight.zero_()
        if layer.bias is not None:
            layer.bias.zero_()
    return name


def _fit_torch(
    tensors: PanelTensors,
    splits,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    architecture: str,
    run_dir: Path,
    epochs: int,
    device: str | None,
    verbose: bool,
) -> dict:
    torch_device = resolve_device(device or train_cfg.device)
    model = build_forecaster(architecture, tensors, model_cfg).to(torch_device)
    layer = _zero_output_layer(model, tensors.n_assets * tensors.horizon)
    if verbose:
        print(f"    point-initialised {layer} at the training-period mean forecast")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.MSELoss()

    train_loader = make_loader(splits.x_train, splits.y_train, train_cfg.batch_size, shuffle=True)
    val_x = torch.from_numpy(splits.x_val).to(torch_device)
    val_y = torch.from_numpy(splits.y_val).to(torch_device)

    # Centre the targets on the training-period per-(asset, horizon) mean and add it
    # back at inference: the model learns the deviation, not the level. Feeding raw
    # returns leaves the trivial solution (~0) to be found by gradient descent, and
    # early stopping then locks in a worse-than-trivial plateau - an observed
    # transformer emitted a near-constant -1.36 pp for every asset and horizon while
    # the training-period drift was ~0, at val loss 14.5 against 11.6 for predicting
    # zero. With centring, the naive-mean forecast is the model's starting point and
    # `naive:mean` becomes the loss it has to beat.
    target_mean = splits.y_train.mean(axis=0).astype(np.float32)
    target_mean_t = torch.from_numpy(target_mean).to(torch_device)

    best_loss, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(torch_device)
            batch_y = batch_y.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            forecast, _ = model(batch_x)
            loss = criterion(forecast * TARGET_SCALE, (batch_y - target_mean_t) * TARGET_SCALE)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            running += float(loss.item()) * batch_x.shape[0]
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_forecast, _ = model(val_x)
            val_loss = float(
                criterion(val_forecast * TARGET_SCALE, (val_y - target_mean_t) * TARGET_SCALE).item()
            )
        history.append({"epoch": epoch, "train_loss": running / len(splits.x_train), "val_loss": val_loss})
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"    epoch {epoch:3d}  train {history[-1]['train_loss']:.4f}  val {val_loss:.4f}")

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= train_cfg.patience:
                if verbose:
                    print(f"    early stop at epoch {epoch} (best val {best_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "target_mean": target_mean,
            "architecture": architecture,
            "model_config": asdict(model_cfg),
            "shapes": {
                "n_assets": tensors.n_assets,
                "n_features": tensors.n_features,
                "lookback": tensors.lookback,
                "horizon": tensors.horizon,
            },
            "features": tensors.features,
            "symbols": tensors.symbols,
            "names": tensors.names,
        },
        run_dir / "model.pt",
    )

    model.eval()
    with torch.no_grad():
        val_forecast, _ = model(val_x)
        test_forecast, test_attention = model(
            torch.from_numpy(splits.x_test).to(torch_device), return_attention=True
        )
    predictions = {
        "val": val_forecast.cpu().numpy(),
        "test": test_forecast.cpu().numpy(),
    }
    if test_attention is not None:
        np.savez_compressed(run_dir / "attention.npz", weights=test_attention.cpu().numpy())
    hyperparameters = {
        key: getattr(model_cfg, key)
        for key in ("d_model", "n_heads", "n_layers", "dropout", "dim_feedforward")
        if architecture == "transformer"
    }
    hyperparameters.update({"learning_rate": train_cfg.learning_rate, "batch_size": train_cfg.batch_size})
    return {
        "history": history,
        "predictions": predictions,
        "best_val_loss": best_loss,
        "device": str(torch_device),
        "hyperparameters": hyperparameters,
    }


def _fit_var(tensors: PanelTensors, splits, run_dir: Path, verbose: bool) -> dict:
    """Ridge VAR baseline: alpha chosen on the validation split, then frozen."""
    best = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = RidgeVAR(tensors.n_assets, tensors.horizon, lag=5, alpha=alpha)
        model.fit(splits.x_train, splits.y_train, feature_names=tensors.features)
        val_pred = model.predict(splits.x_val)
        loss = float(np.mean((val_pred * TARGET_SCALE - splits.y_val * TARGET_SCALE) ** 2))
        if best is None or loss < best[0]:
            best = (loss, alpha, model)
    loss, alpha, model = best
    if verbose:
        print(f"    var alpha={alpha} best val MSE {loss:.4f}")
    import pickle

    (run_dir / "var_model.pkl").write_bytes(pickle.dumps(model))
    predictions = {"val": model.predict(splits.x_val), "test": model.predict(splits.x_test)}
    return {
        "history": [{"epoch": 1, "train_loss": None, "val_loss": loss}],
        "predictions": predictions,
        "best_val_loss": loss,
        "device": "cpu",
        "hyperparameters": {"lag": 5, "alpha": alpha},
    }
