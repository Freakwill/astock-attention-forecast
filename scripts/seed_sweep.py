#!/usr/bin/env python
"""Seed sweep: is the directional edge of the attention model real or noise?

A single chronological split with a single seed cannot distinguish a genuine
signal from lucky initialisation. This script retrains the attention model and
its LSTM baseline across several seeds on both asset groups and reports
mean +/- std, which is the minimum evidence needed before claiming an edge.

Usage:
    python scripts/seed_sweep.py --seeds=5 --epochs=150 --groups=cross_sector,same_sector
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from astock_af.config import load_config
from astock_af.data.features import PanelTensors
from astock_af.train import train_architecture

GROUPS = {
    "cross_sector": "data/cache/tensors.npz",
    "same_sector": "data/cache/tensors_same.npz",
}
ARCHITECTURES = ("transformer", "lstm")
REPORT = Path("reports/seed_sweep.md")


@dataclass(slots=True)
class Cell:
    """Aggregated metrics for one (group, architecture) pair."""

    group: str
    architecture: str
    runs: int
    direction_mean: float
    direction_std: float
    mae_mean: float
    mae_std: float
    rmse_mean: float

    def row(self) -> str:
        return (
            f"| {self.group} | {self.architecture} | {self.runs} | "
            f"{self.direction_mean:.3f} +/- {self.direction_std:.3f} | "
            f"{self.mae_mean:.3f} +/- {self.mae_std:.3f} | {self.rmse_mean:.3f} |"
        )


def sweep_group(group: str, tensors_path: str, seeds: list[int], epochs: int) -> list[Cell]:
    """Train every architecture once per seed on one group."""
    cfg = load_config("config/model.yaml")
    tensors = PanelTensors.load(tensors_path)
    cells: list[Cell] = []
    for architecture in ARCHITECTURES:
        directions, maes, rmses = [], [], []
        for seed in seeds:
            outcome = train_architecture(
                tensors,
                cfg.model,
                cfg.train,
                architecture,
                out_dir="reports/runs/sweep",
                epochs=epochs,
                seed=seed,
                verbose=False,
            )
            metrics = outcome["metrics"]["test_metrics"]
            directions.append(metrics["direction_accuracy"])
            maes.append(metrics["mae_pct"])
            rmses.append(metrics["rmse_pct"])
        cells.append(
            Cell(
                group=group,
                architecture=architecture,
                runs=len(seeds),
                direction_mean=statistics.fmean(directions),
                direction_std=statistics.pstdev(directions),
                mae_mean=statistics.fmean(maes),
                mae_std=statistics.pstdev(maes),
                rmse_mean=statistics.fmean(rmses),
            )
        )
        print(f"  {group:12} {architecture:11} done ({len(seeds)} seeds)")
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds per architecture")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--groups", default="cross_sector,same_sector")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    cells: list[Cell] = []
    for group in [g.strip() for g in args.groups.split(",") if g.strip()]:
        path = GROUPS[group]
        if not Path(path).is_file():
            print(f"  skipping {group}: {path} not found (run `aaf tensors` first)")
            continue
        cells.extend(sweep_group(group, path, seeds, args.epochs))

    lines = [
        "# Seed sweep",
        "",
        f"{args.seeds} seeds per cell, {args.epochs} epoch budget, identical splits within a group.",
        "A directional edge that only shows up for one seed is noise, not skill.",
        "",
        "| group | model | runs | direction acc (mean +/- std) | MAE % (mean +/- std) | RMSE % |",
        "|---|---|---|---|---|---|",
    ]
    lines += [cell.row() for cell in cells]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[4:]))
    print(f"\n[seed_sweep] -> {REPORT}")


if __name__ == "__main__":
    main()
