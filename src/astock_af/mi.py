"""Mutual information between consecutive sessions, with a permutation null.

"Does yesterday tell you anything about today?" is a *dependence* question, not a
correlation one: a nonlinear but deterministic map has zero Pearson correlation and
large MI. The problem is that every MI estimator carries a positive finite-sample
bias - of order k/N for the kNN estimator used here - and on a few hundred daily
returns that bias can look exactly like a signal.

So every number is reported three ways:

* ``mi_nats`` / ``mi_bits`` - the raw KSG estimate;
* ``null_mean`` / ``null_p95`` - the same estimator on the series with today's
  values shuffled (marginals preserved, dependence destroyed), which is what the
  bias alone produces;
* ``excess_bits = mi_bits - null_mean_bits`` and a permutation ``p_value``.

An "MI of 0.02 bits" means nothing without those two columns next to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

NATS_TO_BITS = 1.0 / np.log(2.0)


@dataclass(frozen=True, slots=True)
class MIScore:
    """One directed estimate ``I(x_{t-1}; y_t)`` with its null distribution."""

    source: str
    target: str
    transform: str
    n: int
    mi_nats: float
    null_mean_nats: float
    null_p95_nats: float
    p_value: float
    gaussian_nats: float
    pearson: float

    @property
    def mi_bits(self) -> float:
        return self.mi_nats * NATS_TO_BITS

    @property
    def null_mean_bits(self) -> float:
        return self.null_mean_nats * NATS_TO_BITS

    @property
    def excess_bits(self) -> float:
        """Bias-corrected estimate: observed minus what shuffling already produces."""
        return self.mi_bits - self.null_mean_bits

    @property
    def excess_se(self) -> float:
        """Standard error of the excess, using the null spread as the scale."""
        return float(self.null_p95_nats - self.null_mean_nats) * NATS_TO_BITS / 1.645

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "source": self.source,
            "target": self.target,
            "transform": self.transform,
            "n": self.n,
            "pearson": round(self.pearson, 4),
            "mi_bits": round(self.mi_bits, 4),
            "null_mean_bits": round(self.null_mean_bits, 4),
            "null_p95_bits": round(self.null_p95_nats * NATS_TO_BITS, 4),
            "excess_bits": round(self.excess_bits, 4),
            "p_value": round(self.p_value, 4),
        }


def mi_ksg(x: np.ndarray, y: np.ndarray, n_neighbors: int = 3) -> float:
    """Kraskov-style kNN mutual information in nats (sklearn's implementation).

    One-dimensional inputs only: the estimator is used as ``I(x_{t-1}; y_t)``, not
    as a multivariate information measure.
    """
    from sklearn.feature_selection import mutual_info_regression

    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"length mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] < 10:
        raise ValueError("need at least 10 paired observations")
    return float(mutual_info_regression(x, y, n_neighbors=n_neighbors, random_state=0)[0])


def gaussian_mi(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """MI implied by a bivariate Gaussian with the same correlation, plus the r.

    This is the "linear reference": if the KSG estimate matches it, nothing
    nonlinear is being picked up.
    """
    r = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(r) or abs(r) >= 1.0:
        return float("inf"), r
    return float(-0.5 * np.log(1.0 - r * r)), r


def transform_series(values: np.ndarray, transform: Literal["level", "abs", "sign"]) -> np.ndarray:
    """Level (signed returns), absolute (volatility clustering) or sign."""

    values = np.asarray(values, dtype=float)
    if transform == "level":
        return values
    if transform == "abs":
        return np.abs(values)
    if transform == "sign":
        return np.sign(values)
    raise ValueError(f"unknown transform {transform!r}")


def estimate(
    source_values: np.ndarray,
    target_values: np.ndarray,
    source: str = "x",
    target: str = "y",
    transform: Literal["level", "abs", "sign"] = "level",
    n_permutations: int = 200,
    seed: int = 0,
    n_neighbors: int = 3,
) -> MIScore:
    """Estimate ``I(source_{t-1}; target_t)`` with a shuffle null.

    ``source_values`` and ``target_values`` must already be aligned and lagged the
    way the caller wants; this function only pairs ``a[i]`` with ``b[i]``.
    """
    a = transform_series(source_values, transform)
    b = transform_series(target_values, transform)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    observed = mi_ksg(a, b, n_neighbors)
    gauss, pearson = gaussian_mi(a, b)

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for index in range(n_permutations):
        null[index] = mi_ksg(a, rng.permutation(b), n_neighbors)

    # One-sided: we test for *more* dependence than shuffling produces.
    p_value = float((1 + np.sum(null >= observed)) / (1 + n_permutations))
    return MIScore(
        source=source,
        target=target,
        transform=transform,
        n=int(a.shape[0]),
        mi_nats=observed,
        null_mean_nats=float(null.mean()),
        null_p95_nats=float(np.quantile(null, 0.95)),
        p_value=p_value,
        gaussian_nats=gauss,
        pearson=pearson,
    )


def lag1_scores(
    returns: pd.DataFrame,
    transform: Literal["level", "abs", "sign"] = "level",
    n_permutations: int = 200,
    seed: int = 0,
    n_neighbors: int = 3,
) -> list[MIScore]:
    """All directed ``i -> j`` estimates from session ``t-1`` to session ``t``."""
    scores: list[MIScore] = []
    for source in returns.columns:
        for target in returns.columns:
            # Build the pair explicitly: `returns[[c, c]]` yields duplicate column
            # names, so selecting it back returns a 2-D frame instead of a Series,
            # and the diagonal (the autocorrelation) would be computed on a
            # scrambled pairing with a NaN correlation.
            pair = pd.DataFrame(
                {"source": returns[source].astype(float), "target": returns[target].astype(float)}
            ).dropna()
            if len(pair) < 50:
                continue
            scores.append(
                estimate(
                    pair["source"].to_numpy()[:-1],
                    pair["target"].to_numpy()[1:],
                    source=str(source),
                    target=str(target),
                    transform=transform,
                    n_permutations=n_permutations,
                    seed=seed,
                    n_neighbors=n_neighbors,
                )
            )
    return scores


def format_scores(scores: list[MIScore], title: str) -> str:
    """Markdown table of the estimates, sorted by excess information."""
    ordered = sorted(scores, key=lambda score: score.excess_bits, reverse=True)
    lines = [
        f"## {title}",
        "",
        "| source (t-1) | target (t) | n | Pearson | MI bits | shuffle mean | shuffle p95 | excess bits | p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for score in ordered:
        lines.append(
            f"| {score.source} | {score.target} | {score.n} | {score.pearson:+.4f} | {score.mi_bits:.4f} "
            f"| {score.null_mean_bits:.4f} | {score.null_p95_nats * NATS_TO_BITS:.4f} "
            f"| {score.excess_bits:+.4f} | {score.p_value:.3f} |"
        )
    return "\n".join(lines)
