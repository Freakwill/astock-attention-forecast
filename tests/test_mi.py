"""Mutual-information estimator tests: the null must absorb the bias."""

from __future__ import annotations

import numpy as np
import pytest

from astock_af.mi import estimate, gaussian_mi, mi_ksg, transform_series


def test_mi_detects_a_nonlinear_relation_that_correlation_misses() -> None:
    """y = x^2: zero correlation, real dependence. This is why MI is used at all."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 2.0, 4000)
    y = x**2 + rng.normal(0.0, 0.05, 4000)
    score = estimate(x, y, n_permutations=50, seed=1)
    assert abs(score.pearson) < 0.1, "the correlation should be ~0 by symmetry"
    assert score.excess_bits > 0.3, "MI must still see the dependence"
    assert score.p_value <= 0.02


def test_mi_of_independent_series_is_absorbed_by_the_null() -> None:
    """Independent data: the bias-corrected estimate must sit near zero."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=1500)
    y = rng.normal(size=1500)
    score = estimate(x, y, n_permutations=100, seed=2)
    assert abs(score.excess_bits) < 0.02
    assert score.p_value > 0.05


def test_gaussian_reference_matches_the_analytic_value() -> None:
    """For a bivariate Gaussian, MI = -0.5 ln(1 - rho^2)."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=20000)
    y = 0.6 * x + np.sqrt(1 - 0.36) * rng.normal(size=20000)
    gauss, pearson = gaussian_mi(x, y)
    assert pearson == pytest.approx(0.6, abs=0.02)
    assert gauss == pytest.approx(-0.5 * np.log(1 - 0.36), rel=0.1)


def test_transform_series_variants() -> None:
    values = np.array([-1.0, 0.0, 2.0])
    assert np.allclose(transform_series(values, "level"), values)
    assert np.allclose(transform_series(values, "abs"), [1.0, 0.0, 2.0])
    assert np.allclose(transform_series(values, "sign"), [-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        transform_series(values, "squared")


def test_lag1_scores_handles_the_diagonal() -> None:
    """A self-pair must produce a finite autocorrelation, not NaN.

    `returns[[c, c]]` selects duplicate columns, which used to hand a 2-D frame to
    the estimator: Pearson came out NaN and the MI was computed on a scrambled
    pairing. The diagonal is the most interesting row, so this is worth pinning.
    """
    import pandas as pd

    from astock_af.mi import lag1_scores

    rng = np.random.default_rng(0)
    values = np.zeros(400)
    for index in range(1, 400):
        values[index] = 0.25 * values[index - 1] + rng.normal()
    frame = pd.DataFrame({"AAA.SZ": values}, index=pd.bdate_range("2024-01-01", periods=400))

    scores = lag1_scores(frame, n_permutations=20)
    diagonal = [score for score in scores if score.source == score.target]
    assert len(diagonal) == 1
    assert np.isfinite(diagonal[0].pearson)
    assert diagonal[0].pearson == pytest.approx(np.corrcoef(values[:-1], values[1:])[0, 1], abs=1e-9)


def test_mi_ksg_rejects_short_or_mismatched_input() -> None:
    with pytest.raises(ValueError):
        mi_ksg(np.arange(5.0), np.arange(5.0))
    with pytest.raises(ValueError):
        mi_ksg(np.arange(20.0), np.arange(19.0))
