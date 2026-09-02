import numpy as np
import pandas as pd
import pytest

from quant_pairs.short_straddle_bootstrap import (
    build_joint_history_with_levels,
    sample_block_paths,
    sample_conditioned_block_paths,
)


def _history_with_two_regimes(n_calm: int = 400, n_wild: int = 400):
    rng = np.random.default_rng(5)
    calm = rng.normal(0.0, 0.01, n_calm)
    wild = rng.normal(0.0, 0.05, n_wild)
    returns = np.concatenate([calm, wild])
    history = np.column_stack([returns, np.zeros_like(returns)])
    levels = np.concatenate([np.full(n_calm, 35.0), np.full(n_wild, 85.0)])
    return history, levels


def test_levels_align_with_history() -> None:
    timestamps = pd.date_range("2026-01-01", periods=100, freq="1D", tz="UTC")
    prices = pd.DataFrame({"timestamp": timestamps, "close": np.linspace(100, 120, 100)})
    dvol = pd.DataFrame({"timestamp": timestamps, "close": np.linspace(40, 60, 100)})
    history, levels = build_joint_history_with_levels(prices, dvol)
    assert len(history) == len(levels)
    # First joint row is the second day (pct_change drops the first).
    assert levels[0] == pytest.approx(40 + 20 / 99)


def test_conditioning_restricts_to_matching_regime() -> None:
    history, levels = _history_with_two_regimes()
    rng = np.random.default_rng(11)
    calm_paths, calm_info = sample_conditioned_block_paths(
        history,
        levels,
        entry_dvol_points=35.0,
        horizon=12,
        n_paths=500,
        block_size=4,
        rng=rng,
        tolerance_points=10.0,
    )
    wild_paths, wild_info = sample_conditioned_block_paths(
        history,
        levels,
        entry_dvol_points=85.0,
        horizon=12,
        n_paths=500,
        block_size=4,
        rng=np.random.default_rng(11),
        tolerance_points=10.0,
    )
    assert calm_info["tolerance_points_used"] == 10.0
    assert wild_info["tolerance_points_used"] == 10.0
    # The conditioned samples must inherit their regime's volatility.
    assert np.std(calm_paths[:, :, 0]) < 0.5 * np.std(wild_paths[:, :, 0])


def test_tolerance_widens_when_regime_is_rare() -> None:
    history, levels = _history_with_two_regimes(n_calm=780, n_wild=20)
    _, info = sample_conditioned_block_paths(
        history,
        levels,
        entry_dvol_points=85.0,
        horizon=8,
        n_paths=50,
        block_size=4,
        rng=np.random.default_rng(2),
        tolerance_points=10.0,
        widening_step_points=5.0,
        min_starts=100,
    )
    assert info["tolerance_points_used"] > 10.0
    assert info["eligible_starts"] >= 100


def test_unconditional_equivalence_when_tolerance_is_huge() -> None:
    history, levels = _history_with_two_regimes()
    conditioned, info = sample_conditioned_block_paths(
        history,
        levels,
        entry_dvol_points=60.0,
        horizon=10,
        n_paths=300,
        block_size=4,
        rng=np.random.default_rng(3),
        tolerance_points=1000.0,
    )
    unconditional = sample_block_paths(
        history, horizon=10, n_paths=300, block_size=4, rng=np.random.default_rng(3)
    )
    # Same shape and similar dispersion; start sets differ by one index only.
    assert conditioned.shape == unconditional.shape
    assert np.std(conditioned[:, :, 0]) == pytest.approx(np.std(unconditional[:, :, 0]), rel=0.15)
