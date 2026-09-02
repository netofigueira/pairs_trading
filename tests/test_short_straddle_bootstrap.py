import numpy as np
import pandas as pd
import pytest

from quant_pairs.inverse_options import inverse_option_price
from quant_pairs.short_straddle_bootstrap import (
    _inverse_price_grid,
    build_joint_history,
    loss_statistics,
    sample_block_paths,
    simulate_trade_losses,
)


def _series(start: str, n: int, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=n, freq="1D", tz="UTC"),
            "close": values,
        }
    )


def test_build_joint_history_guards_short_series() -> None:
    prices = _series("2021-01-01", 4, [100.0, 110.0, 99.0, 99.0])
    dvol = _series("2021-01-01", 4, [50.0, 55.0, 45.0, 45.0])
    with pytest.raises(ValueError):
        build_joint_history(prices, dvol)


def test_build_joint_history_aligns_return_and_dvol_change() -> None:
    n = 60
    prices = _series("2021-01-01", n, [100.0 * (1.01**i) for i in range(n)])
    dvol = _series("2021-01-01", n, [50.0 + i for i in range(n)])
    hist = build_joint_history(prices, dvol)
    assert hist.shape == (n - 1, 2)
    assert hist[:, 0] == pytest.approx(0.01, abs=1e-9)  # +1% return each day
    assert hist[:, 1] == pytest.approx(0.01, abs=1e-9)  # +1 dvol point = 0.01 decimal


def test_block_bootstrap_shape_and_determinism() -> None:
    hist = np.column_stack([np.linspace(-0.05, 0.05, 200), np.linspace(-0.03, 0.03, 200)])
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    a = sample_block_paths(hist, horizon=9, n_paths=100, block_size=4, rng=rng1)
    b = sample_block_paths(hist, horizon=9, n_paths=100, block_size=4, rng=rng2)
    assert a.shape == (100, 9, 2)
    assert np.array_equal(a, b)  # fixed seed is reproducible


def test_block_bootstrap_preserves_within_block_correlation() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    ret = rng.normal(0, 0.03, n)
    dvol = -0.6 * ret + rng.normal(0, 0.01, n)  # strong negative return-vol link
    hist = np.column_stack([ret, dvol])
    true_corr = np.corrcoef(hist[:, 0], hist[:, 1])[0, 1]
    paths = sample_block_paths(hist, horizon=20, n_paths=500, block_size=5, rng=rng)
    flat = paths.reshape(-1, 2)
    boot_corr = np.corrcoef(flat[:, 0], flat[:, 1])[0, 1]
    # block sampling keeps contemporaneous pairs intact -> correlation preserved
    assert abs(boot_corr - true_corr) < 0.05


def test_loss_statistics_multiples_and_tail() -> None:
    # credit 0.10; losses of 0.05, 0.15, 0.60 (=0.5x, 1.5x, 6x) and one gain
    pnl = np.array([0.02, -0.05, -0.15, -0.60])
    stats = loss_statistics(pnl, entry_credit_btc=0.10)
    assert stats["prob_loss"] == 0.75
    assert stats["prob_loss_gt_1x_credit"] == 0.5
    assert stats["prob_loss_gt_5x_credit"] == 0.25
    assert stats["worst_loss_mult_credit"] == pytest.approx(6.0)
    assert stats["mean_return_on_credit"] == pytest.approx(pnl.mean() / 0.10)
    assert stats["es99_credit_multiple"] == pytest.approx(stats["es99_btc"] / 0.10)


def test_vectorized_inverse_price_matches_scalar_model() -> None:
    forwards = np.array([[80.0, 100.0, 120.0], [50.0, 100.0, 200.0]])
    times = np.array([10 / 365, 5 / 365, 0.0])
    volatilities = np.array([[0.3, 0.6, 0.9], [1.2, 0.5, 0.2]])
    for option_type in ("call", "put"):
        grid = _inverse_price_grid(option_type, forwards, 100.0, times, volatilities)
        expected = np.array(
            [
                [
                    inverse_option_price(
                        option_type,
                        forward=float(forwards[row, column]),
                        strike=100.0,
                        time_years=float(times[column]),
                        volatility=float(volatilities[row, column]),
                    )
                    for column in range(forwards.shape[1])
                ]
                for row in range(forwards.shape[0])
            ]
        )
        assert grid == pytest.approx(expected, abs=1e-12)


def test_hold_rule_disables_profit_and_stop_barriers() -> None:
    paths = np.zeros((2, 3, 2))
    pnl = simulate_trade_losses(
        paths,
        entry_credit_btc=1.0,
        entry_fees_btc=0.0,
        strike=100.0,
        entry_forward=100.0,
        entry_iv=0.60,
        dte_days=3.0,
        relative_half_spread=0.0,
        profit_target=None,
        stop_multiple=None,
        exit_dte=0.0,
    )
    assert pnl == pytest.approx(np.ones(2))


def test_simulate_flat_path_returns_credit_minus_buyback() -> None:
    # A perfectly flat path: forward and IV never move, so the short bleeds only
    # theta.  P&L should be positive and identical across paths.
    paths = np.zeros((5, 8, 2))  # 8 days, no return, no dvol change
    pnl = simulate_trade_losses(
        paths,
        entry_credit_btc=0.05,
        entry_fees_btc=0.0005,
        strike=100.0,
        entry_forward=100.0,
        entry_iv=0.60,
        dte_days=10.0,
        relative_half_spread=0.016,
        profit_target=0.50,
        stop_multiple=2.0,
        exit_dte=3.0,
    )
    assert pnl.shape == (5,)
    assert np.allclose(pnl, pnl[0])  # deterministic given identical flat paths
