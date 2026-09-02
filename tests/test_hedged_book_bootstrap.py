import numpy as np
import pytest

from quant_pairs.hedged_book_bootstrap import (
    kelly_fraction,
    simulate_capital_sequences,
    simulate_hedged_trade_paths,
)


def _flat_paths(n_paths: int, horizon: int) -> np.ndarray:
    return np.zeros((n_paths, horizon, 2))


def test_flat_path_short_keeps_credit_minus_costs() -> None:
    # No moves, no vol change: the ATM straddle decays to zero intrinsic and
    # the short keeps the credit minus entry fees and small hedge frictions.
    result = simulate_hedged_trade_paths(
        _flat_paths(4, 14),
        strike=100_000.0,
        entry_forward=100_000.0,
        entry_iv=0.6,
        dte_days=14.0,
        entry_credit_btc=0.09,
        entry_fees_btc=0.0006,
        funding_rate_hourly=0.0,
    )
    assert np.allclose(result["total_pnl"], result["total_pnl"][0])
    assert 0.07 < result["total_pnl"][0] < 0.09
    # Final cum column is the settled total by construction.
    assert np.allclose(result["cum_pnl"][:, -1], result["total_pnl"])
    # Margin while open must at least cover the two option add-ons.
    assert (result["margin"][:, :-1] >= 2 * 0.075).all()
    assert (result["margin"][:, -1] == 0).all()


def test_hedged_book_cuts_variance_of_directional_paths() -> None:
    rng = np.random.default_rng(7)
    n, horizon = 500, 14
    paths = np.zeros((n, horizon, 2))
    paths[:, :, 0] = rng.normal(0.0, 0.03, size=(n, horizon))
    credit, fees = 0.09, 0.0006
    hedged = simulate_hedged_trade_paths(
        paths,
        strike=100_000.0,
        entry_forward=100_000.0,
        entry_iv=0.6,
        dte_days=14.0,
        entry_credit_btc=credit,
        entry_fees_btc=fees,
    )["total_pnl"]
    # Unhedged short: credit minus terminal intrinsic of the ATM straddle.
    terminal_forward = 100_000.0 * np.prod(1 + paths[:, :, 0], axis=1)
    intrinsic = np.abs(1.0 - 100_000.0 / terminal_forward)
    unhedged = credit - fees - intrinsic
    assert np.std(hedged) < 0.6 * np.std(unhedged)


def test_kelly_positive_for_positive_edge() -> None:
    rng = np.random.default_rng(11)
    pnl = rng.normal(0.01, 0.03, size=20_000)
    kelly = kelly_fraction(pnl)
    assert kelly["kelly_contracts_per_btc"] > 0
    assert kelly["half_kelly_contracts_per_btc"] == pytest.approx(
        kelly["kelly_contracts_per_btc"] / 2
    )


def test_ruin_probability_increases_with_size_without_barrier() -> None:
    # With no margin barrier (nothing force-closes a losing trade), bigger
    # fractional size must raise both ruin and drawdown probabilities.
    rng = np.random.default_rng(3)
    n_paths, width = 400, 15
    total = np.where(rng.random(n_paths) < 0.85, 0.05, -0.60)
    cum = np.zeros((n_paths, width))
    cum[:, -1] = total
    for column in range(1, width - 1):
        cum[:, column] = total * column / (width - 1)
    margin = np.zeros((n_paths, width))
    credit = np.full(n_paths, 0.09)

    results = {}
    for size in (0.25, 1.5):
        results[size] = simulate_capital_sequences(
            cum,
            margin,
            credit,
            contracts_per_btc=size,
            n_sequences=2000,
            trades_per_sequence=20,
            rng=np.random.default_rng(99),
            ruin_fraction=0.5,
        )
    assert results[1.5]["prob_ruin"] > results[0.25]["prob_ruin"]
    assert results[1.5]["prob_drawdown_gt_50pct"] >= results[0.25]["prob_drawdown_gt_50pct"]


def test_margin_breach_forces_liquidation() -> None:
    # A mark plunging below the requirement mid-trade must force a close at
    # that day's mark minus the penalty, not ride to the final recovery.
    width = 6
    cum = np.zeros((1, width))
    cum[0] = [0.0, -0.4, -0.4, -0.4, -0.4, -0.1]
    margin = np.full((1, width), 0.3)
    margin[:, -1] = 0.0
    credit = np.array([0.09])
    outcome = simulate_capital_sequences(
        cum,
        margin,
        credit,
        contracts_per_btc=2.0,
        n_sequences=64,
        trades_per_sequence=1,
        rng=np.random.default_rng(1),
        ruin_fraction=0.0,
        liquidation_penalty_credit=0.25,
    )
    # equity path: 1 + 2*(-0.4) = 0.2 < requirement 2*0.3 = 0.6 -> breach.
    assert outcome["prob_liquidation"] == 1.0
    expected_terminal = 1.0 + 2.0 * (-0.4 - 0.25 * 0.09)
    assert outcome["median_terminal_btc"] == pytest.approx(expected_terminal)
