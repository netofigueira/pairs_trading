"""Bootstrap of the delta-hedged short straddle book with margin and sizing.

Phase 3 of the volatility pipeline plan: turn the per-trade loss distribution
into capital-level answers.  Layer 1 reprices resampled (return, DVOL) paths as
a *hedged* book — daily perp delta hedge, taker fees, constant funding drag —
and tracks an approximate Deribit maintenance-margin requirement along each
path.  Layer 2 compounds sequences of such trades under fractional sizing with
a liquidation barrier: a trade whose marked equity falls below the maintenance
requirement is force-closed with a penalty.

Approximations are deliberate and declared: the underlying is proxied by the
forward, funding is a constant mean hourly rate, and margin uses the standard
(non-portfolio) Deribit formulas — conservative for a hedged book.
"""

from __future__ import annotations

import numpy as np

from quant_pairs.short_straddle_bootstrap import _inverse_price_grid

_RELATIVE_BUMP = 1e-4
_SETTLEMENT_FEE_FLAT = 0.00015
_SETTLEMENT_FEE_CAP = 0.125
_OPTION_MM_ADDON = 0.075  # Deribit maintenance margin add-on per short option, BTC
_PERP_MM_RATE = 0.005  # approximate perp maintenance margin on notional


def simulate_hedged_trade_paths(
    paths: np.ndarray,
    *,
    strike: float,
    entry_forward: float,
    entry_iv: float,
    dte_days: float,
    entry_credit_btc: float,
    entry_fees_btc: float,
    contracts: float = 1.0,
    funding_rate_hourly: float = 0.0,
    perp_taker_fee_rate: float = 0.0005,
    annualization_days: int = 365,
) -> dict[str, np.ndarray]:
    """Reprice bootstrap paths as a daily delta-hedged short straddle.

    Returns per-path arrays: ``total_pnl`` (settled, BTC), ``cum_pnl`` of shape
    (n_paths, horizon + 1) marked to mid daily with the final column equal to
    the settled total, and ``margin`` (same shape) with the approximate
    maintenance requirement while the position is open (0 after expiry).
    """

    if paths.ndim != 3 or paths.shape[2] != 2:
        raise ValueError("paths must have shape (n_paths, horizon, 2)")
    if strike <= 0 or entry_forward <= 0 or entry_iv <= 0 or dte_days <= 0:
        raise ValueError("strike, entry_forward, entry_iv and dte_days must be positive")
    if entry_credit_btc <= 0 or entry_fees_btc < 0 or contracts <= 0:
        raise ValueError("credit and contracts must be positive; fees non-negative")

    n_paths, horizon, _ = paths.shape
    day_index = np.arange(0, horizon + 1, dtype=float)
    t_years = np.maximum(dte_days - day_index, 0.0) / annualization_days  # (horizon+1,)

    forward = np.empty((n_paths, horizon + 1))
    forward[:, 0] = entry_forward
    forward[:, 1:] = entry_forward * np.cumprod(1.0 + paths[:, :, 0], axis=1)
    iv = np.empty((n_paths, horizon + 1))
    iv[:, 0] = entry_iv
    iv[:, 1:] = np.maximum(entry_iv + np.cumsum(paths[:, :, 1], axis=1), 0.01)

    call = _inverse_price_grid("call", forward, strike, t_years, iv)
    put = _inverse_price_grid("put", forward, strike, t_years, iv)
    straddle = call + put

    bumped_up = _straddle_grid(forward * (1.0 + _RELATIVE_BUMP), strike, t_years, iv)
    bumped_down = _straddle_grid(forward * (1.0 - _RELATIVE_BUMP), strike, t_years, iv)
    delta = (bumped_up - bumped_down) / (2.0 * _RELATIVE_BUMP * forward)

    # The hedge set at day t is held until day t+1; the underlying is proxied
    # by the forward, so H = c * delta * F^2 neutralizes the book's BTC delta.
    held = contracts * delta[:, :-1] * forward[:, :-1] ** 2  # (n_paths, horizon)
    hedge_steps = held * (1.0 / forward[:, :-1] - 1.0 / forward[:, 1:])
    traded = np.diff(held, axis=1, prepend=0.0)
    fee_steps = perp_taker_fee_rate * np.abs(traded) / forward[:, :-1]
    close_fee = perp_taker_fee_rate * np.abs(held[:, -1]) / forward[:, -1]
    funding_steps = -held / forward[:, :-1] * funding_rate_hourly * 24.0

    settle_fees = contracts * (_settlement_fee_grid(call[:, -1]) + _settlement_fee_grid(put[:, -1]))
    option_total = entry_credit_btc - entry_fees_btc - contracts * straddle[:, -1] - settle_fees
    total_pnl = (
        option_total
        + hedge_steps.sum(axis=1)
        + funding_steps.sum(axis=1)
        - fee_steps.sum(axis=1)
        - close_fee
    )

    option_marked = entry_credit_btc - entry_fees_btc - contracts * straddle
    hedge_cum = np.concatenate(
        [
            np.zeros((n_paths, 1)),
            np.cumsum(hedge_steps + funding_steps - fee_steps, axis=1),
        ],
        axis=1,
    )
    cum_pnl = option_marked + hedge_cum
    cum_pnl[:, -1] = total_pnl

    margin = contracts * (2.0 * _OPTION_MM_ADDON + straddle)
    margin[:, :-1] += _PERP_MM_RATE * np.abs(held) / forward[:, :-1]
    margin[:, -1] = 0.0

    return {"total_pnl": total_pnl, "cum_pnl": cum_pnl, "margin": margin}


def kelly_fraction(
    pnl_per_contract: np.ndarray,
    *,
    max_size: float = 50.0,
    grid_points: int = 2000,
) -> dict[str, float]:
    """Kelly-optimal contracts per 1 BTC of equity via grid search on E[log].

    The candidate grid is truncated where any outcome would wipe the equity
    (1 + s * min(pnl) <= 0).  Returns the optimum and the half-Kelly point.
    """

    pnl = np.asarray(pnl_per_contract, dtype=float)
    if pnl.size == 0:
        raise ValueError("pnl sample is empty")
    worst = pnl.min()
    ceiling = max_size if worst >= 0 else min(max_size, -1.0 / worst * (1 - 1e-9))
    sizes = np.linspace(0.0, ceiling, grid_points)[1:]
    growth = np.array([np.mean(np.log1p(size * pnl)) for size in sizes])
    best = int(np.argmax(growth))
    return {
        "kelly_contracts_per_btc": float(sizes[best]),
        "half_kelly_contracts_per_btc": float(sizes[best] / 2.0),
        "expected_log_growth_per_trade": float(growth[best]),
    }


def simulate_capital_sequences(
    cum_pnl: np.ndarray,
    margin: np.ndarray,
    credit_per_contract: np.ndarray,
    *,
    contracts_per_btc: float,
    n_sequences: int,
    trades_per_sequence: int,
    rng: np.random.Generator,
    initial_capital_btc: float = 1.0,
    ruin_fraction: float = 0.1,
    liquidation_penalty_credit: float = 0.25,
) -> dict[str, float]:
    """Compound sequences of hedged trades under fractional sizing.

    Position size is ``contracts_per_btc * equity`` at each entry.  During a
    trade, marked equity below the maintenance requirement forces a close at
    that day's mark minus ``liquidation_penalty_credit`` times the entry credit
    (a crude, declared slippage for a forced unwind).  Ruin is equity at or
    below ``ruin_fraction`` of the initial capital; ruined sequences stop.
    """

    if cum_pnl.shape != margin.shape or len(credit_per_contract) != len(cum_pnl):
        raise ValueError("cum_pnl, margin and credit_per_contract must align")
    if contracts_per_btc <= 0 or n_sequences <= 0 or trades_per_sequence <= 0:
        raise ValueError("sizing, sequences and trades must be positive")
    if not 0 <= ruin_fraction < 1:
        raise ValueError("ruin_fraction must be in [0, 1)")

    n_paths = len(cum_pnl)
    equity = np.full(n_sequences, float(initial_capital_btc))
    peak = equity.copy()
    max_drawdown = np.zeros(n_sequences)
    alive = np.ones(n_sequences, dtype=bool)
    ever_liquidated = np.zeros(n_sequences, dtype=bool)
    ruin_level = ruin_fraction * initial_capital_btc

    for _ in range(trades_per_sequence):
        if not alive.any():
            break
        idx = rng.integers(0, n_paths, size=n_sequences)
        position = np.where(alive, contracts_per_btc * equity, 0.0)
        trade_cum = cum_pnl[idx]  # (n_sequences, horizon+1)
        trade_margin = margin[idx]
        equity_path = equity[:, None] + position[:, None] * trade_cum
        breach = equity_path < position[:, None] * trade_margin
        breach[:, -1] = False  # position already settled on the last column
        breached = breach.any(axis=1)
        first = np.where(breached, breach.argmax(axis=1), trade_cum.shape[1] - 1)
        pnl_pc = trade_cum[np.arange(n_sequences), first]
        penalty = liquidation_penalty_credit * credit_per_contract[idx]
        pnl_pc = np.where(breached, pnl_pc - penalty, pnl_pc)
        ever_liquidated |= breached & alive

        equity = np.where(alive, np.maximum(equity + position * pnl_pc, 0.0), equity)
        # Intratrade trough counts toward drawdown even without liquidation.
        trough = equity_path.min(axis=1)
        low = np.where(alive, np.minimum(np.maximum(trough, 0.0), equity), equity)
        peak = np.maximum(peak, equity)
        max_drawdown = np.where(alive, np.maximum(max_drawdown, 1.0 - low / peak), max_drawdown)
        alive &= equity > ruin_level

    terminal = equity
    return {
        "contracts_per_btc": float(contracts_per_btc),
        "n_sequences": int(n_sequences),
        "trades_per_sequence": int(trades_per_sequence),
        "prob_ruin": float((~alive).mean()),
        "prob_liquidation": float(ever_liquidated.mean()),
        "prob_drawdown_gt_30pct": float((max_drawdown > 0.30).mean()),
        "prob_drawdown_gt_50pct": float((max_drawdown > 0.50).mean()),
        "median_terminal_btc": float(np.median(terminal)),
        "mean_terminal_btc": float(terminal.mean()),
        "p05_terminal_btc": float(np.quantile(terminal, 0.05)),
        "median_max_drawdown": float(np.median(max_drawdown)),
    }


def _straddle_grid(
    forward: np.ndarray, strike: float, t_years: np.ndarray, iv: np.ndarray
) -> np.ndarray:
    return _inverse_price_grid("call", forward, strike, t_years, iv) + _inverse_price_grid(
        "put", forward, strike, t_years, iv
    )


def _settlement_fee_grid(payoff_btc: np.ndarray) -> np.ndarray:
    fee = np.minimum(_SETTLEMENT_FEE_FLAT, _SETTLEMENT_FEE_CAP * payoff_btc)
    return np.where(payoff_btc > 0, fee, 0.0)
