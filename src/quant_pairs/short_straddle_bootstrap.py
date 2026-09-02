"""Block-bootstrap loss distribution for observed short BTC straddles.

This is *not* an adversarial gap test: instead of forcing a tail event onto
every trade, it resamples the joint (BTC return, DVOL change) path from history
using a moving-block bootstrap, preserving volatility clustering and the
return-vol correlation.  Each simulated path has the real DTE of an observed
contract, is repriced daily with inverse Black-76, and bought back on the
synthetic ask.  The three pre-declared exit rules are compared against
hold-to-expiry.  No new rule is introduced after seeing results.

The output is a *loss distribution conditional on a single trade* -- probability
of loss, probability of exceeding multiples of the credit, VaR and Expected
Shortfall.  It is deliberately NOT called probability of ruin: ruin requires
capital, sizing, margin and a liquidation barrier, which this round does not
model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

_OPTION_FEE_FLAT = 0.0003  # OPTION_FEE_BTC_PER_CONTRACT
_OPTION_FEE_CAP = 0.125  # OPTION_FEE_PREMIUM_CAP


def build_joint_history(prices: pd.DataFrame, dvol: pd.DataFrame) -> np.ndarray:
    """Return an (n, 2) array of aligned daily (BTC return, DVOL change).

    DVOL change is expressed in decimal vol points (index points / 100) so it
    can be added directly to a decimal IV.
    """

    price_daily = _daily_close(prices)
    dvol_daily = _daily_close(dvol)
    joint = pd.concat({"btc": price_daily, "dvol": dvol_daily}, axis=1).dropna()
    joint = joint.sort_index()
    joint["ret"] = joint["btc"].pct_change()
    joint["dvol_chg"] = joint["dvol"].diff() / 100
    joint = joint.dropna()
    if len(joint) < 50:
        raise ValueError("insufficient joint history for bootstrap")
    return joint[["ret", "dvol_chg"]].to_numpy(dtype=float)


def sample_block_paths(
    history: np.ndarray,
    *,
    horizon: int,
    n_paths: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Moving-block bootstrap of length-``horizon`` joint paths.

    Blocks of ``block_size`` consecutive rows are drawn with replacement and
    concatenated until each path reaches ``horizon`` days, then truncated.
    Sampling whole blocks (not independent days) preserves clustering and the
    contemporaneous return-vol correlation within each block.
    """

    if horizon <= 0 or n_paths <= 0 or block_size <= 0:
        raise ValueError("horizon, n_paths and block_size must be positive")
    n = len(history)
    if block_size > n:
        raise ValueError("block_size exceeds available history")
    max_start = n - block_size
    blocks_per_path = -(-horizon // block_size)  # ceil
    starts = rng.integers(0, max_start + 1, size=(n_paths, blocks_per_path))
    offsets = np.arange(block_size)
    # index[path, block, day_in_block]
    index = starts[:, :, None] + offsets[None, None, :]
    index = index.reshape(n_paths, blocks_per_path * block_size)[:, :horizon]
    return history[index]  # (n_paths, horizon, 2)


def simulate_trade_losses(
    paths: np.ndarray,
    *,
    entry_credit_btc: float,
    entry_fees_btc: float,
    strike: float,
    entry_forward: float,
    entry_iv: float,
    dte_days: float,
    relative_half_spread: float,
    profit_target: float | None,
    stop_multiple: float | None,
    exit_dte: float,
    contracts: float = 1.0,
    annualization_days: int = 365,
) -> np.ndarray:
    """Return an array of net P&L (BTC) for one exit rule over all paths.

    The forward compounds the sampled BTC returns; IV accumulates the sampled
    DVOL changes.  Each day is repriced with inverse Black-76 and the short is
    bought back on the synthetic ask.  The exit rule (profit target, stop
    multiple, DTE cutoff) is applied on the first day any condition fires, using
    the same priority as ``synthetic_option_backfill.evaluate_short_exit``.
    """

    n_paths, horizon, _ = paths.shape
    day_index = np.arange(1, horizon + 1, dtype=float)
    remaining_dte = np.maximum(dte_days - day_index, 0.0)  # (horizon,)
    t_years = remaining_dte / annualization_days

    forward = entry_forward * np.cumprod(1.0 + paths[:, :, 0], axis=1)  # (n_paths, horizon)
    iv = np.maximum(entry_iv + np.cumsum(paths[:, :, 1], axis=1), 0.01)

    call_mid = _inverse_price_grid("call", forward, strike, t_years, iv)
    put_mid = _inverse_price_grid("put", forward, strike, t_years, iv)
    call_ask = call_mid * (1.0 + relative_half_spread)
    put_ask = put_mid * (1.0 + relative_half_spread)
    close_ask = (call_ask + put_ask) * contracts
    close_fees = contracts * (_fee_grid(call_ask) + _fee_grid(put_ask))
    close_cost = close_ask + close_fees  # (n_paths, horizon)

    hit_profit = (
        np.zeros_like(close_cost, dtype=bool)
        if profit_target is None
        else close_cost <= entry_credit_btc * (1.0 - profit_target)
    )
    hit_stop = (
        np.zeros_like(close_cost, dtype=bool)
        if stop_multiple is None
        else close_cost >= entry_credit_btc * stop_multiple
    )
    hit_dte = remaining_dte[None, :] <= exit_dte
    triggered = hit_profit | hit_stop | hit_dte  # (n_paths, horizon)

    # First triggered day per path; if none, use the last available mark.
    any_trigger = triggered.any(axis=1)
    first_idx = np.where(any_trigger, triggered.argmax(axis=1), horizon - 1)
    chosen_cost = close_cost[np.arange(n_paths), first_idx]
    return entry_credit_btc - chosen_cost - entry_fees_btc


def loss_statistics(
    pnl: np.ndarray, *, entry_credit_btc: float
) -> dict[str, float]:
    """Summarize a P&L sample as a conditional-loss distribution.

    Losses are reported as positive multiples of the entry credit.  VaR/ES are
    reported at 95% and 99% on the loss side.
    """

    if entry_credit_btc <= 0:
        raise ValueError("entry_credit_btc must be positive")
    loss = -pnl  # positive = loss
    loss_mult = loss / entry_credit_btc
    var95 = float(np.quantile(loss, 0.95))
    var99 = float(np.quantile(loss, 0.99))
    tail95 = loss[loss >= var95]
    tail99 = loss[loss >= var99]
    return {
        "paths": int(pnl.size),
        "mean_pnl_btc": float(pnl.mean()),
        "median_pnl_btc": float(np.median(pnl)),
        "mean_return_on_credit": float(pnl.mean() / entry_credit_btc),
        "median_return_on_credit": float(np.median(pnl) / entry_credit_btc),
        "prob_loss": float((pnl < 0).mean()),
        "prob_loss_gt_1x_credit": float((loss_mult > 1).mean()),
        "prob_loss_gt_2x_credit": float((loss_mult > 2).mean()),
        "prob_loss_gt_5x_credit": float((loss_mult > 5).mean()),
        "var95_btc": var95,
        "var99_btc": var99,
        "var95_credit_multiple": var95 / entry_credit_btc,
        "var99_credit_multiple": var99 / entry_credit_btc,
        "es95_btc": float(tail95.mean()) if tail95.size else var95,
        "es99_btc": float(tail99.mean()) if tail99.size else var99,
        "es95_credit_multiple": float(tail95.mean() / entry_credit_btc),
        "es99_credit_multiple": float(tail99.mean() / entry_credit_btc),
        "worst_loss_btc": float(loss.max()),
        "worst_loss_mult_credit": float(loss_mult.max()),
    }


def _inverse_price_grid(
    option_type: str,
    forward: np.ndarray,
    strike: float,
    t_years: np.ndarray,
    volatility: np.ndarray,
) -> np.ndarray:
    """Vectorized inverse Black-76 over an (n_paths, horizon) grid, in BTC.

    Matches ``inverse_options.inverse_option_price``: at zero time it returns
    intrinsic value under the forward.
    """

    t = np.broadcast_to(t_years, forward.shape)
    sqrt_t = np.sqrt(np.maximum(t, 0.0))
    intrinsic_call = np.maximum(1.0 - strike / forward, 0.0)
    intrinsic_put = np.maximum(strike / forward - 1.0, 0.0)

    live = (t > 0) & (volatility > 0) & (sqrt_t > 0)
    std = np.where(live, volatility * sqrt_t, 1.0)  # avoid /0; masked out below
    d1 = (np.log(forward / strike) + 0.5 * volatility**2 * t) / std
    d2 = d1 - std
    if option_type == "call":
        priced = norm.cdf(d1) - strike / forward * norm.cdf(d2)
        return np.where(live, np.maximum(priced, intrinsic_call), intrinsic_call)
    priced = strike / forward * norm.cdf(-d2) - norm.cdf(-d1)
    return np.where(live, np.maximum(priced, intrinsic_put), intrinsic_put)


def _fee_grid(fill_price_btc: np.ndarray) -> np.ndarray:
    """Vectorized Deribit option fee: min(flat, cap * premium)."""

    return np.minimum(_OPTION_FEE_FLAT, _OPTION_FEE_CAP * fill_price_btc)


def _daily_close(frame: pd.DataFrame) -> pd.Series:
    missing = {"timestamp", "close"}.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")
    time = pd.to_datetime(frame["timestamp"], utc=True, format="mixed").dt.floor("D")
    return pd.to_numeric(frame["close"], errors="raise").groupby(time).last()
