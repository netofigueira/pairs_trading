"""Daily synthetic marking for inverse BTC straddles with explicit provenance."""

from __future__ import annotations

import math

import pandas as pd

from quant_pairs.inverse_options import inverse_option_price, synthetic_quote
from quant_pairs.settlement import settlement_fee_btc
from quant_pairs.tardis_intraday import _option_fee


def build_daily_straddle_marks(
    prices: pd.DataFrame,
    dvol: pd.DataFrame,
    *,
    entry_at: pd.Timestamp,
    expiry_at: pd.Timestamp,
    strike: float,
    entry_underlying: float,
    entry_forward: float,
    entry_iv: float,
    relative_half_spread: float,
    iv_stress_points: float = 0.0,
    basis_stress_bps: float = 0.0,
    annualization_days: int = 365,
    contracts: float = 1.0,
) -> pd.DataFrame:
    """Mark a fixed call/put pair using only daily bars available by each decision.

    Deribit daily candle timestamps denote interval starts.  Their close is
    therefore made available one day later.  IV follows the change in DVOL
    from the last value available at entry, anchored to observed entry IV.
    The entry forward basis is carried as a constant annualized log yield.
    """

    entry = _utc(entry_at)
    expiry = _utc(expiry_at)
    if expiry <= entry:
        raise ValueError("expiry_at must be after entry_at")
    if strike <= 0 or entry_underlying <= 0 or entry_forward <= 0 or entry_iv <= 0:
        raise ValueError("entry prices, strike and IV must be positive")
    if (
        relative_half_spread < 0
        or iv_stress_points < 0
        or annualization_days <= 0
        or contracts <= 0
    ):
        raise ValueError("spread, IV stress and annualization must be non-negative")

    price_panel = _available_daily_closes(prices, value_name="underlying")
    dvol_panel = _available_daily_closes(dvol, value_name="dvol_points")
    entry_dvol = _last_available(dvol_panel, entry, "dvol_points") / 100
    decisions = price_panel.loc[
        (price_panel["available_at"] > entry) & (price_panel["available_at"] < expiry)
    ].copy()
    if decisions.empty:
        return pd.DataFrame()
    decisions = pd.merge_asof(
        decisions.sort_values("available_at"),
        dvol_panel.sort_values("available_at"),
        on="available_at",
        direction="backward",
    ).dropna(subset=["dvol_points"])

    entry_time = (expiry - entry).total_seconds() / (annualization_days * 86_400)
    basis_yield = math.log(entry_forward / entry_underlying) / entry_time
    rows: list[dict[str, object]] = []
    for row in decisions.itertuples(index=False):
        remaining = (expiry - row.available_at).total_seconds() / (
            annualization_days * 86_400
        )
        forward = float(row.underlying) * math.exp(basis_yield * remaining)
        forward *= math.exp(basis_stress_bps / 10_000)
        volatility = max(
            entry_iv + float(row.dvol_points) / 100 - entry_dvol + iv_stress_points / 100,
            0.01,
        )
        call_mid = inverse_option_price(
            "call", forward=forward, strike=strike, time_years=remaining, volatility=volatility
        )
        put_mid = inverse_option_price(
            "put", forward=forward, strike=strike, time_years=remaining, volatility=volatility
        )
        call_quote = synthetic_quote(
            call_mid, relative_half_spread=relative_half_spread
        )
        put_quote = synthetic_quote(put_mid, relative_half_spread=relative_half_spread)
        close_ask = (call_quote["ask_btc"] + put_quote["ask_btc"]) * contracts
        rows.append(
            {
                "source": "synthetic_model",
                "decision_at": row.available_at,
                "remaining_dte": (expiry - row.available_at).total_seconds() / 86_400,
                "underlying_usd": float(row.underlying),
                "forward_usd": forward,
                "dvol": float(row.dvol_points) / 100,
                "modeled_iv": volatility,
                "call_mid_btc": call_mid,
                "put_mid_btc": put_mid,
                "close_mid_btc": call_mid + put_mid,
                "close_ask_btc": close_ask,
                "close_fees_btc": contracts
                * (
                    _option_fee(call_quote["ask_btc"])
                    + _option_fee(put_quote["ask_btc"])
                ),
            }
        )
    return pd.DataFrame(rows)


def inject_gap_shock(
    marks: pd.DataFrame,
    *,
    strike: float,
    gap_return: float,
    iv_bump_points: float,
    contracts: float = 1.0,
) -> pd.DataFrame:
    """Insert an instantaneous close-to-close gap into an existing mark path.

    Daily marks cannot see intraday moves, so a real overnight gap is invisible
    to a close-based stop.  This applies the gap to the worst decision day: the
    day whose modeled forward, shocked by ``gap_return``, produces the most
    expensive straddle.  Skew widens in a crash, so IV is bumped by
    ``iv_bump_points`` at the same instant.  The re-marked row replaces the
    original for that day; later days keep their un-gapped path, since the model
    has no basis to propagate the shock forward.

    Returns the marks with the gapped row substituted in place.  The gapped row
    is flagged with ``gap_applied = True`` and carries the ``gap_return`` used.
    """

    if marks.empty:
        return marks
    if strike <= 0 or contracts <= 0:
        raise ValueError("strike and contracts must be positive")
    if not math.isfinite(gap_return) or gap_return <= -1:
        raise ValueError("gap_return must be finite and greater than -1")
    if iv_bump_points < 0:
        raise ValueError("iv_bump_points must be non-negative")

    ordered = marks.sort_values("decision_at").reset_index(drop=True)
    ordered["gap_applied"] = False
    ordered["gap_return"] = 0.0

    best_index = None
    best_cost = -math.inf
    best_payload: dict[str, object] | None = None
    for index, row in ordered.iterrows():
        remaining_years = float(row["remaining_dte"]) / 365
        forward = float(row["forward_usd"]) * (1 + gap_return)
        volatility = max(float(row["modeled_iv"]) + iv_bump_points / 100, 0.01)
        call_mid = inverse_option_price(
            "call", forward=forward, strike=strike,
            time_years=remaining_years, volatility=volatility,
        )
        put_mid = inverse_option_price(
            "put", forward=forward, strike=strike,
            time_years=remaining_years, volatility=volatility,
        )
        # A short position closes on the ask; the same relative half-spread the
        # original mark used is recovered from its own mid and ask.
        original_mid = float(row["close_mid_btc"])
        original_ask = float(row["close_ask_btc"]) / contracts
        relative_half_spread = (original_ask / original_mid - 1) if original_mid > 0 else 0.0
        call_quote = synthetic_quote(call_mid, relative_half_spread=relative_half_spread)
        put_quote = synthetic_quote(put_mid, relative_half_spread=relative_half_spread)
        close_ask = (call_quote["ask_btc"] + put_quote["ask_btc"]) * contracts
        cost = close_ask
        if cost > best_cost:
            best_cost = cost
            best_index = index
            best_payload = {
                "forward_usd": forward,
                "underlying_usd": float(row["underlying_usd"]) * (1 + gap_return),
                "modeled_iv": volatility,
                "call_mid_btc": call_mid,
                "put_mid_btc": put_mid,
                "close_mid_btc": call_mid + put_mid,
                "close_ask_btc": close_ask,
                "close_fees_btc": contracts
                * (_option_fee(call_quote["ask_btc"]) + _option_fee(put_quote["ask_btc"])),
                "gap_applied": True,
                "gap_return": gap_return,
            }

    if best_index is not None and best_payload is not None:
        for key, value in best_payload.items():
            ordered.loc[best_index, key] = value
    return ordered


def evaluate_short_exit(
    marks: pd.DataFrame,
    *,
    entry_credit_btc: float,
    profit_target: float,
    stop_multiple: float,
    exit_dte: float,
) -> dict[str, object]:
    """Apply a pre-declared short-straddle exit rule to chronological marks."""

    if entry_credit_btc <= 0:
        raise ValueError("entry_credit_btc must be positive")
    if not 0 < profit_target < 1 or stop_multiple <= 1 or exit_dte < 0:
        raise ValueError("invalid exit rule")
    required = {"decision_at", "remaining_dte", "close_ask_btc", "close_fees_btc"}
    missing = required.difference(marks.columns)
    if missing:
        raise ValueError(f"marks are missing required columns: {sorted(missing)}")
    if marks.empty:
        raise ValueError("no synthetic marks available")

    ordered = marks.sort_values("decision_at")
    chosen = None
    trigger = None
    for row in ordered.itertuples(index=False):
        cost = float(row.close_ask_btc) + float(row.close_fees_btc)
        if cost <= entry_credit_btc * (1 - profit_target):
            chosen, trigger = row, "profit_target"
            break
        if cost >= entry_credit_btc * stop_multiple:
            chosen, trigger = row, "stop_loss"
            break
        if float(row.remaining_dte) <= exit_dte:
            chosen, trigger = row, "dte_exit"
            break
    if chosen is None:
        chosen, trigger = ordered.iloc[-1], "last_available_mark"
        close_ask = float(chosen["close_ask_btc"])
        close_fees = float(chosen["close_fees_btc"])
        decision_at = chosen["decision_at"]
        remaining_dte = float(chosen["remaining_dte"])
    else:
        close_ask = float(chosen.close_ask_btc)
        close_fees = float(chosen.close_fees_btc)
        decision_at = chosen.decision_at
        remaining_dte = float(chosen.remaining_dte)
    pnl_before_entry_fee = entry_credit_btc - close_ask - close_fees
    return {
        "exit_trigger": trigger,
        "exit_at": str(decision_at),
        "remaining_dte": remaining_dte,
        "close_ask_btc": close_ask,
        "close_fees_btc": close_fees,
        "pnl_before_entry_fee_btc": pnl_before_entry_fee,
    }


def settle_short_straddle(
    *,
    entry_credit_btc: float,
    entry_fees_btc: float,
    settlement_payoff_per_contract_btc: float,
    contracts: float = 1.0,
) -> float:
    """Net short P&L for the observed settlement baseline."""

    return (
        entry_credit_btc
        - entry_fees_btc
        - settlement_payoff_per_contract_btc * contracts
        - settlement_fee_btc(settlement_payoff_per_contract_btc) * contracts
    )


def _available_daily_closes(frame: pd.DataFrame, *, value_name: str) -> pd.DataFrame:
    missing = {"timestamp", "close"}.difference(frame.columns)
    if missing:
        raise ValueError(f"daily frame is missing required columns: {sorted(missing)}")
    panel = frame.loc[:, ["timestamp", "close"]].copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True, format="mixed")
    panel[value_name] = pd.to_numeric(panel.pop("close"), errors="raise")
    panel["available_at"] = panel.pop("timestamp") + pd.Timedelta(days=1)
    return panel.drop_duplicates("available_at", keep="last").sort_values("available_at")


def _last_available(panel: pd.DataFrame, at: pd.Timestamp, column: str) -> float:
    available = panel.loc[panel["available_at"] <= at, column]
    if available.empty:
        raise ValueError(f"no {column} value available at entry")
    return float(available.iloc[-1])


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return timestamp.tz_convert("UTC")
