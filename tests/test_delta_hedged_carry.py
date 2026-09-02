import math

import pandas as pd
import pytest

from quant_pairs.delta_hedged_carry import (
    simulate_delta_hedged_short,
    straddle_delta_btc,
)
from quant_pairs.inverse_options import inverse_option_price
from quant_pairs.synthetic_option_backfill import build_daily_straddle_marks


def _flat_funding(start: str, hours: int, index_price: float) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=hours, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "index_price": index_price,
            "interest_1h": 0.0,
            "interest_8h": 0.0,
        }
    )


def _daily(start: str, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(closes), freq="1D", tz="UTC"),
            "close": closes,
        }
    )


def test_straddle_delta_signs_follow_moneyness() -> None:
    deep_call = straddle_delta_btc(
        underlying=200.0, forward=200.0, strike=100.0, time_years=0.05, volatility=0.6
    )
    deep_put = straddle_delta_btc(
        underlying=50.0, forward=50.0, strike=100.0, time_years=0.05, volatility=0.6
    )
    assert deep_call > 0
    assert deep_put < 0
    # Deep in the call region the BTC value tends to 1 - K/F, so dV/dS -> K/(F*S).
    assert deep_call == pytest.approx(100.0 / (200.0 * 200.0), rel=0.05)


def test_straddle_delta_matches_manual_finite_difference() -> None:
    kwargs = dict(strike=100.0, time_years=14 / 365, volatility=0.8)
    bump = 1e-2
    manual = (
        sum(inverse_option_price(t, forward=105.0 + bump, **kwargs) for t in ("call", "put"))
        - sum(inverse_option_price(t, forward=105.0 - bump, **kwargs) for t in ("call", "put"))
    ) / (2 * bump)
    computed = straddle_delta_btc(
        underlying=105.0, forward=105.0, time_years=14 / 365, volatility=0.8, strike=100.0
    )
    assert computed == pytest.approx(manual, rel=1e-4)


def test_hedge_reduces_directional_daily_variance() -> None:
    # Strong drift, constant vol: the hedge should absorb most of the option
    # leg's directional P&L on mid marks.
    closes = [100.0, 106.0, 112.0, 118.0, 124.0]
    prices = _daily("2026-01-01 00:00", closes)
    dvol = _daily("2026-01-01 00:00", [60.0] * len(closes))
    entry_at = pd.Timestamp("2026-01-02 12:00", tz="UTC")
    expiry_at = pd.Timestamp("2026-01-08 08:00", tz="UTC")
    marks = build_daily_straddle_marks(
        prices,
        dvol,
        entry_at=entry_at,
        expiry_at=expiry_at,
        strike=100.0,
        entry_underlying=100.0,
        entry_forward=100.0,
        entry_iv=0.60,
        relative_half_spread=0.0,
        contracts=1.0,
    )
    funding = _flat_funding("2026-01-02 13:00", 24 * 7, 110.0)
    result = simulate_delta_hedged_short(
        marks,
        entry_at=entry_at,
        expiry_at=expiry_at,
        strike=100.0,
        contracts=1.0,
        entry_underlying=100.0,
        entry_forward=100.0,
        entry_iv=0.60,
        entry_credit_btc=0.05,
        entry_fees_btc=0.0005,
        delivery_price=124.0,
        funding=funding,
        perp_contract_size_usd=0.01,  # fine-grained rounding for a small test book
    )
    assert result["funding_pnl_btc"] == 0.0
    # The rally hurts the short straddle; the long hedge must earn part back.
    assert result["hedge_trading_pnl_btc"] > 0
    assert result["hedged_pnl_btc"] > result["unhedged_pnl_btc"]
    assert result["hedged_pnl_btc"] == pytest.approx(
        result["option_pnl_btc"]
        + result["hedge_trading_pnl_btc"]
        + result["funding_pnl_btc"]
        - result["hedge_fees_btc"]
    )
    assert result["rebalances"] == 1 + len(marks)


def test_funding_charges_long_hedge_when_interest_positive() -> None:
    closes = [100.0, 130.0]
    prices = _daily("2026-01-01 00:00", closes)
    dvol = _daily("2026-01-01 00:00", [60.0, 60.0])
    entry_at = pd.Timestamp("2026-01-02 12:00", tz="UTC")
    expiry_at = pd.Timestamp("2026-01-04 08:00", tz="UTC")
    marks = build_daily_straddle_marks(
        prices,
        dvol,
        entry_at=entry_at,
        expiry_at=expiry_at,
        strike=90.0,  # call-dominant: positive straddle delta, long hedge
        entry_underlying=100.0,
        entry_forward=100.0,
        entry_iv=0.60,
        relative_half_spread=0.0,
    )
    funding = _flat_funding("2026-01-02 13:00", 24 * 3, 100.0)
    funding["interest_1h"] = 0.0001
    result = simulate_delta_hedged_short(
        marks,
        entry_at=entry_at,
        expiry_at=expiry_at,
        strike=90.0,
        contracts=1.0,
        entry_underlying=100.0,
        entry_forward=100.0,
        entry_iv=0.60,
        entry_credit_btc=0.15,
        entry_fees_btc=0.0005,
        delivery_price=130.0,
        funding=funding,
        perp_contract_size_usd=0.01,
    )
    assert result["funding_pnl_btc"] < 0


def test_rejects_unaligned_entry_hour() -> None:
    funding = _flat_funding("2026-01-02 13:00", 24, 100.0)
    with pytest.raises(ValueError, match="full UTC hours"):
        simulate_delta_hedged_short(
            pd.DataFrame(),
            entry_at=pd.Timestamp("2026-01-02 12:30", tz="UTC"),
            expiry_at=pd.Timestamp("2026-01-03 08:00", tz="UTC"),
            strike=100.0,
            contracts=1.0,
            entry_underlying=100.0,
            entry_forward=100.0,
            entry_iv=0.6,
            entry_credit_btc=0.05,
            entry_fees_btc=0.0,
            delivery_price=100.0,
            funding=funding,
        )


def test_delta_neutral_book_is_flat_to_first_order() -> None:
    # One tiny move with no time passing pricing effects dominating: the hedged
    # first step should be far smaller than the unhedged option step.
    entry_underlying = 100.0
    entry_iv = 0.60
    strike = 100.0
    closes = [100.0, 100.4]
    prices = _daily("2026-01-01 00:00", closes)
    dvol = _daily("2026-01-01 00:00", [60.0, 60.0])
    entry_at = pd.Timestamp("2026-01-02 00:00", tz="UTC")
    expiry_at = pd.Timestamp("2026-01-16 00:00", tz="UTC")
    marks = build_daily_straddle_marks(
        prices,
        dvol,
        entry_at=entry_at,
        expiry_at=expiry_at,
        strike=strike,
        entry_underlying=entry_underlying,
        entry_forward=entry_underlying,
        entry_iv=entry_iv,
        relative_half_spread=0.0,
    )
    assert len(marks) == 1
    funding = _flat_funding("2026-01-02 01:00", 24 * 14, 100.0)
    result = simulate_delta_hedged_short(
        marks,
        entry_at=entry_at,
        expiry_at=expiry_at,
        strike=strike,
        contracts=1.0,
        entry_underlying=entry_underlying,
        entry_forward=entry_underlying,
        entry_iv=entry_iv,
        entry_credit_btc=0.2,
        entry_fees_btc=0.0,
        delivery_price=100.4,
        funding=funding,
        perp_taker_fee_rate=0.0,
        perp_contract_size_usd=0.01,
    )
    daily = result["daily"]
    entry_row, mark_row = daily[0], daily[1]
    option_step = float(mark_row["short_straddle_mid_btc"]) - float(
        entry_row["short_straddle_mid_btc"]
    )
    hedge_step = float(entry_row["segment_hedge_pnl_btc"])
    # Theta helps the short here, so compare directional sensitivity instead:
    # the hedge offsets the delta component within second-order terms.
    residual = option_step + hedge_step
    assert abs(residual) < abs(option_step)
    assert not math.isnan(residual)


def test_basket_matches_straddle_simulation() -> None:
    closes = [100.0, 104.0, 99.0, 103.0]
    prices = _daily("2026-01-01 00:00", closes)
    dvol = _daily("2026-01-01 00:00", [60.0, 62.0, 58.0, 61.0])
    entry_at = pd.Timestamp("2026-01-02 12:00", tz="UTC")
    expiry_at = pd.Timestamp("2026-01-07 08:00", tz="UTC")
    common = dict(
        entry_at=entry_at,
        expiry_at=expiry_at,
        contracts=0.5,
        entry_underlying=100.0,
        entry_forward=100.5,
        entry_credit_btc=0.06,
        entry_fees_btc=0.0004,
        delivery_price=103.0,
        funding=_flat_funding("2026-01-02 13:00", 24 * 6, 100.0),
    )
    marks = build_daily_straddle_marks(
        prices,
        dvol,
        entry_at=entry_at,
        expiry_at=expiry_at,
        strike=100.0,
        entry_underlying=100.0,
        entry_forward=100.5,
        entry_iv=0.60,
        relative_half_spread=0.0,
    )
    from quant_pairs.delta_hedged_carry import simulate_delta_hedged_short_basket

    via_marks = simulate_delta_hedged_short(marks, strike=100.0, entry_iv=0.60, **common)
    via_basket = simulate_delta_hedged_short_basket(
        prices,
        dvol,
        legs=[
            {"type": "call", "strike": 100.0, "entry_iv": 0.60},
            {"type": "put", "strike": 100.0, "entry_iv": 0.60},
        ],
        **common,
    )
    assert via_basket["hedged_pnl_btc"] == pytest.approx(via_marks["hedged_pnl_btc"], abs=1e-12)
    assert via_basket["hedge_fees_btc"] == pytest.approx(via_marks["hedge_fees_btc"], abs=1e-12)
