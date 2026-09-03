import pandas as pd
import pytest

from quant_pairs.tape_straddle import select_daily_straddle_prints, short_entry_from_prints

DECISION = pd.Timestamp("2025-06-10 12:00", tz="UTC")


def _trade(instrument: str, minutes_off: int, price: float, iv: float, index: float = 100_000.0):
    return {
        "instrument_name": instrument,
        "traded_at": DECISION + pd.Timedelta(minutes=minutes_off),
        "price": price,
        "iv": iv,
        "index_price": index,
    }


def test_selects_pairable_atm_strike_near_target_dte() -> None:
    trades = pd.DataFrame(
        [
            # 14 DTE expiry, two strikes; only 100k is pairable pre-decision.
            _trade("BTC-24JUN25-100000-C", -30, 0.045, 52.0),
            _trade("BTC-24JUN25-100000-C", -5, 0.046, 52.5),
            _trade("BTC-24JUN25-100000-C", 5, 0.050, 60.0),  # future: must be ignored
            _trade("BTC-24JUN25-100000-P", -20, 0.044, 51.0),
            _trade("BTC-24JUN25-95000-P", -1, 0.02, 50.0),
            # nearer expiry pairable but further from 14 DTE target
            _trade("BTC-13JUN25-100000-C", -2, 0.02, 55.0),
            _trade("BTC-13JUN25-100000-P", -3, 0.019, 54.0),
            # out of DTE bounds
            _trade("BTC-26SEP25-100000-C", -4, 0.11, 60.0),
            # older than max_age: must be ignored
            _trade("BTC-24JUN25-100000-P", -180, 0.060, 70.0),
        ]
    )
    legs = select_daily_straddle_prints(trades, decision_at=DECISION)
    assert len(legs) == 2
    assert set(legs["type"]) == {"call", "put"}
    assert (legs["strike"] == 100_000.0).all()
    assert legs["instrument_name"].str.contains("24JUN25").all()
    call = legs.loc[legs["type"] == "call"].iloc[0]
    assert call["print_price_btc"] == 0.046  # the LAST pre-decision print
    assert call["print_iv"] == pytest.approx(0.525)
    assert (legs["seconds_from_decision"] >= 0).all()


def test_empty_when_no_pairable_strike() -> None:
    trades = pd.DataFrame(
        [
            _trade("BTC-24JUN25-100000-C", 0, 0.045, 52.0),
            _trade("BTC-24JUN25-95000-P", -1, 0.02, 50.0),
        ]
    )
    assert select_daily_straddle_prints(trades, decision_at=DECISION).empty


def test_short_entry_discounts_prints_and_inverts_bid_iv() -> None:
    trades = pd.DataFrame(
        [
            _trade("BTC-24JUN25-100000-C", 0, 0.045, 52.0),
            _trade("BTC-24JUN25-100000-P", -1, 0.044, 51.0),
        ]
    )
    legs = select_daily_straddle_prints(trades, decision_at=DECISION)
    entry = short_entry_from_prints(legs, relative_half_spread=0.02, contracts=0.1)
    assert entry["entry_credit_btc"] == pytest.approx((0.045 + 0.044) * 0.98 * 0.1)
    for leg in entry["legs"]:
        assert leg["bid_iv"] < leg["entry_iv"] + 0.05
        assert leg["bid_iv"] > 0.2
    assert entry["mean_bid_variance"] > 0
