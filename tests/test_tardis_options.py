import pandas as pd
import pytest

from quant_pairs.inverse_options import inverse_option_price, synthetic_quote
from quant_pairs.tardis_options import select_atm_straddle, select_strangle_by_delta

AS_OF = pd.Timestamp("2024-01-01T12:00:00Z")


def _books(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"symbol": symbols, "bid_price": 0.03, "ask_price": 0.032})


def test_expiry_is_chosen_by_target_dte_before_the_atm_strike() -> None:
    # The 8 DTE expiry holds the most-ATM strike; the rule must still pick the
    # expiry nearest 14 DTE and only then go ATM within it.
    books = _books(
        [
            "BTC-9JAN24-42000-C",
            "BTC-9JAN24-42000-P",
            "BTC-15JAN24-41000-C",
            "BTC-15JAN24-41000-P",
            "BTC-15JAN24-44000-C",
            "BTC-15JAN24-44000-P",
        ]
    )

    selected = select_atm_straddle(books, underlying_mid=42_000.0, as_of=AS_OF, target_dte=14.0)

    assert sorted(selected["symbol"]) == ["BTC-15JAN24-41000-C", "BTC-15JAN24-41000-P"]


def test_target_dte_must_lie_within_bounds() -> None:
    with pytest.raises(ValueError, match="target_dte"):
        select_atm_straddle(_books([]), underlying_mid=42_000.0, as_of=AS_OF, min_dte=7, max_dte=10)


def _book(forward: float, expiry_label: str, strikes: list[float], iv: float, tau: float):
    rows = []
    for strike in strikes:
        for option_type, suffix in (("call", "C"), ("put", "P")):
            mid = inverse_option_price(
                option_type, forward=forward, strike=strike, time_years=tau, volatility=iv
            )
            quote = synthetic_quote(mid, relative_half_spread=0.02)
            rows.append(
                {
                    "symbol": f"BTC-{expiry_label}-{int(strike)}-{suffix}",
                    "bid_price": quote["bid_btc"],
                    "ask_price": quote["ask_btc"],
                    "bid_amount": 10.0,
                    "ask_amount": 10.0,
                }
            )
    return pd.DataFrame(rows)


def test_selects_otm_pair_near_target_delta() -> None:
    as_of = pd.Timestamp("2026-01-02 12:00", tz="UTC")
    expiry = pd.Timestamp("2026-01-16 08:00", tz="UTC")
    tau = (expiry - as_of).total_seconds() / (365 * 86_400)
    strikes = [80_000, 90_000, 95_000, 100_000, 105_000, 110_000, 120_000]
    book = _book(100_000.0, "16JAN26", strikes, iv=0.6, tau=tau)
    picked = select_strangle_by_delta(
        book, forward=100_000.0, as_of=as_of, expiry=expiry, target_delta=0.25
    )
    assert len(picked) == 2
    call = picked.loc[picked["type"] == "call"].iloc[0]
    put = picked.loc[picked["type"] == "put"].iloc[0]
    assert call["strike"] > 100_000
    assert put["strike"] < 100_000
    assert call["forward_delta"] == pytest.approx(0.25, abs=0.12)
    assert put["forward_delta"] == pytest.approx(-0.25, abs=0.12)


def test_empty_when_size_below_minimum() -> None:
    as_of = pd.Timestamp("2026-01-02 12:00", tz="UTC")
    expiry = pd.Timestamp("2026-01-16 08:00", tz="UTC")
    tau = (expiry - as_of).total_seconds() / (365 * 86_400)
    book = _book(100_000.0, "16JAN26", [95_000, 105_000], iv=0.6, tau=tau)
    picked = select_strangle_by_delta(
        book, forward=100_000.0, as_of=as_of, expiry=expiry, min_contracts=50.0
    )
    assert picked.empty
