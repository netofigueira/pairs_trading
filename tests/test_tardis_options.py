import pandas as pd
import pytest

from quant_pairs.tardis_options import select_atm_straddle

AS_OF = pd.Timestamp("2024-01-01T12:00:00Z")


def _books(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"symbol": symbols, "bid_price": 0.03, "ask_price": 0.032}
    )


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

    selected = select_atm_straddle(
        books, underlying_mid=42_000.0, as_of=AS_OF, target_dte=14.0
    )

    assert sorted(selected["symbol"]) == ["BTC-15JAN24-41000-C", "BTC-15JAN24-41000-P"]


def test_target_dte_must_lie_within_bounds() -> None:
    with pytest.raises(ValueError, match="target_dte"):
        select_atm_straddle(
            _books([]), underlying_mid=42_000.0, as_of=AS_OF, min_dte=7, max_dte=10
        )
