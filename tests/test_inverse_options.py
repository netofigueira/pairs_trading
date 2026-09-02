import pytest

from quant_pairs.inverse_options import (
    implied_volatility,
    inverse_forward_from_parity,
    inverse_intrinsic_value,
    inverse_option_price,
    synthetic_quote,
)


def test_inverse_prices_respect_put_call_parity() -> None:
    parameters = dict(forward=42_000.0, strike=40_000.0, time_years=30 / 365, volatility=0.6)
    call = inverse_option_price("call", **parameters)
    put = inverse_option_price("put", **parameters)

    assert call - put == pytest.approx(1 - parameters["strike"] / parameters["forward"])


def test_inverse_forward_round_trips_put_call_parity() -> None:
    parameters = dict(forward=43_150.0, strike=43_000.0, time_years=14 / 365, volatility=0.65)
    call = inverse_option_price("call", **parameters)
    put = inverse_option_price("put", **parameters)

    assert inverse_forward_from_parity(
        call_price_btc=call, put_price_btc=put, strike=parameters["strike"]
    ) == pytest.approx(parameters["forward"])


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_volatility_round_trip(option_type: str) -> None:
    parameters = dict(forward=42_000.0, strike=43_000.0, time_years=14 / 365)
    price = inverse_option_price(option_type, volatility=0.725, **parameters)

    solved = implied_volatility(option_type, price_btc=price, **parameters)

    assert solved == pytest.approx(0.725, abs=1e-7)


def test_expiry_price_is_inverse_intrinsic_value() -> None:
    assert inverse_option_price(
        "call", forward=46_200, strike=42_000, time_years=0, volatility=0.8
    ) == pytest.approx(4_200 / 46_200)
    assert inverse_intrinsic_value("put", forward=40_000, strike=42_000) == pytest.approx(0.05)


def test_implied_volatility_rejects_price_below_intrinsic() -> None:
    with pytest.raises(ValueError, match="below inverse intrinsic"):
        implied_volatility(
            "call",
            price_btc=0.01,
            forward=50_000,
            strike=40_000,
            time_years=14 / 365,
        )


def test_synthetic_quote_applies_spread_to_option_premium() -> None:
    quote = synthetic_quote(0.05, relative_half_spread=0.10)

    assert quote["source"] == "synthetic_model"
    assert quote["bid_btc"] == pytest.approx(0.045)
    assert quote["ask_btc"] == pytest.approx(0.055)
    assert quote["half_spread_btc"] == pytest.approx(0.005)


def test_synthetic_quote_respects_absolute_spread_floor_and_nonnegative_bid() -> None:
    quote = synthetic_quote(
        0.0002, relative_half_spread=0.10, minimum_half_spread_btc=0.0005
    )

    assert quote["bid_btc"] == 0.0
    assert quote["ask_btc"] == pytest.approx(0.0007)
