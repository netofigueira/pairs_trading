import pandas as pd
import pytest

from quant_pairs.settlement import (
    delivery_price_on,
    fetch_delivery_prices,
    settlement_fee_btc,
    settlement_payoff_btc,
)


def test_fetch_delivery_prices_pages_until_total() -> None:
    data = [
        {"date": f"2024-01-{day:02d}", "delivery_price": 40_000.0 + day}
        for day in range(1, 8)
    ]

    def transport(endpoint, params):
        offset = int(params["offset"])
        return {"result": {"data": data[offset : offset + 5], "records_total": len(data)}}

    frame = fetch_delivery_prices("btc_usd", transport=transport)

    assert len(frame) == 7
    assert frame["date"].is_monotonic_increasing
    assert delivery_price_on(frame, pd.Timestamp("2024-01-03")) == 40_003.0
    with pytest.raises(ValueError, match="no unique delivery price"):
        delivery_price_on(frame, pd.Timestamp("2024-02-01"))


def test_inverse_settlement_payoff_and_fee() -> None:
    assert settlement_payoff_btc("call", 42_000, 46_200) == pytest.approx(4_200 / 46_200)
    assert settlement_payoff_btc("put", 42_000, 46_200) == 0.0
    assert settlement_payoff_btc("put", 42_000, 40_000) == pytest.approx(2_000 / 40_000)

    assert settlement_fee_btc(0.0) == 0.0
    assert settlement_fee_btc(0.1) == pytest.approx(0.00015)  # capped by per-contract fee
    assert settlement_fee_btc(0.0002) == pytest.approx(0.000025)  # capped at 12.5% of value
