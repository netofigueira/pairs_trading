import pandas as pd
import pytest

from quant_pairs.tardis_carry import run_carry_straddle

ENTRY = pd.Timestamp("2024-01-01T12:00:00Z")
EXPIRY = pd.Timestamp("2024-01-12T08:00:00Z")


def _quote(symbol: str, timestamp: int, bid: float, ask: float, *, amount: float) -> dict:
    return {
        "exchange": "deribit",
        "symbol": symbol,
        "timestamp": timestamp,
        "local_timestamp": timestamp,
        "ask_amount": amount,
        "ask_price": ask,
        "bid_price": bid,
        "bid_amount": amount,
    }


def _write_entry_books(tmp_path):
    option_path = tmp_path / "OPTIONS.csv.gz"
    perp_path = tmp_path / "BTC-PERPETUAL.csv.gz"
    entry_us = int(ENTRY.timestamp() * 1_000_000)
    pd.DataFrame(
        [
            _quote("BTC-12JAN24-42000-C", entry_us, 0.040, 0.042, amount=2.0),
            _quote("BTC-12JAN24-42000-P", entry_us, 0.030, 0.032, amount=2.0),
        ]
    ).to_csv(option_path, index=False, compression="gzip")
    pd.DataFrame(
        [_quote("BTC-PERPETUAL", entry_us, 41_999.0, 42_001.0, amount=100_000)]
    ).to_csv(perp_path, index=False, compression="gzip")
    return option_path, perp_path


DELIVERY = pd.DataFrame(
    {"date": [pd.Timestamp("2024-01-12", tz="UTC")], "delivery_price": [46_200.0]}
)


def test_carry_settles_at_official_delivery_price(tmp_path) -> None:
    option_path, perp_path = _write_entry_books(tmp_path)

    result = run_carry_straddle(
        option_path,
        perp_path,
        entry_at=ENTRY,
        delivery_prices=DELIVERY,
        max_age=pd.Timedelta(seconds=1),
    )

    call_payoff = 4_200 / 46_200
    assert result["status"] == "carry_unhedged_settled"
    assert result["delivery_price_usd"] == 46_200.0
    assert result["settlement_payoff_btc"] == pytest.approx(call_payoff)
    assert result["entry_premium_btc"] == pytest.approx(0.074)
    assert result["option_entry_fees_btc"] == pytest.approx(0.0006)
    assert result["settlement_fees_btc"] == pytest.approx(0.00015)
    assert result["net_unhedged_pnl_btc"] == pytest.approx(
        call_payoff - 0.074 - 0.0006 - 0.00015
    )
    assert result["net_static_hedged_pnl_btc"] is None


def test_carry_static_hedge_holds_perp_to_settlement_with_funding(tmp_path) -> None:
    option_path, perp_path = _write_entry_books(tmp_path)
    chain_path = tmp_path / "CHAIN.csv.gz"
    entry_us = int(ENTRY.timestamp() * 1_000_000)
    pd.DataFrame(
        [
            {"symbol": "BTC-12JAN24-42000-C", "timestamp": entry_us, "local_timestamp": entry_us, "delta": 0.55},
            {"symbol": "BTC-12JAN24-42000-P", "timestamp": entry_us, "local_timestamp": entry_us, "delta": -0.45},
        ]
    ).to_csv(chain_path, index=False, compression="gzip")
    hours = int((EXPIRY - ENTRY) / pd.Timedelta(hours=1))
    funding = pd.DataFrame(
        {
            "timestamp": [ENTRY + pd.Timedelta(hours=i) for i in range(1, hours + 1)],
            "index_price": [42_000.0] * hours,
            "interest_1h": [1e-5] * hours,
            "interest_8h": [8e-5] * hours,
        }
    )

    result = run_carry_straddle(
        option_path,
        perp_path,
        entry_at=ENTRY,
        delivery_prices=DELIVERY,
        max_age=pd.Timedelta(seconds=1),
        options_chain_path=chain_path,
        funding=funding,
    )

    assert result["status"] == "carry_static_hedged_settled"
    assert result["hedge_contracts"] == -420  # -0.1 BTC * 42000 / 10 USD
    assert result["hedge_exit_price_source"] == "delivery_price"
    # short entered at bid 41999, closed at delivery 46200
    assert result["hedge_pnl_btc"] == pytest.approx(-420 * 10 * (1 / 41_999 - 1 / 46_200))
    assert result["hedge_pnl_btc"] < 0
    # short position receives positive funding
    assert result["funding_pnl_btc"] == pytest.approx(420 * 10 / 42_000 * 1e-5 * hours)
    assert result["net_static_hedged_pnl_btc"] == pytest.approx(
        result["net_unhedged_pnl_btc"]
        + result["hedge_pnl_btc"]
        - result["hedge_fees_btc"]
        + result["funding_pnl_btc"]
    )


def test_carry_hedge_requires_funding_history(tmp_path) -> None:
    option_path, perp_path = _write_entry_books(tmp_path)

    with pytest.raises(ValueError, match="requires funding history"):
        run_carry_straddle(
            option_path,
            perp_path,
            entry_at=ENTRY,
            delivery_prices=DELIVERY,
            max_age=pd.Timedelta(seconds=1),
            options_chain_path=tmp_path / "CHAIN.csv.gz",
        )
