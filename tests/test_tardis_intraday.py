import pandas as pd
import pytest

from quant_pairs.tardis_intraday import run_intraday_straddle


def test_intraday_runner_uses_same_contracts_and_executable_sides(tmp_path) -> None:
    option_path = tmp_path / "OPTIONS.csv.gz"
    perp_path = tmp_path / "BTC-PERPETUAL.csv.gz"
    entry_us = int(pd.Timestamp("2024-01-01T12:00:00Z").timestamp() * 1_000_000)
    exit_us = int(pd.Timestamp("2024-01-01T20:00:00Z").timestamp() * 1_000_000)
    option_rows = []
    for symbol, entry_bid, entry_ask, exit_bid, exit_ask in (
        ("BTC-12JAN24-42000-C", 0.040, 0.042, 0.045, 0.047),
        ("BTC-12JAN24-42000-P", 0.030, 0.032, 0.025, 0.027),
        ("BTC-12JAN24-50000-C", 0.010, 0.012, 0.011, 0.013),
        ("BTC-12JAN24-50000-P", 0.190, 0.192, 0.180, 0.182),
    ):
        option_rows.extend(
            [
                _quote(symbol, entry_us, entry_bid, entry_ask, amount=2.0),
                _quote(symbol, exit_us, exit_bid, exit_ask, amount=2.0),
            ]
        )
    pd.DataFrame(option_rows).to_csv(option_path, index=False, compression="gzip")
    pd.DataFrame(
        [
            _quote("BTC-PERPETUAL", entry_us, 41_999.0, 42_001.0, amount=100_000),
            _quote("BTC-PERPETUAL", exit_us, 42_999.0, 43_001.0, amount=100_000),
        ]
    ).to_csv(perp_path, index=False, compression="gzip")

    result = run_intraday_straddle(
        option_path,
        perp_path,
        entry_at=pd.Timestamp("2024-01-01T12:00:00Z"),
        exit_at=pd.Timestamp("2024-01-01T20:00:00Z"),
        max_age=pd.Timedelta(seconds=1),
    )

    assert [leg["symbol"] for leg in result["legs"]] == [
        "BTC-12JAN24-42000-C",
        "BTC-12JAN24-42000-P",
    ]
    assert result["gross_mid_pnl_btc"] == 0.0
    assert result["spread_cost_btc"] == pytest.approx(0.004)
    assert result["executable_pnl_before_fees_btc"] == pytest.approx(-0.004)
    assert result["option_fees_btc"] == pytest.approx(0.0012)
    assert result["net_unhedged_pnl_btc"] == pytest.approx(-0.0052)
    assert result["net_delta_hedged_pnl_btc"] is None


def test_intraday_runner_uses_observed_deltas_for_inverse_perp_hedge(tmp_path) -> None:
    option_path = tmp_path / "OPTIONS.csv.gz"
    perp_path = tmp_path / "BTC-PERPETUAL.csv.gz"
    chain_path = tmp_path / "CHAIN.csv.gz"
    entry = pd.Timestamp("2024-01-01T12:00:00Z")
    exit_ = pd.Timestamp("2024-01-01T20:00:00Z")
    entry_us = int(entry.timestamp() * 1_000_000)
    exit_us = int(exit_.timestamp() * 1_000_000)
    symbols = ["BTC-12JAN24-42000-C", "BTC-12JAN24-42000-P"]
    pd.DataFrame(
        [
            _quote(symbol, timestamp, bid, ask, amount=2.0)
            for symbol in symbols
            for timestamp, bid, ask in (
                (entry_us, 0.03, 0.032),
                (exit_us, 0.031, 0.033),
            )
        ]
    ).to_csv(option_path, index=False, compression="gzip")
    pd.DataFrame(
        [
            _quote("BTC-PERPETUAL", entry_us, 41_999.0, 42_001.0, amount=100_000),
            _quote("BTC-PERPETUAL", exit_us, 42_999.0, 43_001.0, amount=100_000),
        ]
    ).to_csv(perp_path, index=False, compression="gzip")
    pd.DataFrame(
        [
            {
                "symbol": symbols[0],
                "timestamp": entry_us,
                "local_timestamp": entry_us,
                "delta": 0.55,
            },
            {
                "symbol": symbols[1],
                "timestamp": entry_us,
                "local_timestamp": entry_us,
                "delta": -0.45,
            },
            {
                "symbol": symbols[0],
                "timestamp": exit_us,
                "local_timestamp": exit_us,
                "delta": 0.60,
            },
        ]
    ).to_csv(chain_path, index=False, compression="gzip")

    result = run_intraday_straddle(
        option_path,
        perp_path,
        entry_at=entry,
        exit_at=exit_,
        max_age=pd.Timedelta(seconds=1),
        options_chain_path=chain_path,
    )

    assert result["status"] == "delta_hedged_intraday_plumbing_missing_funding"
    assert result["entry_option_delta_btc"] == pytest.approx(0.1)
    assert result["hedge_contracts"] == -420
    assert abs(result["entry_residual_delta_btc"]) < 1e-12
    assert result["delta_hedge_pnl_btc"] < 0
    assert result["delta_hedge_fees_btc"] > 0
    assert result["net_delta_hedged_before_funding_btc"] is not None
    assert result["funding_pnl_btc"] is None
    assert result["net_delta_hedged_pnl_btc"] is None


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
