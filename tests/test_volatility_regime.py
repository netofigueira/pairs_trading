import pytest

from quant_pairs.volatility_regime import build_economic_gate, classify_volatility_regime


def test_frozen_rule_uses_executable_iv_boundaries() -> None:
    assert classify_volatility_regime(0.55**2, bid_ivs=[0.40, 0.42], ask_ivs=[0.45, 0.47]) == "long"
    assert (
        classify_volatility_regime(0.35**2, bid_ivs=[0.40, 0.42], ask_ivs=[0.45, 0.47]) == "short"
    )
    assert classify_volatility_regime(0.44**2, bid_ivs=[0.40, 0.42], ask_ivs=[0.45, 0.47]) == "flat"


def test_gate_uses_latest_asof_forecast_and_normalizes_settlement() -> None:
    forecasts = [
        {"forecast_at": "2022-03-30T08:00:00Z", "garch_corrected_rv": 0.60},
        {"forecast_at": "2022-04-02T08:00:00Z", "garch_corrected_rv": 0.10},
    ]
    options = [
        {
            "entry_at": "2022-04-01T12:00:00Z",
            "bid_iv": 0.40,
            "ask_iv": 0.45,
            "bid_btc": 0.04,
            "ask_btc": 0.05,
        },
        {
            "entry_at": "2022-04-01T12:00:00Z",
            "bid_iv": 0.41,
            "ask_iv": 0.46,
            "bid_btc": 0.04,
            "ask_btc": 0.05,
        },
    ]
    carry = [
        {
            "status": "carry_unhedged_settled",
            "entry_at": "2022-04-01T12:00:00Z",
            "contracts_per_leg": 0.1,
            "settlement_payoff_btc": 0.02,
            "settlement_fees_btc": 0.000015,
        }
    ]

    gate = build_economic_gate(
        forecasts, options, carry, contracts=0.1, minimum_actionable=1, minimum_each_side=1
    )

    point = gate["points"][0]
    assert point["action"] == "long"
    assert point["forecast_at"].startswith("2022-03-30")
    assert point["long_pnl_btc"] == pytest.approx((0.2 - 0.1 - 0.0006 - 0.00015) * 0.1)
