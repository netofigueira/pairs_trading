import pandas as pd
import pytest

from quant_pairs.volatility_report import build_volatility_report


def test_report_exposes_variance_premium_and_comparable_carry_returns() -> None:
    timestamps = pd.date_range("2026-01-01T08:00:00Z", periods=4, freq="D")
    prices = pd.DataFrame({"timestamp": timestamps, "close": [100.0, 101.0, 102.0, 103.0]})
    dvol = pd.DataFrame({"timestamp": ["2026-01-01T00:00:00Z"], "close": [50.0]})
    carry = [_carry_row("2026-01-01", contracts=1.0, premium=0.1, pnl=-0.02)]
    recovered = [_carry_row("2026-04-01", contracts=0.1, premium=0.01, pnl=0.005)]

    report = build_volatility_report(
        dvol, prices, carry, recovered, horizon_days=2
    )

    point = report["calibration"]["independent"][0]
    assert point["variance_premium"] == pytest.approx(
        point["iv"] ** 2 - point["forward_rv"] ** 2
    )
    assert report["carry"]["summary"]["comparable_observations"] == 2
    assert report["carry"]["summary"]["positive"] == 1
    assert [point["return_on_premium"] for point in report["carry"]["points"]] == [
        pytest.approx(-0.2),
        pytest.approx(0.5),
    ]


def _carry_row(date: str, *, contracts: float, premium: float, pnl: float) -> dict:
    return {
        "status": "carry_unhedged_settled",
        "entry_at": f"{date} 12:00:00+00:00",
        "expiry_at": f"{date} 12:00:00+00:00",
        "days_held": 14.0,
        "contracts_per_leg": contracts,
        "entry_premium_btc": premium,
        "settlement_payoff_btc": premium + pnl,
        "option_entry_fees_btc": 0.0,
        "settlement_fees_btc": 0.0,
        "net_unhedged_pnl_btc": pnl,
        "legs": [{"strike": 100_000.0}, {"strike": 100_000.0}],
    }
