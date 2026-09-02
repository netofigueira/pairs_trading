import pandas as pd
import pytest

from quant_pairs.synthetic_option_backfill import (
    build_daily_straddle_marks,
    evaluate_short_exit,
    inject_gap_shock,
)


def _daily(start: str, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(closes), freq="1D", tz="UTC"),
            "close": closes,
        }
    )


def test_daily_marks_delay_candle_close_until_next_day() -> None:
    prices = _daily("2026-01-01 08:00", [100.0, 101.0, 102.0])
    dvol = _daily("2026-01-01 00:00", [50.0, 55.0, 60.0])

    marks = build_daily_straddle_marks(
        prices,
        dvol,
        entry_at=pd.Timestamp("2026-01-02 12:00", tz="UTC"),
        expiry_at=pd.Timestamp("2026-01-05 08:00", tz="UTC"),
        strike=100,
        entry_underlying=100,
        entry_forward=100,
        entry_iv=0.50,
        relative_half_spread=0.02,
    )

    assert marks["decision_at"].tolist() == [
        pd.Timestamp("2026-01-03 08:00", tz="UTC"),
        pd.Timestamp("2026-01-04 08:00", tz="UTC"),
    ]
    assert marks.iloc[0]["underlying_usd"] == 101.0
    assert marks.iloc[0]["dvol"] == 0.55
    assert marks.iloc[0]["modeled_iv"] == pytest.approx(0.55)
    assert set(marks["source"]) == {"synthetic_model"}


def test_short_exit_uses_synthetic_ask_and_fees() -> None:
    marks = pd.DataFrame(
        {
            "decision_at": pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True),
            "remaining_dte": [5.0, 4.0],
            "close_ask_btc": [0.08, 0.04],
            "close_fees_btc": [0.001, 0.001],
        }
    )

    result = evaluate_short_exit(
        marks,
        entry_credit_btc=0.10,
        profit_target=0.50,
        stop_multiple=2.0,
        exit_dte=1.0,
    )

    assert result["exit_trigger"] == "profit_target"
    assert result["close_ask_btc"] == 0.04
    assert result["pnl_before_entry_fee_btc"] == pytest.approx(0.059)


def test_gap_shock_makes_short_straddle_more_expensive_and_fires_stop() -> None:
    prices = _daily("2026-01-01 08:00", [100.0, 100.0, 100.0, 100.0])
    dvol = _daily("2026-01-01 00:00", [50.0, 50.0, 50.0, 50.0])
    marks = build_daily_straddle_marks(
        prices,
        dvol,
        entry_at=pd.Timestamp("2026-01-02 12:00", tz="UTC"),
        expiry_at=pd.Timestamp("2026-01-06 08:00", tz="UTC"),
        strike=100,
        entry_underlying=100,
        entry_forward=100,
        entry_iv=0.50,
        relative_half_spread=0.02,
        contracts=1.0,
    )
    base_credit = float(marks.iloc[0]["close_mid_btc"])

    gapped = inject_gap_shock(
        marks, strike=100, gap_return=-0.20, iv_bump_points=15.0, contracts=1.0
    )

    shocked = gapped.loc[gapped["gap_applied"]]
    assert len(shocked) == 1
    # A -20% underlying move plus a vol bump makes the ATM straddle far more
    # expensive to buy back than the un-gapped mid at entry.
    assert float(shocked.iloc[0]["close_ask_btc"]) > base_credit * 1.5
    # The un-gapped days keep their original half-spread relationship.
    assert (gapped.loc[~gapped["gap_applied"], "gap_return"] == 0.0).all()

    result = evaluate_short_exit(
        gapped,
        entry_credit_btc=base_credit,
        profit_target=0.50,
        stop_multiple=2.0,
        exit_dte=1.0,
    )
    assert result["exit_trigger"] == "stop_loss"
    # The realized buy-back cost overshoots the 2x stop level: the daily close
    # cannot catch the gap in between.
    assert result["close_ask_btc"] > base_credit * 2.0


def test_gap_shock_is_noop_on_empty_marks() -> None:
    empty = pd.DataFrame(columns=["decision_at", "remaining_dte", "close_ask_btc"])
    out = inject_gap_shock(empty, strike=100, gap_return=-0.2, iv_bump_points=10.0)
    assert out.empty


def test_gap_shock_rejects_a_non_positive_forward_multiplier() -> None:
    marks = pd.DataFrame({"decision_at": [pd.Timestamp("2026-01-02", tz="UTC")]})
    with pytest.raises(ValueError, match="greater than -1"):
        inject_gap_shock(marks, strike=100, gap_return=-1.0, iv_bump_points=10.0)


def test_short_exit_stop_has_priority_before_dte() -> None:
    marks = pd.DataFrame(
        {
            "decision_at": pd.to_datetime(["2026-01-02"], utc=True),
            "remaining_dte": [1.0],
            "close_ask_btc": [0.21],
            "close_fees_btc": [0.0],
        }
    )

    result = evaluate_short_exit(
        marks,
        entry_credit_btc=0.10,
        profit_target=0.50,
        stop_multiple=2.0,
        exit_dte=3.0,
    )

    assert result["exit_trigger"] == "stop_loss"
