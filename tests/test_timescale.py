from decimal import Decimal

import pandas as pd

from quant_pairs.timescale import _base_asset, _int_or_none, _numeric, _text_or_none, _timestamp


def test_timescale_value_helpers_are_sql_safe() -> None:
    assert _base_asset("BTCUSDT") == "BTC"
    assert _base_asset("BTCUSD") == "BTCUSD"
    assert _numeric(1.25) == Decimal("1.25")
    assert _numeric(float("nan")) is None
    assert _timestamp(pd.Timestamp("2026-01-01T00:00:00Z")).tzinfo is not None


def test_option_trade_value_helpers_are_sql_safe() -> None:
    assert _int_or_none(3) == 3
    assert _int_or_none(float("nan")) is None
    assert _text_or_none("liquidated") == "liquidated"
    assert _text_or_none(float("nan")) is None
