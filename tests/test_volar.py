import pandas as pd
import pytest

from quant_pairs.volar import VolarDataError, load_executable_chain


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp": pd.Timestamp("2026-07-14T12:00:00Z"),
        "underlying": "ETH",
        "instrument": "ETH-25JUL26-3000-C",
        "strike": 3_000.0,
        "expiry": pd.Timestamp("2026-07-25T08:00:00Z"),
        "type": "call",
        "mark_iv": 0.65,
        "bid_price": 0.02,
        "ask_price": 0.022,
        "underlying_price": 3_000.0,
        "delta": 0.5,
        "source": "live_ws",
    }
    row.update(overrides)
    return row


def test_loader_keeps_only_live_executable_unexpired_quotes(tmp_path) -> None:
    rows = [
        _row(instrument="KEEP"),
        _row(instrument="NO_BID", bid_price=0.0),
        _row(instrument="CROSSED", bid_price=0.03),
        _row(instrument="MODELED", source="modeled_surface"),
        _row(instrument="EXPIRED", expiry=pd.Timestamp("2026-07-14T11:00:00Z")),
    ]
    path = tmp_path / "chain.parquet"
    pd.DataFrame(rows).to_parquet(path)

    result = load_executable_chain(path)

    assert result.source_rows == 5
    assert result.executable_rows == 1
    assert result.rejected_rows == 4
    assert result.quotes["instrument"].tolist() == ["KEEP"]


def test_loader_rejects_unknown_schema(tmp_path) -> None:
    path = tmp_path / "invalid.parquet"
    pd.DataFrame({"timestamp": [pd.Timestamp("2026-07-14T12:00:00Z")]}).to_parquet(path)

    with pytest.raises(VolarDataError, match="missing columns"):
        load_executable_chain(path)
