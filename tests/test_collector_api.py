import pandas as pd
import pytest

from quant_pairs.collector_api import _incremental_tape_start


class _FakeStore:
    def __init__(self, latest: pd.Timestamp | None) -> None:
        self._latest = latest

    def latest_option_trade_time(self, currency: str) -> pd.Timestamp | None:
        return self._latest


def test_incremental_tape_start_uses_lookback_when_no_prior_trades() -> None:
    now = pd.Timestamp("2026-09-03T12:00:00Z")
    store = _FakeStore(latest=None)

    start = _incremental_tape_start(store, "BTC", now, initial_lookback_hours=24)

    assert start == now - pd.Timedelta(hours=24)


def test_incremental_tape_start_overlaps_five_minutes_before_the_cursor() -> None:
    now = pd.Timestamp("2026-09-03T12:00:00Z")
    latest = pd.Timestamp("2026-09-03T11:50:00Z")
    store = _FakeStore(latest=latest)

    start = _incremental_tape_start(store, "BTC", now, initial_lookback_hours=24)

    assert start == latest - pd.Timedelta(minutes=5)


@pytest.mark.parametrize("token", ["", "wrong-token"])
def test_authorize_rejects_missing_or_wrong_token(monkeypatch, token: str) -> None:
    from fastapi import HTTPException

    from quant_pairs.collector_api import _authorize

    monkeypatch.setenv("QUANT_COLLECTOR_TOKEN", "expected-token")

    with pytest.raises(HTTPException) as error:
        _authorize(token)
    assert error.value.status_code == 401


def test_authorize_accepts_matching_token(monkeypatch) -> None:
    from quant_pairs.collector_api import _authorize

    monkeypatch.setenv("QUANT_COLLECTOR_TOKEN", "expected-token")

    _authorize("expected-token")
