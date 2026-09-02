"""Official Deribit delivery prices and inverse-option settlement accounting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pairs.deribit import DeribitAPIError, Transport, _http_transport, _result

DELIVERY_PRICES_ENDPOINT = "/public/get_delivery_prices"
SETTLEMENT_FEE_BTC_PER_CONTRACT = 0.00015
SETTLEMENT_FEE_PAYOFF_CAP = 0.125
_PAGE_SIZE = 100


def fetch_delivery_prices(
    index_name: str = "btc_usd", *, transport: Transport | None = None
) -> pd.DataFrame:
    """Return the full official daily delivery-price history for an index."""

    transport = transport or _http_transport
    offset = 0
    rows: list[dict[str, object]] = []
    while True:
        payload = transport(
            DELIVERY_PRICES_ENDPOINT,
            {"index_name": index_name, "offset": offset, "count": _PAGE_SIZE},
        )
        result = _result(payload, DELIVERY_PRICES_ENDPOINT)
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            raise DeribitAPIError("delivery prices result is missing data")
        page = result["data"]
        rows.extend(page)
        total = int(result.get("records_total", len(rows)))
        offset += len(page)
        if not page or offset >= total:
            break
    frame = pd.DataFrame(rows, columns=["date", "delivery_price"])
    if frame.empty:
        raise DeribitAPIError(f"no delivery prices returned for {index_name}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["delivery_price"] = pd.to_numeric(frame["delivery_price"], errors="raise")
    return (
        frame.drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_delivery_prices(
    index_name: str = "btc_usd",
    *,
    cache_path: Path | str,
    required_date: pd.Timestamp | None = None,
    transport: Transport | None = None,
) -> pd.DataFrame:
    """Fetch delivery prices through a local CSV cache, refreshing when it lags."""

    path = Path(cache_path)
    if path.exists():
        cached = pd.read_csv(path, parse_dates=["date"])
        cached["date"] = pd.to_datetime(cached["date"], utc=True)
        if required_date is None or (
            not cached.empty and cached["date"].iloc[-1] >= _utc_date(required_date)
        ):
            return cached
    frame = fetch_delivery_prices(index_name, transport=transport)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def delivery_price_on(prices: pd.DataFrame, date: pd.Timestamp) -> float:
    """Return the official delivery price for one UTC calendar date."""

    target = _utc_date(date)
    match = prices.loc[prices["date"] == target, "delivery_price"]
    if len(match) != 1:
        raise ValueError(f"no unique delivery price for {target.date()}")
    return float(match.iloc[0])


def settlement_payoff_btc(option_type: str, strike: float, delivery_price: float) -> float:
    """Cash-settled payoff of one inverse option contract, in BTC."""

    if strike <= 0 or delivery_price <= 0:
        raise ValueError("strike and delivery price must be positive")
    if option_type == "call":
        return max(0.0, delivery_price - strike) / delivery_price
    if option_type == "put":
        return max(0.0, strike - delivery_price) / delivery_price
    raise ValueError(f"unknown option type: {option_type}")


def settlement_fee_btc(payoff_btc: float) -> float:
    """Deribit delivery fee: 0.015% of underlying, capped at 12.5% of option value.

    Options expiring worthless are not charged.
    """

    if payoff_btc < 0:
        raise ValueError("payoff cannot be negative")
    if payoff_btc == 0:
        return 0.0
    return min(SETTLEMENT_FEE_BTC_PER_CONTRACT, SETTLEMENT_FEE_PAYOFF_CAP * payoff_btc)


def _utc_date(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").normalize()
