"""Daily short/flat backtest over real Deribit tape prints.

For each day with tape coverage: select the ATM ~14 DTE straddle from real
option prints near 12:00 UTC, apply the frozen gate (GARCH 14-day causally
corrected forecast from 08:00 UTC vs mean bid-IV variance), and when the gate
says short, simulate the delta-hedged hold-to-expiry book used across the
pipeline.  Entry credit sells each print minus the Tardis-calibrated
half-spread; prints are real executions but NOT our fills.

Requires QUANT_PAIRS_DATABASE_URL pointing at the tape database (tunnel to the
VM).  Overlapping entries share market paths, so the primary statistics come
from the non-overlapping subsample; the full set is reported with that caveat.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import fmean, stdev

import pandas as pd

from quant_pairs.delta_hedged_carry import simulate_delta_hedged_short_basket
from quant_pairs.settlement import delivery_price_on
from quant_pairs.tape_straddle import select_daily_straddle_prints, short_entry_from_prints
from quant_pairs.tardis_intraday import _option_fee

FUNDING_CACHE_DIR = Path("data/market/deribit/funding")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--decision-time", default="12:00")
    parser.add_argument("--forecast", default="artifacts/btc-volatility-forecast-v1.json")
    parser.add_argument(
        "--calibration", default="artifacts/tardis-option-spread-calibration-v1.json"
    )
    parser.add_argument("--spread-scenario", default="p50")
    parser.add_argument(
        "--prices", default="data/market/deribit/price-bars/BTC-PERPETUAL/1D.csv.gz"
    )
    parser.add_argument("--dvol", default="data/market/deribit/volatility-index/BTC.csv.gz")
    parser.add_argument("--delivery", default="data/market/deribit/delivery_prices/btc_usd.csv")
    parser.add_argument("--contracts", type=float, default=0.1)
    parser.add_argument("--output", default="artifacts/tape-backtest-v1.json")
    args = parser.parse_args()

    database_url = os.environ.get("QUANT_PAIRS_DATABASE_URL")
    if not database_url:
        raise SystemExit("QUANT_PAIRS_DATABASE_URL is not set (open the VM tunnel first)")

    half_spread = float(
        _read_json(args.calibration)["summary"]["relative_half_spread"][args.spread_scenario]
    )
    forecasts = pd.DataFrame(_read_json(args.forecast)["horizons"]["14"]["daily"])
    forecasts["date"] = pd.to_datetime(forecasts["forecast_at"], utc=True).dt.date
    forecast_by_date = forecasts.set_index("date")["garch_corrected_rv"].to_dict()

    prices = pd.read_csv(args.prices)
    dvol = pd.read_csv(args.dvol)
    delivery = pd.read_csv(args.delivery)
    delivery["date"] = pd.to_datetime(delivery["date"], utc=True)
    funding_rate = _mean_funding_rate()
    tape = _load_tape(database_url, start=args.start, end=args.end)

    decisions: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    for day, day_trades in tape.groupby(tape["traded_at"].dt.date, sort=True):
        decision_at = pd.Timestamp(f"{day}T{args.decision_time}:00Z")
        record: dict[str, object] = {"date": str(day)}
        forecast_rv = forecast_by_date.get(day)
        if forecast_rv is None or pd.isna(forecast_rv):
            record["status"] = "no_forecast"
            decisions.append(record)
            continue
        legs = select_daily_straddle_prints(day_trades, decision_at=decision_at)
        if legs.empty:
            record["status"] = "no_pairable_prints"
            decisions.append(record)
            continue
        try:
            entry = short_entry_from_prints(
                legs, relative_half_spread=half_spread, contracts=args.contracts
            )
        except ValueError as error:
            record["status"] = f"iv_inversion_failed: {error}"
            decisions.append(record)
            continue
        signal_short = float(forecast_rv) ** 2 < float(entry["mean_bid_variance"])
        record.update(
            {
                "status": "short" if signal_short else "flat",
                "forecast_rv": float(forecast_rv),
                "mean_bid_iv": float(entry["mean_bid_variance"]) ** 0.5,
                "strike": float(legs["strike"].iloc[0]),
                "dte": float(legs["dte"].iloc[0]),
                "max_print_distance_s": float(legs["seconds_from_decision"].max()),
            }
        )
        decisions.append(record)
        if not signal_short:
            continue

        expiry_at = pd.Timestamp(legs["expiry"].iloc[0])
        try:
            settle = delivery_price_on(delivery, expiry_at)
        except ValueError:
            record["status"] = "no_delivery_price"
            continue
        fees = args.contracts * sum(
            _option_fee(float(leg["bid_price_btc"])) for leg in entry["legs"]
        )
        underlying = float(legs["index_price"].mean())
        result = simulate_delta_hedged_short_basket(
            prices,
            dvol,
            entry_at=decision_at,
            expiry_at=expiry_at,
            legs=entry["legs"],
            contracts=args.contracts,
            entry_underlying=underlying,
            entry_forward=underlying,
            entry_credit_btc=float(entry["entry_credit_btc"]),
            entry_fees_btc=fees,
            delivery_price=settle,
            funding=None,
            funding_rate_hourly=funding_rate,
        )
        result.pop("daily")
        trades.append(
            {
                "date": str(day),
                "expiry_at": str(expiry_at),
                "strike": float(legs["strike"].iloc[0]),
                "dte": float(legs["dte"].iloc[0]),
                "entry_credit_btc": float(entry["entry_credit_btc"]),
                "forecast_rv": float(forecast_rv),
                "mean_bid_iv": float(entry["mean_bid_variance"]) ** 0.5,
                **result,
            }
        )

    status_counts = pd.Series([d["status"] for d in decisions]).value_counts().to_dict()
    pnl_all = [float(t["hedged_pnl_btc"]) for t in trades]
    non_overlap = _non_overlapping(trades)
    pnl_no = [float(t["hedged_pnl_btc"]) for t in non_overlap]
    payload = {
        "schema_version": 1,
        "study": "daily short/flat backtest on real Deribit tape prints (2025)",
        "spread_scenario": args.spread_scenario,
        "relative_half_spread": half_spread,
        "contracts": args.contracts,
        "funding_rate_hourly": funding_rate,
        "decision_counts": status_counts,
        "short_trades": {
            "all_overlapping": _stats(pnl_all),
            "non_overlapping": _stats(pnl_no),
        },
        "limitations": [
            "prints are real executions by third parties, not our fills; queue and size unknown",
            "half-spread from the quarterly Tardis calibration applied to print prices",
            "index price used as forward proxy at entry (basis unobservable in the tape)",
            "daily marks remain synthetic (Black-76 + DVOL anchor); funding constant",
            "overlapping trades share paths: inference belongs to the non-overlapping set",
        ],
        "decisions": decisions,
        "trades": trades,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "decision_counts": status_counts,
                "short_trades_all": payload["short_trades"]["all_overlapping"],
                "short_trades_non_overlapping": payload["short_trades"]["non_overlapping"],
            },
            indent=2,
        )
    )


def _load_tape(database_url: str, *, start: str, end: str) -> pd.DataFrame:
    import psycopg

    query = """
        select instrument_name, traded_at, price, iv, index_price
        from market.option_trade
        where currency = 'BTC' and iv is not null
          and traded_at >= %s and traded_at < (%s::date + interval '1 day')
    """
    with psycopg.connect(database_url) as connection:
        frame = pd.read_sql(query, connection, params=(start, end))
    frame["traded_at"] = pd.to_datetime(frame["traded_at"], utc=True)
    return frame


def _non_overlapping(trades: list[dict[str, object]]) -> list[dict[str, object]]:
    chosen: list[dict[str, object]] = []
    last_expiry: pd.Timestamp | None = None
    for trade in trades:
        entry = pd.Timestamp(trade["date"], tz="UTC")
        if last_expiry is None or entry >= last_expiry:
            chosen.append(trade)
            last_expiry = pd.Timestamp(trade["expiry_at"])
    return chosen


def _stats(pnl: list[float]) -> dict[str, object]:
    if not pnl:
        return {"n": 0}
    mean = fmean(pnl)
    std = stdev(pnl) if len(pnl) >= 2 else 0.0
    return {
        "n": len(pnl),
        "total_btc": sum(pnl),
        "mean_btc": mean,
        "std_btc": std,
        "t_stat": mean / (std / len(pnl) ** 0.5) if std > 0 else None,
        "positive": sum(1 for value in pnl if value > 0),
        "min_btc": min(pnl),
        "max_btc": max(pnl),
    }


def _mean_funding_rate() -> float:
    files = sorted(FUNDING_CACHE_DIR.glob("BTC-PERPETUAL-hedge-*.csv"))
    if not files:
        raise SystemExit("no cached funding windows; run scripts/run_delta_hedged_carry.py first")
    import numpy as np

    return float(np.mean([pd.read_csv(path)["interest_1h"].mean() for path in files]))


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    main()
