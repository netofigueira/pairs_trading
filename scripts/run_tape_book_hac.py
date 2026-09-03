"""HAC inference on the aggregated daily P&L of the tape-driven short book.

Re-runs the frozen-gate tape backtest keeping each trade's daily P&L steps,
aggregates them into one calendar series for the whole book (overlapping
positions simply add), and tests the mean daily P&L with Newey-West standard
errors.  This uses every entry without the waste of non-overlapping
subsampling; the price is trusting the HAC correction for the ~14-day overlap,
so t-stats are reported for several lag choices.

Flat days inside the sample count as zero P&L: the test is on the strategy's
unconditional daily mean, which is what capital experiences.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from quant_pairs.delta_hedged_carry import simulate_delta_hedged_short_basket
from quant_pairs.settlement import delivery_price_on
from quant_pairs.tape_straddle import select_daily_straddle_prints, short_entry_from_prints
from quant_pairs.tardis_intraday import _option_fee

FUNDING_CACHE_DIR = Path("data/market/deribit/funding")


def newey_west_tstat(series: np.ndarray, *, lags: int) -> dict[str, float]:
    """t-stat of the mean under Bartlett-kernel HAC standard errors."""

    values = np.asarray(series, dtype=float)
    n = values.size
    if n <= lags + 1:
        raise ValueError("series too short for the requested lags")
    demeaned = values - values.mean()
    variance = float(demeaned @ demeaned) / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cov = float(demeaned[lag:] @ demeaned[:-lag]) / n
        variance += 2.0 * weight * cov
    se = (variance / n) ** 0.5
    return {
        "lags": lags,
        "mean": float(values.mean()),
        "hac_se": se,
        "t_stat": float(values.mean() / se) if se > 0 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-04-01")
    parser.add_argument("--end", default="2026-08-18")
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
    parser.add_argument("--output", default="artifacts/tape-book-hac-v1.json")
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

    daily_pnl: dict[pd.Timestamp, float] = {}
    open_by_day: dict[pd.Timestamp, int] = {}
    n_trades = 0
    for day, day_trades in tape.groupby(tape["traded_at"].dt.date, sort=True):
        decision_at = pd.Timestamp(f"{day}T{args.decision_time}:00Z")
        forecast_rv = forecast_by_date.get(day)
        if forecast_rv is None or pd.isna(forecast_rv):
            continue
        legs = select_daily_straddle_prints(day_trades, decision_at=decision_at)
        if legs.empty:
            continue
        try:
            entry = short_entry_from_prints(
                legs, relative_half_spread=half_spread, contracts=args.contracts
            )
        except ValueError:
            continue
        if float(forecast_rv) ** 2 >= float(entry["mean_bid_variance"]):
            continue
        expiry_at = pd.Timestamp(legs["expiry"].iloc[0])
        try:
            settle = delivery_price_on(delivery, expiry_at)
        except ValueError:
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
        n_trades += 1
        rows = result["daily"]
        # Daily steps: option mid change plus that segment's hedge/funding/fees;
        # the final step closes any gap so the steps sum exactly to the total.
        steps: list[tuple[pd.Timestamp, float]] = []
        running = 0.0
        for previous, current in zip(rows, rows[1:]):
            step = (
                float(current["short_straddle_mid_btc"])
                - float(previous["short_straddle_mid_btc"])
                + float(previous["segment_hedge_pnl_btc"])
                + float(previous["segment_funding_btc"])
                - float(current["hedge_fee_btc"])
            )
            steps.append((pd.Timestamp(current["at"]).normalize(), step))
            running += step
        settle_day = expiry_at.normalize()
        steps.append((settle_day, float(result["hedged_pnl_btc"]) - running))
        for at, step in steps:
            daily_pnl[at] = daily_pnl.get(at, 0.0) + step
        cursor = decision_at.normalize()
        while cursor <= settle_day:
            open_by_day[cursor] = open_by_day.get(cursor, 0) + 1
            cursor += pd.Timedelta(days=1)

    if not daily_pnl:
        raise SystemExit("no short trades in the sample")
    calendar = pd.date_range(min(daily_pnl), max(daily_pnl), freq="D", tz="UTC")
    series = pd.Series(0.0, index=calendar)
    for at, value in daily_pnl.items():
        series.loc[at] += value
    hac = [newey_west_tstat(series.to_numpy(), lags=lags) for lags in (14, 21, 28)]

    payload = {
        "schema_version": 1,
        "study": "HAC inference on the aggregated daily P&L of the tape-driven short book",
        "spread_scenario": args.spread_scenario,
        "contracts_per_entry": args.contracts,
        "n_trades": n_trades,
        "n_days": int(series.size),
        "days_with_position": int(sum(1 for v in open_by_day.values() if v > 0)),
        "max_concurrent_positions": int(max(open_by_day.values())),
        "total_pnl_btc": float(series.sum()),
        "mean_daily_pnl_btc": float(series.mean()),
        "std_daily_pnl_btc": float(series.std(ddof=1)),
        "newey_west": hac,
        "limitations": [
            "same synthetic daily marks and print-entry approximations as the tape backtest",
            "book has no position cap here: every gate-short entry is taken at 0.1 contract",
            "flat days count as zero: the test is the unconditional daily mean",
            "HAC corrects overlap dependence only up to the chosen lag",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "n_trades",
                    "n_days",
                    "max_concurrent_positions",
                    "total_pnl_btc",
                    "mean_daily_pnl_btc",
                    "newey_west",
                )
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


def _mean_funding_rate() -> float:
    files = sorted(FUNDING_CACHE_DIR.glob("BTC-PERPETUAL-hedge-*.csv"))
    if not files:
        raise SystemExit("no cached funding windows; run scripts/run_delta_hedged_carry.py first")
    return float(np.mean([pd.read_csv(path)["interest_1h"].mean() for path in files]))


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    main()
