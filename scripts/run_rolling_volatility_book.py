"""Run the pre-declared synthetic daily rolling short-volatility book."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_pairs.rolling_volatility_book import (
    RollingBookParameters,
    run_synthetic_rolling_short_book,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="config/experiment.rolling-volatility-book-v1.json")
    parser.add_argument("--forecast", default="artifacts/btc-volatility-forecast-v1.json")
    parser.add_argument(
        "--prices", default="data/market/deribit/price-bars/BTC-PERPETUAL/1D.csv.gz"
    )
    parser.add_argument("--dvol", default="data/market/deribit/volatility-index/BTC.csv.gz")
    parser.add_argument("--output", default="artifacts/rolling-volatility-book-v1.json")
    args = parser.parse_args()

    experiment = json.loads(Path(args.experiment).read_text())
    book = experiment["book"]
    forecasts = pd.DataFrame(json.loads(Path(args.forecast).read_text())["horizons"]["14"]["daily"])
    result = run_synthetic_rolling_short_book(
        pd.read_csv(args.prices),
        pd.read_csv(args.dvol),
        forecasts,
        parameters=RollingBookParameters(
            horizon_days=int(book["horizon_days"]),
            contracts_per_entry=float(book["contracts_per_entry"]),
            max_contracts_per_btc=float(book["max_contracts_per_btc"]),
            bid_iv_discount_points=float(book["bid_iv_discount_points"]),
            funding_rate_hourly=float(book["funding_rate_hourly"]),
            perp_taker_fee_rate=float(book["perp_taker_fee_rate"]),
            initial_equity_btc=float(book["initial_equity_btc"]),
        ),
        start_at=pd.Timestamp(experiment["sample"]["start_at"]),
        end_at=pd.Timestamp(experiment["sample"]["end_at"]),
    )
    payload = {
        "schema_version": 1,
        "study": "daily rolling delta-hedged short-volatility synthetic book",
        "experiment": experiment["experiment_id"],
        **result,
        "limitations": experiment["limitations"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"coverage": payload["coverage"], "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
