"""Evaluate daily BTC realized-volatility forecasts without future leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_pairs.volatility_forecast import (
    attach_dvol,
    build_forecast_panel,
    current_forecast,
    forecast_metrics,
    non_overlapping_forecasts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/market/deribit")
    parser.add_argument("--horizon", action="append", type=int, default=[])
    parser.add_argument("--min-train-days", type=int, default=365)
    parser.add_argument("--rolling-window", type=int, default=30)
    parser.add_argument("--ewma-lambda", type=float, default=0.94)
    parser.add_argument("--garch-refit-days", type=int, default=30)
    parser.add_argument("--output", default="artifacts/btc-volatility-forecast-v1.json")
    arguments = parser.parse_args()
    root = Path(arguments.data_root)
    prices = pd.read_csv(root / "price-bars" / "BTC-PERPETUAL" / "1D.csv.gz")
    dvol = pd.read_csv(root / "volatility-index" / "BTC.csv.gz")

    horizons = arguments.horizon or [14, 30]
    reports = {}
    for horizon in horizons:
        panel = attach_dvol(
            build_forecast_panel(
                prices,
                horizon_days=horizon,
                min_train_days=arguments.min_train_days,
                rolling_window=arguments.rolling_window,
                ewma_lambda=arguments.ewma_lambda,
                garch_refit_days=arguments.garch_refit_days,
            ),
            dvol,
        ).dropna(subset=["dvol"])
        independent = non_overlapping_forecasts(panel)
        current = attach_dvol(
            current_forecast(
                prices,
                horizon_days=horizon,
                rolling_window=arguments.rolling_window,
                ewma_lambda=arguments.ewma_lambda,
            ),
            dvol,
        )
        reports[str(horizon)] = {
            "daily_metrics": forecast_metrics(panel),
            "independent_metrics": forecast_metrics(independent),
            "current": _point(current.iloc[-1]),
            "latest_evaluated": _point(panel.iloc[-1]),
            "daily": [_point(row) for _, row in panel.iterrows()],
        }
    payload = {
        "schema_version": 1,
        "study": "BTC realized-volatility forecast versus DVOL",
        "parameters": {
            "min_train_days": arguments.min_train_days,
            "rolling_window": arguments.rolling_window,
            "ewma_lambda": arguments.ewma_lambda,
            "garch_refit_days": arguments.garch_refit_days,
            "horizons": horizons,
        },
        "horizons": reports,
        "interpretation": "forecast diagnostic; not an executable long/short signal",
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    summaries = {key: value["independent_metrics"] for key, value in reports.items()}
    print(json.dumps(summaries, indent=2))


def _point(row: pd.Series) -> dict[str, object]:
    keys = [
        "forecast_at",
        "target_end",
        "target_rv",
        "dvol",
        "rolling_rv",
        "ewma_rv",
        "garch_rv",
        "dvol_minus_rolling_variance",
        "dvol_minus_ewma_variance",
        "dvol_minus_garch_variance",
    ]
    return {
        key: (str(row[key]) if key in {"forecast_at", "target_end"} else float(row[key]))
        for key in keys
        if key in row.index and not pd.isna(row[key])
    }


if __name__ == "__main__":
    main()
