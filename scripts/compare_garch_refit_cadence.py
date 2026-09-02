"""Compare daily and periodic expanding-window GARCH refits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_pairs.volatility_forecast import (
    add_expanding_bias_correction,
    build_forecast_panel,
    compare_garch_refit_panels,
    non_overlapping_forecasts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/market/deribit")
    parser.add_argument("--horizon", action="append", type=int, default=[])
    parser.add_argument("--min-train-days", type=int, default=365)
    parser.add_argument("--rolling-window", type=int, default=30)
    parser.add_argument("--ewma-lambda", type=float, default=0.94)
    parser.add_argument("--challenger-refit-days", type=int, default=1)
    parser.add_argument("--incumbent-refit-days", type=int, default=30)
    parser.add_argument("--selection-horizon", type=int, default=14)
    parser.add_argument("--data-cutoff", default="2026-09-03T00:00:00-03:00")
    parser.add_argument("--output", default="artifacts/garch-refit-cadence-v1.json")
    arguments = parser.parse_args()
    prices = pd.read_csv(Path(arguments.data_root) / "price-bars" / "BTC-PERPETUAL" / "1D.csv.gz")
    available_at = pd.to_datetime(prices["timestamp"], utc=True, format="mixed") + pd.Timedelta(
        days=1
    )
    cutoff = pd.Timestamp(arguments.data_cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("data cutoff must be timezone-aware")
    prices = prices.loc[available_at < cutoff].copy()

    horizons = arguments.horizon or [1, 14]
    if arguments.selection_horizon not in horizons:
        raise ValueError("selection horizon must be included in evaluated horizons")
    reports: dict[str, object] = {}
    for horizon in horizons:
        panels = {}
        for name, cadence in (
            ("daily_refit", arguments.challenger_refit_days),
            ("monthly_refit", arguments.incumbent_refit_days),
        ):
            raw = build_forecast_panel(
                prices,
                horizon_days=horizon,
                min_train_days=arguments.min_train_days,
                rolling_window=arguments.rolling_window,
                ewma_lambda=arguments.ewma_lambda,
                garch_refit_days=cadence,
            )
            corrected = add_expanding_bias_correction(raw)
            panels[name] = corrected if horizon == 1 else non_overlapping_forecasts(corrected)
        reports[str(horizon)] = compare_garch_refit_panels(
            panels["daily_refit"], panels["monthly_refit"]
        )

    economic = reports[str(arguments.selection_horizon)]
    metrics = economic["metrics"]
    dm = economic["diebold_mariano"]
    checks = {
        "lower_qlike": metrics["daily_refit"]["qlike"] < metrics["monthly_refit"]["qlike"],
        "no_worse_mse": metrics["daily_refit"]["mse_variance"]
        <= metrics["monthly_refit"]["mse_variance"],
        "dm_ci_below_zero": dm["ci_high"] < 0,
    }
    selected = "daily_refit" if all(checks.values()) else "monthly_refit"
    payload = {
        "schema_version": 1,
        "study": "expanding-window GARCH refit-cadence comparison",
        "parameters": {
            "minimum_training_days": arguments.min_train_days,
            "bias_correction_minimum_completed": 30,
            "challenger_refit_days": arguments.challenger_refit_days,
            "incumbent_refit_days": arguments.incumbent_refit_days,
            "horizons_days": horizons,
            "selection_horizon_days": arguments.selection_horizon,
            "horizon_1_role": "one-step-ahead diagnostic",
            "horizon_14_role": "economic-horizon selection",
        },
        "horizons": reports,
        "promotion_checks": checks,
        "selected_refit_cadence": selected,
        "holdout_policy": f"only data available before {cutoff.isoformat()} were used",
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "selected_refit_cadence": selected,
                "promotion_checks": checks,
                "horizons": reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
