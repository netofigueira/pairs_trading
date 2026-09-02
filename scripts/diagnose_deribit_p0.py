"""Report the public Deribit DVOL-versus-future-RV calibration diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_pairs.dvol import build_iv_rv_panel, non_overlapping


def _summary(panel: pd.DataFrame) -> dict[str, float | int | None]:
    if panel.empty:
        return {
            "observations": 0,
            "mean_iv": None,
            "mean_forward_rv": None,
            "mean_iv_minus_rv": None,
            "iv_above_rv_share": None,
            "iv_rv_correlation": None,
        }
    return {
        "observations": len(panel),
        "mean_iv": float(panel["iv"].mean()),
        "mean_forward_rv": float(panel["forward_rv"].mean()),
        "mean_iv_minus_rv": float(panel["iv_minus_rv"].mean()),
        "iv_above_rv_share": float((panel["iv_minus_rv"] > 0).mean()),
        "iv_rv_correlation": float(panel["iv"].corr(panel["forward_rv"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--horizon-days", type=int, default=30)
    arguments = parser.parse_args()
    root = Path(arguments.data_root) / "market" / "deribit"
    dvol = pd.read_csv(root / "volatility-index" / "BTC.csv.gz")
    prices = pd.read_csv(root / "price-bars" / "BTC-PERPETUAL" / "1D.csv.gz")
    panel = build_iv_rv_panel(dvol, prices, horizon_days=arguments.horizon_days)
    independent = non_overlapping(panel, horizon_days=arguments.horizon_days)
    print(
        json.dumps(
            {
                "study": "DVOL at t versus BTC-PERPETUAL realized volatility after t",
                "horizon_days": arguments.horizon_days,
                "overlapping_outcomes": _summary(panel),
                "non_overlapping_outcomes": _summary(independent),
                "note": (
                    "Calibration only; this is not an executable options P&L "
                    "or proof of tradable edge."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
