"""Build the compact JSON consumed by the volatility research dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_pairs.volatility_report import build_volatility_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/market/deribit")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--output", default="artifacts/volatility-research-v1.json")
    arguments = parser.parse_args()
    data_root = Path(arguments.data_root)
    artifacts_root = Path(arguments.artifacts_root)
    report = build_volatility_report(
        pd.read_csv(data_root / "volatility-index" / "BTC.csv.gz"),
        pd.read_csv(data_root / "price-bars" / "BTC-PERPETUAL" / "1D.csv.gz"),
        json.loads((artifacts_root / "tardis-carry-quarterly-v1.json").read_text()),
        json.loads(
            (artifacts_root / "tardis-carry-quarterly-v1-min-size-failures.json").read_text()
        ),
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"path={output} calibration={len(report['calibration']['independent'])} "
          f"carry={len(report['carry']['points'])}")


if __name__ == "__main__":
    main()
