"""Run the frozen long/short/flat rule on real quarterly option outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_pairs.volatility_regime import build_economic_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", default="artifacts/btc-volatility-forecast-v1.json")
    parser.add_argument(
        "--calibration", default="artifacts/tardis-option-spread-calibration-v1.json"
    )
    parser.add_argument("--carry", default="artifacts/tardis-carry-quarterly-v1.json")
    parser.add_argument(
        "--carry-recovered", default="artifacts/tardis-carry-quarterly-v1-min-size-failures.json"
    )
    parser.add_argument("--contracts", type=float, default=0.1)
    parser.add_argument("--output", default="artifacts/volatility-regime-gate-v1.json")
    arguments = parser.parse_args()

    forecast = _read(arguments.forecast)
    calibration = _read(arguments.calibration)
    carry = [*_read(arguments.carry), *_read(arguments.carry_recovered)]
    gate = build_economic_gate(
        forecast["horizons"]["14"]["daily"],
        calibration["observations"],
        carry,
        contracts=arguments.contracts,
    )
    payload = {
        "schema_version": 1,
        "study": "frozen quarterly volatility-regime economic gate",
        "decision": "promote_monthly" if gate["promote_to_monthly"] else "do_not_promote",
        "gate": gate,
        "limitations": [
            "unhedged hold-to-expiry option P&L is a deliberately small economic gate",
            "short P&L excludes margin, liquidation and intrahorizon path effects",
            "the quarterly sample was not used to fit model parameters or thresholds",
        ],
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                **gate["coverage"],
                "actions": gate["actions"],
                "selected": gate["selected"],
            },
            indent=2,
        )
    )


def _read(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
