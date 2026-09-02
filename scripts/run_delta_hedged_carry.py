"""Delta-hedged short straddle envelope over the quarterly Tardis entries.

Phase 1 of the volatility pipeline plan (docs/2026-09-02-plano-pipeline-vol.md):
re-run the same quarterly short straddles, held to expiry, but delta-hedged
daily with the inverse perp over synthetic Black-76 marks.  Answers whether the
variance premium survives once directional path dependence is removed.  Funding
is real hourly history (cached locally on first download).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, stdev

import pandas as pd

from quant_pairs.delta_hedged_carry import simulate_delta_hedged_short
from quant_pairs.funding import load_funding_history
from quant_pairs.synthetic_option_backfill import build_daily_straddle_marks
from quant_pairs.tardis_intraday import _option_fee

FUNDING_CACHE_DIR = Path("data/market/deribit/funding")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration", default="artifacts/tardis-option-spread-calibration-v1.json"
    )
    parser.add_argument("--carry", default="artifacts/tardis-carry-quarterly-v1.json")
    parser.add_argument(
        "--carry-recovered",
        default="artifacts/tardis-carry-quarterly-v1-min-size-failures.json",
    )
    parser.add_argument(
        "--prices", default="data/market/deribit/price-bars/BTC-PERPETUAL/1D.csv.gz"
    )
    parser.add_argument("--dvol", default="data/market/deribit/volatility-index/BTC.csv.gz")
    parser.add_argument("--contracts", type=float, default=0.1)
    parser.add_argument("--output", default="artifacts/delta-hedged-carry-v1.json")
    args = parser.parse_args()
    if args.contracts <= 0:
        parser.error("contracts must be positive")

    observations = pd.DataFrame(_read(args.calibration)["observations"])
    settled = {
        str(entry["entry_at"])[:10]: entry
        for entry in [*_read(args.carry), *_read(args.carry_recovered)]
        if entry.get("status") == "carry_unhedged_settled"
    }
    prices = pd.read_csv(args.prices)
    dvol = pd.read_csv(args.dvol)
    dvol_start = pd.to_datetime(dvol["timestamp"], utc=True, format="mixed").min()

    trades: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for date, legs in observations.groupby("date", sort=True):
        carry = settled.get(str(date))
        if carry is None:
            skipped.append({"date": str(date), "reason": "no settled carry entry"})
            continue
        if len(legs) != 2 or set(legs["option_type"]) != {"call", "put"}:
            skipped.append({"date": str(date), "reason": "incomplete straddle legs"})
            continue
        if legs[["bid_amount", "ask_amount"]].min().min() < args.contracts:
            skipped.append({"date": str(date), "reason": "top-of-book too small"})
            continue
        entry_at = pd.Timestamp(legs["entry_at"].iloc[0])
        if entry_at < dvol_start:
            skipped.append({"date": str(date), "reason": "before DVOL history"})
            continue

        expiry_at = pd.Timestamp(carry["expiry_at"])
        strike = float(legs["strike_usd"].iloc[0])
        entry_underlying = float(legs["underlying_perp_mid_usd"].iloc[0])
        entry_forward = float(legs["parity_forward_usd"].iloc[0])
        entry_iv = float(legs["mid_iv"].mean())
        credit = float(legs["bid_btc"].sum()) * args.contracts
        fees = args.contracts * sum(_option_fee(float(p)) for p in legs["bid_btc"])

        marks = build_daily_straddle_marks(
            prices,
            dvol,
            entry_at=entry_at,
            expiry_at=expiry_at,
            strike=strike,
            entry_underlying=entry_underlying,
            entry_forward=entry_forward,
            entry_iv=entry_iv,
            relative_half_spread=0.0,
            contracts=args.contracts,
        )
        funding = load_funding_history(
            "BTC-PERPETUAL",
            start=entry_at,
            end=expiry_at,
            cache_path=FUNDING_CACHE_DIR / f"BTC-PERPETUAL-hedge-{date}.csv",
        )
        result = simulate_delta_hedged_short(
            marks,
            entry_at=entry_at,
            expiry_at=expiry_at,
            strike=strike,
            contracts=args.contracts,
            entry_underlying=entry_underlying,
            entry_forward=entry_forward,
            entry_iv=entry_iv,
            entry_credit_btc=credit,
            entry_fees_btc=fees,
            delivery_price=float(carry["delivery_price_usd"]),
            funding=funding,
        )
        daily = result.pop("daily")
        unhedged_steps, hedged_steps = _daily_steps(daily)
        trades.append(
            {
                "date": str(date),
                "entry_at": str(entry_at),
                "expiry_at": str(expiry_at),
                "strike_usd": strike,
                "entry_credit_btc": credit,
                "entry_iv": entry_iv,
                **result,
                "daily_pnl_std_unhedged_btc": _std(unhedged_steps),
                "daily_pnl_std_hedged_btc": _std(hedged_steps),
                "daily": daily,
            }
        )

    if not trades:
        raise SystemExit("no eligible trades")
    unhedged = [float(t["unhedged_pnl_btc"]) for t in trades]
    hedged = [float(t["hedged_pnl_btc"]) for t in trades]
    payload = {
        "schema_version": 1,
        "study": "daily delta-hedged short straddle envelope (synthetic marks)",
        "contracts": args.contracts,
        "n_trades": len(trades),
        "skipped": skipped,
        "summary": {
            "unhedged": _summary(unhedged),
            "hedged": _summary(hedged),
            "mean_daily_std_unhedged_btc": fmean(
                float(t["daily_pnl_std_unhedged_btc"]) for t in trades
            ),
            "mean_daily_std_hedged_btc": fmean(
                float(t["daily_pnl_std_hedged_btc"]) for t in trades
            ),
        },
        "limitations": [
            "synthetic_model marks: IV path anchored to DVOL changes, constant entry basis",
            "hedge fills at daily closes with taker fees; no intraday rebalancing or slippage",
            "no margin, liquidation or portfolio effects; envelope, not observed fills",
        ],
        "trades": trades,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "n_trades": len(trades),
                "skipped": len(skipped),
                "summary": payload["summary"],
            },
            indent=2,
        )
    )


def _daily_steps(daily: list[dict[str, object]]) -> tuple[list[float], list[float]]:
    """Mid-marked daily P&L steps for the short option leg and the hedged book."""

    unhedged: list[float] = []
    hedged: list[float] = []
    for previous, current in zip(daily, daily[1:]):
        option_step = float(current["short_straddle_mid_btc"]) - float(
            previous["short_straddle_mid_btc"]
        )
        hedge_step = (
            float(previous["segment_hedge_pnl_btc"])
            + float(previous["segment_funding_btc"])
            - float(current["hedge_fee_btc"])
        )
        unhedged.append(option_step)
        hedged.append(option_step + hedge_step)
    return unhedged, hedged


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "total_btc": sum(values),
        "mean_btc": fmean(values),
        "std_btc": _std(values),
        "min_btc": ordered[0],
        "max_btc": ordered[-1],
        "positive": sum(1 for value in values if value > 0),
    }


def _read(path: str) -> dict | list:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    main()
