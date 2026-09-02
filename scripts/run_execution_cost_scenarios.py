"""Execution-cost scenarios for the delta-hedged short volatility book.

Phase 2 of the volatility pipeline plan (docs/2026-09-02-plano-pipeline-vol.md):
same quarterly entries, hold to expiry, daily delta hedge, but four fill
policies at entry:

- ``atm_cross``: ATM straddle sold at displayed bids (the Phase 1 baseline);
- ``atm_post_mid``: ATM straddle assumed filled at mid (maker envelope; fill
  probability is NOT modeled and the result is an upper bound);
- ``strangle25_cross``: 25-delta strangle sold at displayed bids;
- ``strangle25_post_mid``: 25-delta strangle at mid (same caveat).

Strangle strikes come from the real Tardis book at each entry, same expiry as
the ATM selection, mid IVs inverted against the ATM parity forward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, stdev

import pandas as pd

from quant_pairs.delta_hedged_carry import simulate_delta_hedged_short_basket
from quant_pairs.funding import load_funding_history
from quant_pairs.tardis_intraday import _option_fee
from quant_pairs.tardis_options import select_strangle_by_delta
from quant_pairs.tardis_quotes import reconstruct_top_of_book

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
    parser.add_argument("--quotes-root", default="data/market/tardis/deribit/quotes")
    parser.add_argument(
        "--prices", default="data/market/deribit/price-bars/BTC-PERPETUAL/1D.csv.gz"
    )
    parser.add_argument("--dvol", default="data/market/deribit/volatility-index/BTC.csv.gz")
    parser.add_argument("--contracts", type=float, default=0.1)
    parser.add_argument("--target-delta", type=float, default=0.25)
    parser.add_argument("--max-age-seconds", type=int, default=300)
    parser.add_argument("--output", default="artifacts/execution-cost-scenarios-v1.json")
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
        if carry is None or len(legs) != 2 or set(legs["option_type"]) != {"call", "put"}:
            skipped.append({"date": str(date), "reason": "no settled ATM straddle"})
            continue
        if legs[["bid_amount", "ask_amount"]].min().min() < args.contracts:
            skipped.append({"date": str(date), "reason": "top-of-book too small"})
            continue
        entry_at = pd.Timestamp(legs["entry_at"].iloc[0])
        if entry_at < dvol_start:
            skipped.append({"date": str(date), "reason": "before DVOL history"})
            continue
        expiry_at = pd.Timestamp(carry["expiry_at"])
        forward = float(legs["parity_forward_usd"].iloc[0])
        underlying = float(legs["underlying_perp_mid_usd"].iloc[0])
        delivery_price = float(carry["delivery_price_usd"])
        funding = load_funding_history(
            "BTC-PERPETUAL",
            start=entry_at,
            end=expiry_at,
            cache_path=FUNDING_CACHE_DIR / f"BTC-PERPETUAL-hedge-{date}.csv",
        )

        atm_legs = [
            {
                "type": str(row.option_type),
                "strike": float(row.strike_usd),
                "entry_iv": float(row.mid_iv),
                "bid_btc": float(row.bid_btc),
                "mid_btc": float(row.mid_btc),
            }
            for row in legs.itertuples(index=False)
        ]
        structures = {"atm": atm_legs}
        try:
            book = reconstruct_top_of_book(
                Path(args.quotes_root) / str(date) / "OPTIONS.csv.gz",
                as_of=entry_at,
                max_age=pd.Timedelta(seconds=args.max_age_seconds),
            )
            strangle = select_strangle_by_delta(
                book.loc[book["symbol"].str.startswith("BTC-")],
                forward=forward,
                as_of=entry_at,
                expiry=expiry_at,
                target_delta=args.target_delta,
                min_contracts=args.contracts,
            )
        except (FileNotFoundError, ValueError) as error:
            strangle = pd.DataFrame()
            skipped.append({"date": str(date), "reason": f"strangle: {error}"})
        if len(strangle) == 2:
            structures["strangle25"] = [
                {
                    "type": str(row.type),
                    "strike": float(row.strike),
                    "entry_iv": float(row.mid_iv),
                    "bid_btc": float(row.bid_btc),
                    "mid_btc": float(row.mid_btc),
                    "symbol": str(row.symbol),
                    "forward_delta": float(row.forward_delta),
                }
                for row in strangle.itertuples(index=False)
            ]
        elif "strangle25" not in structures and not any(
            item["date"] == str(date) and item["reason"].startswith("strangle") for item in skipped
        ):
            skipped.append({"date": str(date), "reason": "strangle: no paired candidates"})

        trade: dict[str, object] = {
            "date": str(date),
            "entry_at": str(entry_at),
            "expiry_at": str(expiry_at),
            "scenarios": {},
        }
        for structure_name, structure_legs in structures.items():
            for fill_name, fill_key in (("cross", "bid_btc"), ("post_mid", "mid_btc")):
                credit = sum(float(leg[fill_key]) for leg in structure_legs) * args.contracts
                fees = args.contracts * sum(
                    _option_fee(float(leg[fill_key])) for leg in structure_legs
                )
                result = simulate_delta_hedged_short_basket(
                    prices,
                    dvol,
                    entry_at=entry_at,
                    expiry_at=expiry_at,
                    legs=structure_legs,
                    contracts=args.contracts,
                    entry_underlying=underlying,
                    entry_forward=forward,
                    entry_credit_btc=credit,
                    entry_fees_btc=fees,
                    delivery_price=delivery_price,
                    funding=funding,
                )
                result.pop("daily")
                trade["scenarios"][f"{structure_name}_{fill_name}"] = {
                    "entry_credit_btc": credit,
                    "legs": structure_legs,
                    **result,
                }
        trades.append(trade)

    if not trades:
        raise SystemExit("no eligible trades")
    scenario_names = sorted({name for t in trades for name in t["scenarios"]})
    summary = {}
    baseline = _collect(trades, "atm_cross")
    for name in scenario_names:
        values = _collect(trades, name)
        paired_dates = sorted(set(values) & set(baseline))
        diffs = [values[d] - baseline[d] for d in paired_dates]
        summary[name] = {
            "n": len(values),
            **_stats(list(values.values())),
            "paired_vs_atm_cross": {
                "n": len(diffs),
                "mean_diff_btc": fmean(diffs) if diffs else None,
                "std_diff_btc": stdev(diffs) if len(diffs) >= 2 else None,
            },
        }
    payload = {
        "schema_version": 1,
        "study": "execution-cost scenarios for the delta-hedged short volatility book",
        "contracts": args.contracts,
        "target_delta": args.target_delta,
        "skipped": skipped,
        "summary": summary,
        "limitations": [
            "post_mid assumes a full maker fill at mid: an upper bound, fill probability unmodeled",
            "maker and taker option fees assumed equal (Deribit cap formula)",
            "strangle mid IVs inverted against the ATM parity forward of the same expiry",
            "synthetic_model daily marks; no margin, liquidation or portfolio effects",
        ],
        "trades": trades,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"n_trades": len(trades), "summary": summary}, indent=2))


def _collect(trades: list[dict[str, object]], scenario: str) -> dict[str, float]:
    return {
        str(t["date"]): float(t["scenarios"][scenario]["hedged_pnl_btc"])
        for t in trades
        if scenario in t["scenarios"]
    }


def _stats(values: list[float]) -> dict[str, object]:
    mean = fmean(values)
    std = stdev(values) if len(values) >= 2 else 0.0
    return {
        "total_btc": sum(values),
        "mean_btc": mean,
        "std_btc": std,
        "t_stat": mean / (std / len(values) ** 0.5) if std > 0 else None,
        "positive": sum(1 for value in values if value > 0),
        "min_btc": min(values),
        "max_btc": max(values),
    }


def _read(path: str) -> dict | list:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    main()
