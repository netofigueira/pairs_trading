"""Calibrate option spread scenarios from observed quarterly Tardis books.

The script selects the same ATM straddles as the carry pilot, measures their
observed bid/ask spreads, and inverts each leg to bid/mid/ask IV with the
Deribit inverse Black-76 model.  BTC-PERPETUAL mid is an explicit first-pass
proxy for the expiry forward; inversion failures remain visible in the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_pairs.inverse_options import implied_volatility, inverse_forward_from_parity
from quant_pairs.tardis_intraday import _mid, _perp_book
from quant_pairs.tardis_options import select_atm_straddle
from quant_pairs.tardis_quotes import reconstruct_top_of_book


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/market/tardis/deribit/quotes")
    parser.add_argument("--date", action="append", dest="dates")
    parser.add_argument("--entry-time", default="12:00:00")
    parser.add_argument("--max-age-seconds", type=int, default=300)
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=30)
    parser.add_argument("--target-dte", type=float, default=14.0)
    parser.add_argument("--output", default="artifacts/tardis-option-spread-calibration-v1.json")
    arguments = parser.parse_args()

    root = Path(arguments.data_root)
    dates = arguments.dates or sorted(path.name for path in root.iterdir() if path.is_dir())
    observations: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for date in dates:
        try:
            observations.extend(_measure_day(root, date, arguments))
        except (FileNotFoundError, ValueError) as error:
            failures.append({"date": date, "error": str(error)})

    payload = {
        "schema_version": 1,
        "study": "observed Tardis spread and inverse-IV calibration",
        "forward_source": "inverse put-call parity from paired option mids",
        "spread_definition": "(ask - bid) / option_mid",
        "summary": _summary(observations),
        "failures": failures,
        "observations": observations,
        "limitations": [
            "quarterly first-day sample is sparse and not a continuous exit history",
            "the parity forward can inherit noise from the paired displayed option mids",
            "calibration describes displayed top-of-book, not guaranteed fill size",
            "synthetic quotes remain modeled data and cannot authorize real-money trading",
        ],
    }
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{serialized}\n", encoding="utf-8")
    print(serialized)


def _measure_day(root: Path, date: str, arguments: argparse.Namespace) -> list[dict[str, object]]:
    entry = pd.Timestamp(f"{date}T{arguments.entry_time}Z")
    max_age = pd.Timedelta(seconds=arguments.max_age_seconds)
    day = root / date
    options = reconstruct_top_of_book(day / "OPTIONS.csv.gz", as_of=entry, max_age=max_age)
    perp = _perp_book(day / "BTC-PERPETUAL.csv.gz", as_of=entry, max_age=max_age)
    underlying_mid = _mid(perp)
    selected = select_atm_straddle(
        options.loc[options["symbol"].str.startswith("BTC-")],
        underlying_mid=underlying_mid,
        as_of=entry,
        min_dte=arguments.min_dte,
        max_dte=arguments.max_dte,
        target_dte=arguments.target_dte,
    )
    if len(selected) != 2:
        raise ValueError("no executable BTC ATM call/put pair in the requested DTE range")

    books = options.set_index("symbol")
    selected_by_type = selected.set_index("type")
    call_book = books.loc[selected_by_type.loc["call", "symbol"]]
    put_book = books.loc[selected_by_type.loc["put", "symbol"]]
    call_mid = (float(call_book["bid_price"]) + float(call_book["ask_price"])) / 2
    put_mid = (float(put_book["bid_price"]) + float(put_book["ask_price"])) / 2
    strike = float(selected["strike"].iloc[0])
    parity_forward = inverse_forward_from_parity(
        call_price_btc=call_mid, put_price_btc=put_mid, strike=strike
    )
    rows: list[dict[str, object]] = []
    for leg in selected.itertuples(index=False):
        book = books.loc[leg.symbol]
        bid = float(book["bid_price"])
        ask = float(book["ask_price"])
        mid = (bid + ask) / 2
        time_years = float(leg.dte) / 365
        row: dict[str, object] = {
            "date": date,
            "entry_at": str(entry),
            "symbol": leg.symbol,
            "option_type": leg.type,
            "strike_usd": float(leg.strike),
            "dte": float(leg.dte),
            "underlying_perp_mid_usd": underlying_mid,
            "parity_forward_usd": parity_forward,
            "bid_btc": bid,
            "mid_btc": mid,
            "ask_btc": ask,
            "bid_amount": float(book["bid_amount"]),
            "ask_amount": float(book["ask_amount"]),
            "relative_spread": (ask - bid) / mid,
            "relative_half_spread": (ask - bid) / (2 * mid),
        }
        for side, price in (("bid", bid), ("mid", mid), ("ask", ask)):
            try:
                row[f"{side}_iv"] = implied_volatility(
                    leg.type,
                    price_btc=price,
                    forward=parity_forward,
                    strike=float(leg.strike),
                    time_years=time_years,
                )
                row[f"{side}_iv_error"] = None
            except ValueError as error:
                row[f"{side}_iv"] = None
                row[f"{side}_iv_error"] = str(error)
        rows.append(row)
    return rows


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"legs": 0, "dates": 0}
    frame = pd.DataFrame(rows)
    summary: dict[str, object] = {
        "legs": len(frame),
        "dates": int(frame["date"].nunique()),
        "iv_inversion_failures": int(frame["mid_iv"].isna().sum()),
    }
    for column in ("relative_spread", "relative_half_spread"):
        values = frame[column].to_numpy(dtype=float)
        summary[column] = {
            f"p{percentile}": float(np.percentile(values, percentile))
            for percentile in (50, 75, 90, 95)
        }
    valid_iv = frame.dropna(subset=["bid_iv", "ask_iv"])
    if not valid_iv.empty:
        widths = (valid_iv["ask_iv"] - valid_iv["bid_iv"]).to_numpy(dtype=float)
        summary["iv_width"] = {
            f"p{percentile}": float(np.percentile(widths, percentile))
            for percentile in (50, 75, 90, 95)
        }
    return summary


if __name__ == "__main__":
    main()
