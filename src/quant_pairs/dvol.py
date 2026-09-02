"""Leakage-aware diagnostics for Deribit's daily DVOL index.

This is deliberately a calibration study, not an options-P&L backtest.  DVOL
is compared with the realized volatility of returns that occur after its
timestamp, so a row never consumes future price information as a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_iv_rv_panel(
    dvol: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizon_days: int = 30,
    annualization_days: int = 365,
) -> pd.DataFrame:
    """Match each DVOL close to the following ``horizon_days`` price returns.

    DVOL is returned by Deribit in percentage points; the panel stores ``iv``
    as a decimal annualized volatility.  Incomplete or gappy forward windows
    are excluded rather than imputed.
    """

    if horizon_days <= 0 or annualization_days <= 0:
        raise ValueError("horizon_days and annualization_days must be positive")
    _require_columns(dvol, {"timestamp", "close"})
    _require_columns(prices, {"timestamp", "close"})

    iv = dvol.loc[:, ["timestamp", "close"]].copy()
    iv["timestamp"] = pd.to_datetime(iv["timestamp"], utc=True, format="mixed")
    iv["iv"] = pd.to_numeric(iv.pop("close"), errors="raise") / 100.0
    iv = iv.drop_duplicates("timestamp", keep="last").sort_values("timestamp")

    price = prices.loc[:, ["timestamp", "close"]].copy()
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True, format="mixed")
    price["close"] = pd.to_numeric(price["close"], errors="raise")
    price = price.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    price["log_return"] = np.log(price["close"]).diff()
    returns = price.dropna(subset=["log_return"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for observation in iv.itertuples(index=False):
        forward = returns.loc[returns["timestamp"] > observation.timestamp].head(horizon_days)
        if len(forward) != horizon_days or _has_gap(forward["timestamp"]):
            continue
        realized = float(
            np.sqrt(np.mean(np.square(forward["log_return"]))) * np.sqrt(annualization_days)
        )
        rows.append(
            {
                "timestamp": observation.timestamp,
                "iv": observation.iv,
                "forward_start": forward["timestamp"].iloc[0],
                "forward_end": forward["timestamp"].iloc[-1],
                "forward_rv": realized,
                "iv_minus_rv": observation.iv - realized,
            }
        )
    return pd.DataFrame(rows)


def non_overlapping(panel: pd.DataFrame, *, horizon_days: int = 30) -> pd.DataFrame:
    """Select chronological entries whose forward outcome windows do not overlap."""

    _require_columns(panel, {"timestamp"})
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    ordered = panel.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True, format="mixed")
    ordered = ordered.sort_values("timestamp")
    chosen: list[int] = []
    available_after: pd.Timestamp | None = None
    for index, row in ordered.iterrows():
        if available_after is None or row["timestamp"] >= available_after:
            chosen.append(index)
            available_after = row["timestamp"] + pd.Timedelta(days=horizon_days)
    return ordered.loc[chosen].reset_index(drop=True)


def _has_gap(timestamps: pd.Series) -> bool:
    return bool((timestamps.diff().dropna() > pd.Timedelta(days=2)).any())


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")
