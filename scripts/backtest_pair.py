"""Run one out-of-sample pair backtest from data already stored in the local lake."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_pairs.backtest import BacktestConfig, run_pair_backtest
from quant_pairs.cointegration import fit_formation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependent", required=True, help="y leg, e.g. BTCUSDT")
    parser.add_argument("--independent", required=True, help="x leg, e.g. ETHUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument(
        "--formation-bars", type=int, default=720, help="default: 30 days of 1h bars"
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--stop-z", type=float, default=4.0)
    parser.add_argument("--max-holding-bars", type=int, default=72)
    parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument(
        "--allow-noncointegrated",
        action="store_true",
        help="run despite a formation p-value >= 0.05; diagnostic use only",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    y = _prices(arguments.data_root, arguments.dependent, arguments.interval)
    x = _prices(arguments.data_root, arguments.independent, arguments.interval)
    prices = pd.concat((y, x), axis=1, join="inner").dropna()
    if len(prices) <= arguments.formation_bars + 2:
        raise SystemExit("not enough aligned bars for the requested formation and trade windows")

    formation = prices.iloc[: arguments.formation_bars]
    trade = prices.iloc[arguments.formation_bars :]
    model = fit_formation_model(formation.iloc[:, 0], formation.iloc[:, 1])
    if not model.is_cointegrated and not arguments.allow_noncointegrated:
        raise SystemExit(
            f"formation rejected: coint_pvalue={model.coint_pvalue:.6f} >= 0.05; "
            "use the screener to find valid candidates"
        )
    config = BacktestConfig(
        entry_z=arguments.entry_z,
        exit_z=arguments.exit_z,
        stop_z=arguments.stop_z,
        max_holding_bars=arguments.max_holding_bars,
        taker_fee_bps=arguments.taker_fee_bps,
        slippage_bps=arguments.slippage_bps,
    )
    result = run_pair_backtest(
        model,
        trade.iloc[:, 0],
        trade.iloc[:, 1],
        config,
        dependent_funding=_funding(arguments.data_root, arguments.dependent),
        independent_funding=_funding(arguments.data_root, arguments.independent),
    )
    print(
        f"formation={model.formation_start}..{model.formation_end} "
        f"observations={model.observations}"
    )
    print(f"beta={model.beta:.6f} coint_pvalue={model.coint_pvalue:.6f}")
    print(
        f"trades={len(result.trades)} net_pnl={result.net_pnl:.6f} "
        f"gross_return={result.gross_return:.4%}"
    )
    if not result.trades.empty:
        print(result.trades.to_string(index=False))


def _prices(data_root: str, symbol: str, interval: str) -> pd.Series:
    path = (
        Path(data_root)
        / "market"
        / "binance-usdm"
        / "klines"
        / symbol.upper()
        / f"{interval}.csv.gz"
    )
    frame = pd.read_csv(path)
    series = pd.Series(
        pd.to_numeric(frame["close"], errors="raise").to_numpy(),
        index=pd.to_datetime(frame["open_time"], utc=True, format="mixed"),
        name=symbol.upper(),
    )
    return series.sort_index()


def _funding(data_root: str, symbol: str) -> pd.DataFrame:
    path = Path(data_root) / "market" / "binance-usdm" / "funding" / f"{symbol.upper()}.csv.gz"
    frame = pd.read_csv(path)
    frame["funding_time"] = pd.to_datetime(frame["funding_time"], utc=True, format="mixed")
    return frame


if __name__ == "__main__":
    main()
