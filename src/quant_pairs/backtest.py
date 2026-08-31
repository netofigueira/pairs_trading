"""Event-driven, single-pair backtest with conservative execution assumptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

from .cointegration import FormationModel

ExitReason = Literal["mean_reversion", "stop_loss", "time_stop", "end_of_data"]


@dataclass(frozen=True)
class BacktestConfig:
    """All values are explicit so research runs remain comparable."""

    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    max_holding_bars: int = 72
    gross_notional: float = 1.0
    taker_fee_bps: float = 5.0
    slippage_bps: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("require exit_z < entry_z < stop_z")
        if self.max_holding_bars <= 0 or self.gross_notional <= 0:
            raise ValueError("max_holding_bars and gross_notional must be positive")
        if self.taker_fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs cannot be negative")

    @property
    def cost_rate(self) -> float:
        return (self.taker_fee_bps + self.slippage_bps) / 10_000


@dataclass(frozen=True)
class PairTrade:
    direction: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_y: float
    entry_x: float
    exit_y: float
    exit_x: float
    qty_y: float
    qty_x: float
    exit_reason: ExitReason
    gross_pnl: float
    trading_cost: float
    funding_pnl: float
    net_pnl: float
    gross_return: float
    holding_bars: int


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    zscores: pd.Series
    config: BacktestConfig

    @property
    def net_pnl(self) -> float:
        return float(self.trades["net_pnl"].sum()) if not self.trades.empty else 0.0

    @property
    def gross_return(self) -> float:
        return self.net_pnl / self.config.gross_notional


@dataclass(frozen=True)
class _OpenPosition:
    direction: int
    entry_index: int
    entry_time: pd.Timestamp
    entry_y: float
    entry_x: float
    qty_y: float
    qty_x: float
    entry_cost: float


def run_pair_backtest(
    model: FormationModel,
    dependent_prices: pd.Series,
    independent_prices: pd.Series,
    config: BacktestConfig,
    *,
    dependent_funding: pd.DataFrame | None = None,
    independent_funding: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run a one-position-per-pair event loop on an out-of-sample trade window.

    A signal at bar t is filled at the close of bar t+1 with adverse slippage.
    This deliberately avoids pretending that the triggering close was tradable.
    """

    prices = pd.concat((dependent_prices, independent_prices), axis=1, join="inner").dropna()
    prices.columns = ["y", "x"]
    if len(prices) < 3:
        raise ValueError("trade window needs at least three aligned prices")
    zscores = model.zscore(prices["y"], prices["x"])
    position: _OpenPosition | None = None
    trades: list[PairTrade] = []

    for index in range(len(prices) - 1):
        zscore = float(zscores.iloc[index])
        next_timestamp = pd.Timestamp(prices.index[index + 1])
        next_y = float(prices["y"].iloc[index + 1])
        next_x = float(prices["x"].iloc[index + 1])

        if position is None:
            previous_zscore = float(zscores.iloc[index - 1]) if index else 0.0
            direction = _entry_direction(previous_zscore, zscore, config.entry_z)
            if direction:
                position = _open_position(
                    direction, index + 1, next_timestamp, next_y, next_x, model.beta, config
                )
            continue

        holding_bars = index - position.entry_index + 1
        exit_reason = _exit_reason(zscore, holding_bars, config)
        if exit_reason is not None:
            trades.append(
                _close_position(
                    position,
                    next_timestamp,
                    next_y,
                    next_x,
                    holding_bars,
                    exit_reason,
                    config,
                    dependent_funding,
                    independent_funding,
                )
            )
            position = None

    if position is not None:
        final_time = pd.Timestamp(prices.index[-1])
        trades.append(
            _close_position(
                position,
                final_time,
                float(prices["y"].iloc[-1]),
                float(prices["x"].iloc[-1]),
                len(prices) - 1 - position.entry_index,
                "end_of_data",
                config,
                dependent_funding,
                independent_funding,
            )
        )
    frame = pd.DataFrame([asdict(trade) for trade in trades])
    return BacktestResult(trades=frame, zscores=zscores, config=config)


def _entry_direction(previous_zscore: float, zscore: float, entry_z: float) -> int:
    if previous_zscore < entry_z <= zscore:
        return -1  # short y / long x: the spread is positive and expensive
    if previous_zscore > -entry_z >= zscore:
        return 1  # long y / short x: the spread is negative and cheap
    return 0


def _open_position(
    direction: int,
    entry_index: int,
    entry_time: pd.Timestamp,
    raw_y: float,
    raw_x: float,
    beta: float,
    config: BacktestConfig,
) -> _OpenPosition:
    abs_beta = abs(beta)
    y_notional = config.gross_notional / (1 + abs_beta)
    x_notional = config.gross_notional - y_notional
    entry_y = _fill_price(raw_y, direction, config.slippage_bps)
    entry_x = _fill_price(raw_x, -direction, config.slippage_bps)
    qty_y = direction * y_notional / entry_y
    qty_x = -direction * x_notional / entry_x
    entry_cost = (abs(qty_y * entry_y) + abs(qty_x * entry_x)) * config.taker_fee_bps / 10_000
    return _OpenPosition(
        direction=direction,
        entry_index=entry_index,
        entry_time=entry_time,
        entry_y=entry_y,
        entry_x=entry_x,
        qty_y=qty_y,
        qty_x=qty_x,
        entry_cost=entry_cost,
    )


def _close_position(
    position: _OpenPosition,
    exit_time: pd.Timestamp,
    raw_y: float,
    raw_x: float,
    holding_bars: int,
    exit_reason: ExitReason,
    config: BacktestConfig,
    dependent_funding: pd.DataFrame | None,
    independent_funding: pd.DataFrame | None,
) -> PairTrade:
    exit_y = _fill_price(raw_y, -position.direction, config.slippage_bps)
    exit_x = _fill_price(raw_x, position.direction, config.slippage_bps)
    gross_pnl = position.qty_y * (exit_y - position.entry_y)
    gross_pnl += position.qty_x * (exit_x - position.entry_x)
    exit_notional = abs(position.qty_y * exit_y) + abs(position.qty_x * exit_x)
    exit_cost = exit_notional * config.taker_fee_bps / 10_000
    funding_pnl = _funding_pnl(
        position.qty_y, position.entry_time, exit_time, dependent_funding
    ) + _funding_pnl(position.qty_x, position.entry_time, exit_time, independent_funding)
    trading_cost = position.entry_cost + exit_cost
    net_pnl = gross_pnl - trading_cost + funding_pnl
    return PairTrade(
        direction=position.direction,
        entry_time=position.entry_time,
        exit_time=exit_time,
        entry_y=position.entry_y,
        entry_x=position.entry_x,
        exit_y=exit_y,
        exit_x=exit_x,
        qty_y=position.qty_y,
        qty_x=position.qty_x,
        exit_reason=exit_reason,
        gross_pnl=gross_pnl,
        trading_cost=trading_cost,
        funding_pnl=funding_pnl,
        net_pnl=net_pnl,
        gross_return=net_pnl / config.gross_notional,
        holding_bars=max(1, holding_bars),
    )


def _fill_price(raw_price: float, side: int, slippage_bps: float) -> float:
    """Apply adverse slippage: buys pay more; shorts sell for less."""

    if raw_price <= 0:
        raise ValueError("prices must be positive")
    return raw_price * (1 + side * slippage_bps / 10_000)


def _exit_reason(zscore: float, holding_bars: int, config: BacktestConfig) -> ExitReason | None:
    if abs(zscore) <= config.exit_z:
        return "mean_reversion"
    if abs(zscore) >= config.stop_z:
        return "stop_loss"
    if holding_bars >= config.max_holding_bars:
        return "time_stop"
    return None


def _funding_pnl(
    quantity: float,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    funding: pd.DataFrame | None,
) -> float:
    if funding is None or funding.empty:
        return 0.0
    required = {"funding_time", "funding_rate", "mark_price"}
    if missing := required.difference(funding.columns):
        raise ValueError(f"funding data is missing columns: {sorted(missing)}")
    events = funding.copy()
    events["funding_time"] = pd.to_datetime(events["funding_time"], utc=True)
    selected = events[(events["funding_time"] > entry_time) & (events["funding_time"] <= exit_time)]
    return float(-(quantity * selected["mark_price"] * selected["funding_rate"]).sum())
