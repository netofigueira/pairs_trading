"""Rolling, out-of-sample evaluation for the pair-selection process.

Every fold selects pairs using *only* its preceding formation window.  The
subsequent trading window is never used to fit a hedge, choose candidates, or
normalise a z-score.  Trade windows must not overlap, so aggregated results do
not accidentally count the same market move more than once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cache
from math import log

import pandas as pd

from .backtest import BacktestConfig, run_pair_backtest
from .screener import ScreenedPair, screen_pairs


@dataclass(frozen=True)
class WalkForwardConfig:
    formation_bars: int = 4_320
    trade_bars: int = 168
    step_bars: int = 168
    fdr_alpha: float = 0.05
    min_half_life_bars: float = 4.0
    max_half_life_bars: float = 72.0
    portfolio_matching: bool = False

    def __post_init__(self) -> None:
        if self.formation_bars < 90:
            raise ValueError("formation_bars must be at least 90")
        if self.trade_bars <= 0 or self.step_bars <= 0:
            raise ValueError("trade_bars and step_bars must be positive")
        if self.trade_bars > self.step_bars:
            raise ValueError(
                "trade_bars cannot exceed step_bars: overlapping OOS windows "
                "must not be aggregated"
            )


@dataclass(frozen=True)
class WalkForwardResult:
    folds: pd.DataFrame
    selections: pd.DataFrame
    trades: pd.DataFrame
    config: WalkForwardConfig

    @property
    def net_pnl(self) -> float:
        return float(self.trades["net_pnl"].sum()) if not self.trades.empty else 0.0

    def metrics(self) -> dict[str, float | int]:
        if self.trades.empty:
            return {"folds": len(self.folds), "selected_pair_folds": 0, "trades": 0, "net_pnl": 0.0}
        return {
            "folds": len(self.folds),
            "selected_pair_folds": int(self.selections["accepted"].sum()),
            "portfolio_pair_folds": int(self.selections["portfolio_selected"].sum()),
            "trades": len(self.trades),
            "net_pnl": self.net_pnl,
            "win_rate": float((self.trades["net_pnl"] > 0).mean()),
            "mean_trade_pnl": float(self.trades["net_pnl"].mean()),
            "max_drawdown": _trade_equity_drawdown(self.trades),
        }


def run_walk_forward(
    prices: pd.DataFrame,
    execution: BacktestConfig,
    config: WalkForwardConfig,
    *,
    funding_by_symbol: Mapping[str, pd.DataFrame] | None = None,
) -> WalkForwardResult:
    """Evaluate a universe via repeated formation-selection-trade folds.

    ``prices`` must be a fully aligned, closed-candle matrix.  A candidate is
    re-screened in each fold with Benjamini--Hochberg correction across the
    complete universe, then traded only in the following non-overlapping slice.
    """

    prices = prices.sort_index().dropna().copy()
    if prices.shape[1] < 2:
        raise ValueError("walk-forward requires at least two symbols")
    if len(prices) < config.formation_bars + config.trade_bars:
        raise ValueError("not enough aligned bars for one formation and trade fold")

    fold_rows: list[dict] = []
    selection_rows: list[dict] = []
    trade_rows: list[dict] = []
    fold_number = 0
    last_start = len(prices) - config.formation_bars - config.trade_bars
    for start in range(0, last_start + 1, config.step_bars):
        formation = prices.iloc[start : start + config.formation_bars]
        trade_start = start + config.formation_bars
        trade = prices.iloc[trade_start : trade_start + config.trade_bars]
        candidates = screen_pairs(
            formation,
            fdr_alpha=config.fdr_alpha,
            min_half_life_bars=config.min_half_life_bars,
            max_half_life_bars=config.max_half_life_bars,
        )
        accepted = [candidate for candidate in candidates if candidate.accepted]
        portfolio = _maximum_weight_matching(accepted) if config.portfolio_matching else accepted
        portfolio_ids = {id(candidate) for candidate in portfolio}
        fold_rows.append(
            {
                "fold": fold_number,
                "formation_start": formation.index.min(),
                "formation_end": formation.index.max(),
                "trade_start": trade.index.min(),
                "trade_end": trade.index.max(),
                "tested_pairs": len(candidates),
                "accepted_pairs": len(accepted),
                "portfolio_pairs": len(portfolio),
            }
        )
        for candidate in candidates:
            _append_selection(
                selection_rows,
                fold_number,
                candidate,
                portfolio_selected=id(candidate) in portfolio_ids,
            )
        for candidate in portfolio:
            spread_history = candidate.model.spread(
                formation[candidate.model.dependent],
                formation[candidate.model.independent],
            )
            candidate_execution = _candidate_execution(execution, candidate)
            result = run_pair_backtest(
                candidate.model,
                trade[candidate.model.dependent],
                trade[candidate.model.independent],
                candidate_execution,
                dependent_funding=_funding_for_symbol(
                    funding_by_symbol, candidate.model.dependent
                ),
                independent_funding=_funding_for_symbol(
                    funding_by_symbol, candidate.model.independent
                ),
                spread_history=spread_history,
            )
            for record in result.trades.to_dict("records"):
                record.update(
                    {
                        "fold": fold_number,
                        "dependent_symbol": candidate.model.dependent,
                        "independent_symbol": candidate.model.independent,
                        "signal_scale": candidate_execution.signal_scale,
                        "volatility_span_bars": candidate_execution.volatility_span_bars,
                    }
                )
                trade_rows.append(record)
        fold_number += 1

    return WalkForwardResult(
        folds=pd.DataFrame(fold_rows),
        selections=pd.DataFrame(selection_rows),
        trades=pd.DataFrame(trade_rows),
        config=config,
    )


def _append_selection(
    rows: list[dict], fold: int, candidate: ScreenedPair, *, portfolio_selected: bool
) -> None:
    model = candidate.model
    rows.append(
        {
            "fold": fold,
            "dependent_symbol": model.dependent,
            "independent_symbol": model.independent,
            "coint_pvalue": model.coint_pvalue,
            "fdr_qvalue": candidate.fdr_qvalue,
            "half_life_bars": candidate.half_life_bars,
            "accepted": candidate.accepted,
            "portfolio_selected": portfolio_selected,
            "formation_start": model.formation_start,
            "formation_end": model.formation_end,
        }
    )


def _maximum_weight_matching(candidates: list[ScreenedPair]) -> list[ScreenedPair]:
    """Choose a deterministic, maximum-quality set of pairs without shared assets."""

    symbols = sorted(
        {symbol for item in candidates for symbol in (item.model.dependent, item.model.independent)}
    )
    positions = {symbol: index for index, symbol in enumerate(symbols)}
    edges: dict[tuple[int, int], ScreenedPair] = {}
    for candidate in candidates:
        first, second = sorted(
            (positions[candidate.model.dependent], positions[candidate.model.independent])
        )
        edges[(first, second)] = candidate

    @cache
    def solve(mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if mask == 0:
            return 0.0, ()
        first = (mask & -mask).bit_length() - 1
        best_score, best_edges = solve(mask & ~(1 << first))
        for second in range(first + 1, len(symbols)):
            candidate = edges.get((first, second))
            if candidate is None or not mask & (1 << second):
                continue
            score, selected = solve(mask & ~(1 << first) & ~(1 << second))
            candidate_score = -log(max(candidate.fdr_qvalue, 1e-12))
            candidate_score -= 0.01 * candidate.half_life_bars
            if score + candidate_score > best_score:
                best_score, best_edges = score + candidate_score, selected + ((first, second),)
        return best_score, best_edges

    _, selected_edges = solve((1 << len(symbols)) - 1)
    return [edges[edge] for edge in selected_edges]


def _candidate_execution(execution: BacktestConfig, candidate: ScreenedPair) -> BacktestConfig:
    """Resolve H1's fixed-per-fold rolling window from formation-only half-life."""

    if execution.signal_scale != "half_life_rolling":
        return execution
    if candidate.half_life_bars is None:
        raise ValueError("accepted candidate is missing its formation half-life")
    return replace(
        execution,
        signal_scale="rolling",
        volatility_span_bars=max(2, round(candidate.half_life_bars)),
    )


def _funding_for_symbol(
    funding_by_symbol: Mapping[str, pd.DataFrame] | None, symbol: str
) -> pd.DataFrame | None:
    if funding_by_symbol is None:
        return None
    return funding_by_symbol.get(symbol)


def _trade_equity_drawdown(trades: pd.DataFrame) -> float:
    """Drawdown of realized trade P&L; portfolio mark-to-market is a later gate."""

    equity = trades.sort_values("exit_time")["net_pnl"].cumsum()
    return float((equity - equity.cummax()).min())
