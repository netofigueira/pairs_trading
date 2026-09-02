import numpy as np
import pandas as pd
import pytest

from quant_pairs.backtest import BacktestConfig
from quant_pairs.cointegration import FormationModel
from quant_pairs.screener import ScreenedPair
from quant_pairs.walk_forward import (
    WalkForwardConfig,
    _candidate_execution,
    _maximum_weight_matching,
    run_walk_forward,
)


def prices() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=160, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    x = np.exp(np.cumsum(rng.normal(0, 0.002, len(index))))
    # A stable relation makes this fixture suitable for exercising fold boundaries.
    y = np.exp(0.2 + 0.9 * np.log(x) + rng.normal(0, 0.002, len(index)))
    return pd.DataFrame({"Y": y, "X": x}, index=index)


def test_walk_forward_separates_formation_from_non_overlapping_trade_windows(monkeypatch) -> None:
    monkeypatch.setattr("quant_pairs.walk_forward.screen_pairs", lambda *_args, **_kwargs: [])
    result = run_walk_forward(
        prices(),
        BacktestConfig(),
        WalkForwardConfig(formation_bars=90, trade_bars=20, step_bars=20),
    )

    assert len(result.folds) == 3
    assert (result.folds["formation_end"] < result.folds["trade_start"]).all()
    prior_ends = result.folds["trade_end"].iloc[:-1].to_numpy()
    next_starts = result.folds["trade_start"].iloc[1:].to_numpy()
    assert (prior_ends < next_starts).all()
    assert result.selections.empty


def test_walk_forward_rejects_overlapping_aggregate_windows() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        WalkForwardConfig(formation_bars=90, trade_bars=30, step_bars=20)


def test_half_life_signal_window_is_fixed_from_the_formation_candidate() -> None:
    candidate = ScreenedPair(
        model=FormationModel(
            dependent="Y",
            independent="X",
            alpha=0.0,
            beta=1.0,
            spread_mean=0.0,
            spread_std=1.0,
            coint_t_stat=-4.0,
            coint_pvalue=0.01,
            critical_values=(-3.9, -3.3, -3.0),
            formation_start=pd.Timestamp("2024-01-01T00:00:00Z"),
            formation_end=pd.Timestamp("2024-01-02T00:00:00Z"),
            observations=100,
        ),
        half_life_bars=17.6,
        fdr_qvalue=0.01,
        accepted=True,
    )

    execution = _candidate_execution(BacktestConfig(signal_scale="half_life_rolling"), candidate)

    assert execution.signal_scale == "rolling"
    assert execution.volatility_span_bars == 18


def test_matching_excludes_shared_assets_and_keeps_best_disjoint_pairs() -> None:
    def candidate(dependent: str, independent: str, qvalue: float) -> ScreenedPair:
        return ScreenedPair(
            model=FormationModel(
                dependent=dependent,
                independent=independent,
                alpha=0.0,
                beta=1.0,
                spread_mean=0.0,
                spread_std=1.0,
                coint_t_stat=-4.0,
                coint_pvalue=qvalue,
                critical_values=(-3.9, -3.3, -3.0),
                formation_start=pd.Timestamp("2024-01-01T00:00:00Z"),
                formation_end=pd.Timestamp("2024-01-02T00:00:00Z"),
                observations=100,
            ),
            half_life_bars=12.0,
            fdr_qvalue=qvalue,
            accepted=True,
        )

    selected = _maximum_weight_matching(
        [
            candidate("BTC", "ETH", 0.001),
            candidate("BTC", "SOL", 0.01),
            candidate("ETH", "XRP", 0.01),
            candidate("SOL", "ADA", 0.001),
        ]
    )

    symbols = [
        symbol
        for item in selected
        for symbol in (item.model.dependent, item.model.independent)
    ]
    assert len(symbols) == len(set(symbols))
    assert {(item.model.dependent, item.model.independent) for item in selected} == {
        ("BTC", "ETH"),
        ("SOL", "ADA"),
    }
