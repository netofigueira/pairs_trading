import numpy as np
import pandas as pd
import pytest

from quant_pairs.backtest import BacktestConfig
from quant_pairs.walk_forward import WalkForwardConfig, run_walk_forward


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
