import numpy as np
import pandas as pd
import pytest

from quant_pairs.cointegration import (
    InvalidFormationWindow,
    fit_formation_model,
    signals_from_zscore,
)


def prices(seed: int = 7) -> tuple[pd.Series, pd.Series]:
    """Synthetic cointegrated log-price process for deterministic unit tests."""

    generator = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=180, freq="h")
    x_log = 4 + np.cumsum(generator.normal(0, 0.01, len(index)))
    spread = np.zeros(len(index))
    for i in range(1, len(index)):
        spread[i] = 0.85 * spread[i - 1] + generator.normal(0, 0.003)
    y_log = 0.2 + 1.4 * x_log + spread
    y = pd.Series(np.exp(y_log), index=index, name="Y")
    x = pd.Series(np.exp(x_log), index=index, name="X")
    return y, x


def test_trade_window_uses_frozen_formation_parameters() -> None:
    y, x = prices()
    model = fit_formation_model(y.iloc[:120], x.iloc[:120])

    zscore = model.zscore(y.iloc[120:], x.iloc[120:])
    manual_spread = np.log(y.iloc[120:]) - model.alpha - model.beta * np.log(x.iloc[120:])
    expected = (manual_spread - model.spread_mean) / model.spread_std

    pd.testing.assert_series_equal(zscore, expected, check_names=False)
    assert model.formation_end < zscore.index.min()


def test_crossings_emit_one_signal_per_threshold_entry() -> None:
    index = pd.date_range("2024-01-01", periods=6)
    zscore = pd.Series([0.0, 1.9, 2.1, 2.3, 1.8, -2.1], index=index)
    signals = signals_from_zscore(zscore, entry_z=2.0)

    assert [(signal.zscore, signal.direction) for signal in signals] == [(2.1, -1), (-2.1, 1)]


def test_rejects_non_positive_prices() -> None:
    index = pd.date_range("2024-01-01", periods=90)
    y = pd.Series(np.ones(90), index=index)
    x = pd.Series(np.ones(90), index=index)
    x.iloc[0] = 0

    with pytest.raises(InvalidFormationWindow, match="strictly positive"):
        fit_formation_model(y, x)
