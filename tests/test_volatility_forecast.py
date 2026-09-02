import numpy as np
import pandas as pd
import pytest

from quant_pairs.volatility_forecast import (
    add_expanding_bias_correction,
    attach_dvol,
    build_forecast_panel,
    correct_current_forecast,
    current_forecast,
    diebold_mariano,
    forecast_metrics,
    non_overlapping_forecasts,
)


def _prices(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    returns = rng.normal(0, 0.02, n)
    closes = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01 08:00", periods=n, freq="D", tz="UTC"),
            "close": closes,
        }
    )


def test_forecasts_use_only_past_and_label_following_returns() -> None:
    panel = build_forecast_panel(
        _prices(), horizon_days=5, min_train_days=50, rolling_window=20, garch_refit_days=10
    )

    assert len(panel) == 45
    assert (panel["target_end"] > panel["forecast_at"]).all()
    assert (panel[["rolling_variance", "ewma_variance", "garch_variance"]] > 0).all().all()


def test_future_price_change_does_not_change_first_forecast() -> None:
    prices = _prices()
    original = build_forecast_panel(
        prices, horizon_days=5, min_train_days=50, rolling_window=20, garch_refit_days=10
    )
    changed = prices.copy()
    changed.loc[changed.index[-1], "close"] *= 2
    revised = build_forecast_panel(
        changed, horizon_days=5, min_train_days=50, rolling_window=20, garch_refit_days=10
    )

    columns = ["rolling_variance", "ewma_variance", "garch_variance"]
    assert revised.loc[0, columns].to_numpy(dtype=float) == pytest.approx(
        original.loc[0, columns].to_numpy(dtype=float)
    )


def test_metrics_and_non_overlapping_sample() -> None:
    panel = build_forecast_panel(
        _prices(), horizon_days=5, min_train_days=50, rolling_window=20, garch_refit_days=10
    )
    metrics = forecast_metrics(panel)
    independent = non_overlapping_forecasts(panel)

    assert set(metrics) == {"rolling", "ewma", "garch"}
    assert all(value["observations"] == len(panel) for value in metrics.values())
    assert len(independent) < len(panel)


def test_attach_dvol_uses_last_available_close() -> None:
    panel = pd.DataFrame(
        {
            "forecast_at": pd.to_datetime(["2026-01-03 08:00"], utc=True),
            "garch_variance": [0.16],
            "ewma_variance": [0.2],
            "rolling_variance": [0.25],
        }
    )
    dvol = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"],
            "close": [50.0, 80.0],
        }
    )

    result = attach_dvol(panel, dvol)

    assert result.iloc[0]["dvol"] == 0.50
    assert result.iloc[0]["dvol_minus_garch_variance"] == pytest.approx(0.09)


def test_current_forecast_uses_latest_close_without_future_label() -> None:
    prices = _prices()
    result = current_forecast(prices, horizon_days=14, rolling_window=20)

    expected_at = pd.Timestamp(prices["timestamp"].iloc[-1]) + pd.Timedelta(days=1)
    assert result.iloc[0]["forecast_at"] == expected_at
    assert "target_rv" not in result.columns
    assert result.iloc[0]["garch_rv"] > 0


def test_bias_correction_waits_for_completed_targets() -> None:
    panel = build_forecast_panel(_prices(140), horizon_days=5, min_train_days=50, rolling_window=20)
    corrected = add_expanding_bias_correction(panel, minimum_completed=3)

    assert pd.isna(corrected.iloc[2]["garch_corrected_variance"])
    first = corrected["garch_corrected_variance"].first_valid_index()
    assert first is not None
    completed = corrected.loc[: first - 1]
    completed = completed.loc[completed["target_end"] <= corrected.loc[first, "forecast_at"]]
    expected_scale = completed["target_variance"].sum() / completed["garch_variance"].sum()
    expected_scale = min(max(expected_scale, 0.5), 1.5)
    assert corrected.loc[first, "garch_bias_scale"] == pytest.approx(expected_scale)


def test_current_correction_and_dm_expose_uncertainty() -> None:
    panel = build_forecast_panel(_prices(160), horizon_days=5, min_train_days=50, rolling_window=20)
    current = current_forecast(_prices(160), horizon_days=5, rolling_window=20)
    corrected_current = correct_current_forecast(current, panel, minimum_completed=5)
    corrected = add_expanding_bias_correction(panel, minimum_completed=5).dropna()
    comparison = diebold_mariano(corrected, model="garch_corrected", benchmark="rolling_corrected")

    assert corrected_current.iloc[0]["garch_corrected_rv"] > 0
    assert comparison["ci_low"] <= comparison["mean_qlike_difference"]
    assert comparison["ci_high"] >= comparison["mean_qlike_difference"]
    assert 0 <= comparison["p_value_two_sided"] <= 1
