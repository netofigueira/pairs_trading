"""Leakage-safe daily forecasts of future BTC realized volatility."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FORECAST_COLUMNS = ("rolling", "ewma", "garch")


def build_forecast_panel(
    prices: pd.DataFrame,
    *,
    horizon_days: int,
    min_train_days: int = 365,
    rolling_window: int = 30,
    ewma_lambda: float = 0.94,
    garch_refit_days: int = 30,
    annualization_days: int = 365,
) -> pd.DataFrame:
    """Produce as-of forecasts and future realized-volatility labels.

    Daily candle closes become available one day after their interval-start
    timestamp. Every forecast uses returns through ``forecast_at`` only; the
    label consumes the following ``horizon_days`` returns.
    """

    if min(horizon_days, min_train_days, rolling_window, garch_refit_days) <= 0:
        raise ValueError("forecast windows must be positive")
    if not 0 < ewma_lambda < 1 or annualization_days <= 0:
        raise ValueError("invalid EWMA lambda or annualization")
    returns = _available_log_returns(prices)
    if len(returns) < min_train_days + horizon_days:
        raise ValueError("insufficient price history for requested forecast")

    squared = returns.to_numpy(dtype=float) ** 2
    ewma = np.empty(len(squared), dtype=float)
    ewma[0] = squared[0]
    for index in range(1, len(squared)):
        ewma[index] = ewma_lambda * ewma[index - 1] + (1 - ewma_lambda) * squared[index]

    rows: list[dict[str, object]] = []
    garch_parameters: tuple[float, float, float] | None = None
    last_fit = -garch_refit_days
    for index in range(min_train_days - 1, len(returns) - horizon_days):
        if garch_parameters is None or index - last_fit >= garch_refit_days:
            garch_parameters = _fit_garch(returns.iloc[: index + 1].to_numpy(dtype=float))
            last_fit = index
        past = squared[index - rolling_window + 1 : index + 1]
        future = squared[index + 1 : index + 1 + horizon_days]
        rolling_variance = float(past.mean() * annualization_days)
        ewma_variance = float(ewma[index] * annualization_days)
        garch_variance = _garch_average_forecast(
            returns.iloc[: index + 1].to_numpy(dtype=float),
            garch_parameters,
            horizon_days=horizon_days,
        ) * annualization_days
        target_variance = float(future.mean() * annualization_days)
        rows.append(
            {
                "forecast_at": returns.index[index],
                "target_end": returns.index[index + horizon_days],
                "horizon_days": horizon_days,
                "target_variance": target_variance,
                "target_rv": math.sqrt(target_variance),
                "rolling_variance": rolling_variance,
                "rolling_rv": math.sqrt(rolling_variance),
                "ewma_variance": ewma_variance,
                "ewma_rv": math.sqrt(ewma_variance),
                "garch_variance": garch_variance,
                "garch_rv": math.sqrt(garch_variance),
                "garch_omega": garch_parameters[0],
                "garch_alpha": garch_parameters[1],
                "garch_beta": garch_parameters[2],
            }
        )
    return pd.DataFrame(rows)


def forecast_metrics(panel: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Evaluate variance forecasts with MSE and Gaussian QLIKE loss."""

    if panel.empty:
        raise ValueError("forecast panel cannot be empty")
    target = panel["target_variance"].to_numpy(dtype=float)
    metrics: dict[str, dict[str, float | int]] = {}
    for name in FORECAST_COLUMNS:
        forecast = panel[f"{name}_variance"].to_numpy(dtype=float)
        safe = np.maximum(forecast, 1e-12)
        metrics[name] = {
            "observations": len(panel),
            "mse_variance": float(np.mean((forecast - target) ** 2)),
            "qlike": float(np.mean(np.log(safe) + target / safe)),
            "mean_forecast_rv": float(np.mean(np.sqrt(safe))),
            "mean_target_rv": float(np.mean(np.sqrt(target))),
            "rv_correlation": float(np.corrcoef(np.sqrt(safe), np.sqrt(target))[0, 1]),
        }
    return metrics


def current_forecast(
    prices: pd.DataFrame,
    *,
    horizon_days: int,
    rolling_window: int = 30,
    ewma_lambda: float = 0.94,
    annualization_days: int = 365,
) -> pd.DataFrame:
    """Forecast from the most recent available close without requiring a label."""

    returns = _available_log_returns(prices)
    if len(returns) < rolling_window:
        raise ValueError("insufficient price history for current forecast")
    squared = returns.to_numpy(dtype=float) ** 2
    ewma_variance = squared[0]
    for value in squared[1:]:
        ewma_variance = ewma_lambda * ewma_variance + (1 - ewma_lambda) * value
    parameters = _fit_garch(returns.to_numpy(dtype=float))
    rolling_variance = float(squared[-rolling_window:].mean() * annualization_days)
    ewma_annualized = float(ewma_variance * annualization_days)
    garch_variance = _garch_average_forecast(
        returns.to_numpy(dtype=float), parameters, horizon_days=horizon_days
    ) * annualization_days
    return pd.DataFrame(
        [
            {
                "forecast_at": returns.index[-1],
                "horizon_days": horizon_days,
                "rolling_variance": rolling_variance,
                "rolling_rv": math.sqrt(rolling_variance),
                "ewma_variance": ewma_annualized,
                "ewma_rv": math.sqrt(ewma_annualized),
                "garch_variance": garch_variance,
                "garch_rv": math.sqrt(garch_variance),
                "garch_omega": parameters[0],
                "garch_alpha": parameters[1],
                "garch_beta": parameters[2],
            }
        ]
    )


def non_overlapping_forecasts(panel: pd.DataFrame) -> pd.DataFrame:
    """Select forecasts with non-overlapping target windows."""

    if panel.empty:
        return panel.copy()
    ordered = panel.sort_values("forecast_at")
    chosen = []
    available_after = None
    for index, row in ordered.iterrows():
        if available_after is None or row["forecast_at"] >= available_after:
            chosen.append(index)
            available_after = row["target_end"]
    return ordered.loc[chosen].reset_index(drop=True)


def attach_dvol(panel: pd.DataFrame, dvol: pd.DataFrame) -> pd.DataFrame:
    """Attach the last DVOL close available at each forecast timestamp."""

    required = {"timestamp", "close"}
    missing = required.difference(dvol.columns)
    if missing:
        raise ValueError(f"DVOL frame is missing columns: {sorted(missing)}")
    iv = dvol.loc[:, ["timestamp", "close"]].copy()
    iv["available_at"] = (
        pd.to_datetime(iv.pop("timestamp"), utc=True, format="mixed") + pd.Timedelta(days=1)
    )
    iv["dvol"] = pd.to_numeric(iv.pop("close"), errors="raise") / 100
    merged = pd.merge_asof(
        panel.sort_values("forecast_at"),
        iv.sort_values("available_at"),
        left_on="forecast_at",
        right_on="available_at",
        direction="backward",
    ).drop(columns="available_at")
    for name in FORECAST_COLUMNS:
        merged[f"dvol_minus_{name}_variance"] = merged["dvol"] ** 2 - merged[
            f"{name}_variance"
        ]
    return merged


def _available_log_returns(prices: pd.DataFrame) -> pd.Series:
    missing = {"timestamp", "close"}.difference(prices.columns)
    if missing:
        raise ValueError(f"price frame is missing columns: {sorted(missing)}")
    frame = prices.loc[:, ["timestamp", "close"]].copy()
    frame["forecast_at"] = (
        pd.to_datetime(frame.pop("timestamp"), utc=True, format="mixed")
        + pd.Timedelta(days=1)
    )
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    frame = frame.drop_duplicates("forecast_at", keep="last").sort_values("forecast_at")
    if frame["close"].le(0).any():
        raise ValueError("prices must be positive")
    values = np.log(frame["close"]).diff()
    values.index = frame["forecast_at"]
    return values.dropna()


def _fit_garch(returns: np.ndarray) -> tuple[float, float, float]:
    """Fit a zero-mean GARCH(1,1) by Gaussian maximum likelihood."""

    scaled = np.asarray(returns, dtype=float) * 100
    variance = float(np.var(scaled))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("returns must have positive variance")

    def objective(parameters: np.ndarray) -> float:
        omega, alpha, beta = parameters
        conditional = _filter_garch_variance(scaled, (omega, alpha, beta), variance)
        return float(0.5 * np.sum(np.log(conditional) + scaled**2 / conditional))

    result = minimize(
        objective,
        x0=np.array([variance * 0.05, 0.08, 0.87]),
        method="SLSQP",
        bounds=((1e-10, variance * 10), (1e-6, 0.5), (1e-6, 0.999)),
        constraints={"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise ValueError(f"GARCH fit failed: {result.message}")
    omega, alpha, beta = (float(value) for value in result.x)
    return omega / 10_000, alpha, beta


def _garch_average_forecast(
    returns: np.ndarray,
    parameters: tuple[float, float, float],
    *,
    horizon_days: int,
) -> float:
    omega, alpha, beta = parameters
    unconditional = float(np.var(returns))
    conditional = _filter_garch_variance(returns, parameters, unconditional)
    next_variance = omega + alpha * returns[-1] ** 2 + beta * conditional[-1]
    persistence = alpha + beta
    long_run = omega / (1 - persistence)
    forecasts = [
        long_run + persistence**step * (next_variance - long_run)
        for step in range(horizon_days)
    ]
    return float(np.mean(forecasts))


def _filter_garch_variance(
    returns: np.ndarray,
    parameters: tuple[float, float, float],
    initial_variance: float,
) -> np.ndarray:
    omega, alpha, beta = parameters
    conditional = np.empty(len(returns), dtype=float)
    conditional[0] = initial_variance
    for index in range(1, len(returns)):
        conditional[index] = (
            omega + alpha * returns[index - 1] ** 2 + beta * conditional[index - 1]
        )
    return np.maximum(conditional, 1e-12)
