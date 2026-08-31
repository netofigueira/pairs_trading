"""Out-of-sample-safe Engle--Granger formation and signal calculations.

The module intentionally separates formation from trading. A model is fit on a
formation window once and is immutable while it scores the subsequent trade
window; refitting it with trade-window prices would introduce look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


class InvalidFormationWindow(ValueError):
    """The input cannot define a statistically meaningful formation model."""


@dataclass(frozen=True)
class FormationModel:
    """A cointegration relationship estimated solely on a formation window.

    `dependent` is the y leg and `independent` is the x leg. Prices are logged
    before regression, so beta is a dimensionless hedge coefficient.
    """

    dependent: str
    independent: str
    alpha: float
    beta: float
    spread_mean: float
    spread_std: float
    coint_t_stat: float
    coint_pvalue: float
    critical_values: tuple[float, float, float]
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    observations: int

    @property
    def is_cointegrated(self) -> bool:
        """A convenience flag only; portfolio selection must still apply FDR."""

        return self.coint_pvalue < 0.05

    def spread(self, dependent_prices: pd.Series, independent_prices: pd.Series) -> pd.Series:
        """Calculate the spread with frozen formation parameters.

        This is the only supported way to score a trade window. It deliberately
        does not estimate a new alpha, beta, mean, or standard deviation.
        """

        y, x = _aligned_log_prices(dependent_prices, independent_prices)
        return y - self.alpha - self.beta * x

    def zscore(self, dependent_prices: pd.Series, independent_prices: pd.Series) -> pd.Series:
        """Return the out-of-sample z-score using formation-window scale only."""

        spread = self.spread(dependent_prices, independent_prices)
        return (spread - self.spread_mean) / self.spread_std


@dataclass(frozen=True)
class PairSignal:
    """A pure signal; it is not an order and carries no execution assumption."""

    timestamp: pd.Timestamp
    zscore: float
    direction: int


def fit_formation_model(
    dependent_prices: pd.Series,
    independent_prices: pd.Series,
    *,
    trend: str = "c",
    min_observations: int = 90,
) -> FormationModel:
    """Fit augmented Engle--Granger and an OLS hedge on a formation window.

    Parameters are intentionally limited to statistically material choices.
    Hyperparameter searches belong in an experiment specification, not hidden
    inside a trading loop.
    """

    y, x = _aligned_log_prices(dependent_prices, independent_prices)
    if len(y) < min_observations:
        raise InvalidFormationWindow(
            f"need at least {min_observations} aligned observations; received {len(y)}"
        )

    design = np.column_stack((np.ones(len(x)), x.to_numpy()))
    alpha, beta = np.linalg.lstsq(design, y.to_numpy(), rcond=None)[0]
    spread = y - alpha - beta * x
    spread_std = float(spread.std(ddof=1))
    if not np.isfinite(spread_std) or spread_std <= 0:
        raise InvalidFormationWindow("formation spread has zero or invalid variance")

    t_stat, pvalue, critical_values = coint(y, x, trend=trend, autolag="aic")
    return FormationModel(
        dependent=str(dependent_prices.name or "y"),
        independent=str(independent_prices.name or "x"),
        alpha=float(alpha),
        beta=float(beta),
        spread_mean=float(spread.mean()),
        spread_std=spread_std,
        coint_t_stat=float(t_stat),
        coint_pvalue=float(pvalue),
        critical_values=tuple(float(value) for value in critical_values),
        formation_start=pd.Timestamp(y.index.min()),
        formation_end=pd.Timestamp(y.index.max()),
        observations=len(y),
    )


def signals_from_zscore(
    zscores: pd.Series,
    *,
    entry_z: float = 2.0,
) -> list[PairSignal]:
    """Emit first-crossing entry signals, without assuming fills or exits.

    Positive spread means y is expensive relative to x, therefore direction -1
    denotes short-y/long-x. The caller must execute no earlier than the next
    tradable observation.
    """

    if entry_z <= 0:
        raise ValueError("entry_z must be positive")

    previous = zscores.shift(1)
    upper_crossing = (zscores >= entry_z) & (previous < entry_z)
    lower_crossing = (zscores <= -entry_z) & (previous > -entry_z)
    signals: list[PairSignal] = []
    for timestamp, zscore in zscores[upper_crossing].items():
        signals.append(PairSignal(pd.Timestamp(timestamp), float(zscore), direction=-1))
    for timestamp, zscore in zscores[lower_crossing].items():
        signals.append(PairSignal(pd.Timestamp(timestamp), float(zscore), direction=1))
    return sorted(signals, key=lambda signal: signal.timestamp)


def _aligned_log_prices(
    dependent_prices: pd.Series,
    independent_prices: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    frame = pd.concat((dependent_prices, independent_prices), axis=1, join="inner").dropna()
    if frame.empty:
        raise InvalidFormationWindow("no aligned non-null prices")
    if (frame <= 0).any().any():
        raise InvalidFormationWindow("prices must be strictly positive before log transformation")
    y = np.log(frame.iloc[:, 0].astype(float))
    x = np.log(frame.iloc[:, 1].astype(float))
    return y, x
