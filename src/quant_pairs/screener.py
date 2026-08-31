"""Formation-window pair discovery with FDR and mean-reversion filters."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .cointegration import FormationModel, InvalidFormationWindow, fit_formation_model


@dataclass(frozen=True)
class ScreenedPair:
    model: FormationModel
    half_life_bars: float | None
    fdr_qvalue: float
    accepted: bool


def screen_pairs(
    prices: pd.DataFrame,
    *,
    fdr_alpha: float = 0.05,
    min_half_life_bars: float = 4,
    max_half_life_bars: float = 72,
) -> list[ScreenedPair]:
    """Test each unordered pair once, then apply Benjamini--Hochberg FDR.

    Each result is a hypothesis; candidates only pass when their adjusted
    q-value and formation-window half-life pass the predeclared gates.
    """

    if not 0 < fdr_alpha < 1:
        raise ValueError("fdr_alpha must be between zero and one")
    models: list[FormationModel] = []
    half_lives: list[float | None] = []
    for dependent, independent in combinations(prices.columns, 2):
        try:
            model = fit_formation_model(prices[dependent], prices[independent])
        except InvalidFormationWindow:
            continue
        models.append(model)
        half_lives.append(_half_life(model, prices[dependent], prices[independent]))

    qvalues = benjamini_hochberg([model.coint_pvalue for model in models])
    results = []
    for model, half_life, qvalue in zip(models, half_lives, qvalues, strict=True):
        accepted = (
            qvalue <= fdr_alpha
            and half_life is not None
            and min_half_life_bars <= half_life <= max_half_life_bars
        )
        results.append(ScreenedPair(model, half_life, qvalue, accepted))
    return sorted(results, key=lambda item: (not item.accepted, item.fdr_qvalue))


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Return monotone BH-adjusted q-values in the original order."""

    if not pvalues:
        return []
    values = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite values in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    qvalues = np.empty_like(adjusted)
    qvalues[order] = np.minimum(adjusted, 1.0)
    return qvalues.tolist()


def _half_life(model: FormationModel, y: pd.Series, x: pd.Series) -> float | None:
    spread = model.spread(y, x).dropna()
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna().reindex(lagged.index)
    design = np.column_stack((np.ones(len(lagged)), lagged.to_numpy()))
    _, speed = np.linalg.lstsq(design, delta.to_numpy(), rcond=None)[0]
    if not np.isfinite(speed) or speed >= 0:
        return None
    half_life = -np.log(2) / speed
    return float(half_life) if np.isfinite(half_life) and half_life > 0 else None
