"""Core primitives for reproducible market-neutral pairs research."""

from .cointegration import FormationModel, PairSignal, fit_formation_model

__all__ = ["FormationModel", "PairSignal", "fit_formation_model"]
