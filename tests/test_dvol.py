import numpy as np
import pandas as pd
import pytest

from quant_pairs.dvol import build_iv_rv_panel, non_overlapping


def test_panel_uses_only_returns_after_the_dvol_timestamp() -> None:
    timestamps = pd.date_range("2026-01-01T08:00:00Z", periods=33, freq="D")
    prices = pd.DataFrame(
        {"timestamp": timestamps, "close": [100 * 1.01**day for day in range(33)]}
    )
    dvol = pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")], "close": [20.0]})

    panel = build_iv_rv_panel(dvol, prices, horizon_days=30)

    assert len(panel) == 1
    assert panel.loc[0, "iv"] == 0.2
    assert panel.loc[0, "forward_start"] == pd.Timestamp("2026-01-02T08:00:00Z")
    assert panel.loc[0, "forward_end"] == pd.Timestamp("2026-01-31T08:00:00Z")
    assert panel.loc[0, "forward_rv"] == pytest.approx(np.log(1.01) * 365**0.5)


def test_non_overlapping_keeps_one_entry_per_outcome_window() -> None:
    panel = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-15", "2026-01-31"], utc=True),
            "iv": [0.4, 0.5, 0.6],
        }
    )

    selected = non_overlapping(panel, horizon_days=30)

    assert selected["iv"].tolist() == [0.4, 0.6]
