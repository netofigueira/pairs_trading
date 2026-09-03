import numpy as np
import pytest

from scripts.run_tape_book_hac import newey_west_tstat


def test_matches_classic_t_for_iid_data() -> None:
    rng = np.random.default_rng(7)
    series = rng.normal(0.5, 1.0, size=5_000)
    classic = series.mean() / (series.std(ddof=0) / len(series) ** 0.5)
    hac = newey_west_tstat(series, lags=14)
    assert hac["t_stat"] == pytest.approx(classic, rel=0.05)


def test_autocorrelation_widens_the_error() -> None:
    rng = np.random.default_rng(11)
    shocks = rng.normal(0.1, 1.0, size=3_000)
    # 14-day moving sum induces the same overlap structure as the book.
    overlapped = np.convolve(shocks, np.ones(14), mode="valid")
    naive_se = overlapped.std(ddof=0) / len(overlapped) ** 0.5
    hac = newey_west_tstat(overlapped, lags=14)
    assert hac["hac_se"] > 2.0 * naive_se


def test_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="too short"):
        newey_west_tstat(np.ones(10), lags=14)
