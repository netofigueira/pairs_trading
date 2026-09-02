from pathlib import Path

import pandas as pd

from scripts.load_deribit_history_trades import _files, _sha256, _timestamps


def test_files_filters_daily_cache_by_inclusive_dates(tmp_path: Path) -> None:
    for day in ("2025-01-01", "2025-01-02", "2025-01-03"):
        target = tmp_path / day / "1200-120m.csv.gz"
        target.parent.mkdir()
        target.write_bytes(day.encode())

    assert [path.parent.name for path in _files(tmp_path, "2025-01-02", "2025-01-03")] == [
        "2025-01-02",
        "2025-01-03",
    ]


def test_sha256_is_content_stable(tmp_path: Path) -> None:
    path = tmp_path / "tape.csv.gz"
    path.write_bytes(b"deribit")

    assert _sha256(path) == "3eb7b4d308d81f86a4349ac0203ad88d28e835f07e42ce1c273cfa9b24022454"


def test_timestamps_accept_mixed_iso8601_fractional_precision() -> None:
    values = pd.Series(["2025-01-01 11:20:18+00:00", "2025-01-01 11:20:18.123+00:00"])

    result = _timestamps(values)

    assert str(result.dtype) == "datetime64[us, UTC]"
    assert result.iloc[1] - result.iloc[0] == pd.Timedelta(milliseconds=123)
