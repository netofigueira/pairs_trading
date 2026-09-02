from pathlib import Path

from scripts.load_deribit_history_trades import _files, _sha256


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
