"""Read-only downloader for public Tardis monthly sample datasets."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

TARDIS_DATASETS_BASE_URL = "https://datasets.tardis.dev/v1"


class TardisDataError(RuntimeError):
    """A Tardis sample could not be retrieved or stored."""


def sample_months(start: str, end: str, *, step: int = 1) -> list[str]:
    """Return inclusive YYYY-MM sample points with a fixed month step."""

    if step <= 0:
        raise ValueError("month step must be positive")
    try:
        start_year, start_month = (int(piece) for piece in start.split("-"))
        end_year, end_month = (int(piece) for piece in end.split("-"))
        date(start_year, start_month, 1)
        date(end_year, end_month, 1)
    except ValueError as error:
        raise ValueError("month must use YYYY-MM") from error
    if (end_year, end_month) < (start_year, start_month):
        raise ValueError("end must not precede start")
    months: list[str] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append(date(year, month, 1).strftime("%Y-%m"))
        for _ in range(step):
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def dataset_url(exchange: str, data_type: str, date: str, symbol: str) -> str:
    """Build the documented CSV.GZ URL for one UTC calendar day."""

    year, month, day = date.split("-")
    if len(year) != 4 or len(month) != 2 or len(day) != 2:
        raise ValueError("date must use YYYY-MM-DD")
    return (
        f"{TARDIS_DATASETS_BASE_URL}/{exchange}/{data_type}/"
        f"{year}/{month}/{day}/{symbol}.csv.gz"
    )


def download_dataset(
    destination_root: Path | str,
    *,
    exchange: str,
    data_type: str,
    date: str,
    symbol: str,
    overwrite: bool = False,
) -> Path:
    """Download one dataset atomically without credentials or shell commands."""

    url = dataset_url(exchange, data_type, date, symbol)
    target = Path(destination_root) / exchange / data_type / date / f"{symbol}.csv.gz"
    if target.is_file() and target.stat().st_size > 0 and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    try:
        with urlopen(url, timeout=120) as response:  # noqa: S310 - fixed Tardis base URL
            with temporary.open("wb") as output:
                while block := response.read(1_048_576):
                    output.write(block)
    except HTTPError as error:
        temporary.unlink(missing_ok=True)
        raise TardisDataError(f"Tardis returned HTTP {error.code} for {url}") from error
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise TardisDataError(f"Tardis download failed for {url}: {error}") from error
    os.replace(temporary, target)
    return target
