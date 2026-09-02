from quant_pairs.tardis import dataset_url, download_dataset, sample_months


def test_dataset_url_matches_tardis_documented_layout() -> None:
    assert dataset_url("deribit", "quotes", "2024-01-01", "OPTIONS") == (
        "https://datasets.tardis.dev/v1/deribit/quotes/2024/01/01/OPTIONS.csv.gz"
    )


def test_download_keeps_an_existing_nonempty_dataset(tmp_path) -> None:
    target = tmp_path / "deribit" / "quotes" / "2024-01-01" / "OPTIONS.csv.gz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"already-downloaded")

    result = download_dataset(
        tmp_path,
        exchange="deribit",
        data_type="quotes",
        date="2024-01-01",
        symbol="OPTIONS",
    )

    assert result == target
    assert target.read_bytes() == b"already-downloaded"


def test_monthly_samples_can_use_a_quarterly_step() -> None:
    assert sample_months("2020-04", "2021-01", step=3) == [
        "2020-04",
        "2020-07",
        "2020-10",
        "2021-01",
    ]
