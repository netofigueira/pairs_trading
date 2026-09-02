from quant_pairs.tardis import dataset_url


def test_dataset_url_matches_tardis_documented_layout() -> None:
    assert dataset_url("deribit", "quotes", "2024-01-01", "OPTIONS") == (
        "https://datasets.tardis.dev/v1/deribit/quotes/2024/01/01/OPTIONS.csv.gz"
    )
