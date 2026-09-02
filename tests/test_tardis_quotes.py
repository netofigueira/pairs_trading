import pandas as pd

from quant_pairs.tardis_quotes import reconstruct_top_of_book


def test_reconstruction_carries_each_book_side_forward(tmp_path) -> None:
    path = tmp_path / "quotes.csv.gz"
    pd.DataFrame(
        {
            "exchange": ["deribit", "deribit", "deribit"],
            "symbol": ["BTC-1JAN27-100000-C"] * 3,
            "timestamp": [1_000_000, 2_000_000, 3_000_000],
            "local_timestamp": [1_100_000, 2_100_000, 3_100_000],
            "ask_amount": [2.0, None, None],
            "ask_price": [0.02, None, 0.021],
            "bid_price": [None, 0.019, None],
            "bid_amount": [None, 3.0, None],
        }
    ).to_csv(path, index=False, compression="gzip")

    books = reconstruct_top_of_book(path, chunk_rows=1)

    assert len(books) == 1
    assert books.loc[0, "bid_price"] == 0.019
    assert books.loc[0, "ask_price"] == 0.021
    assert books.loc[0, "timestamp"] == pd.Timestamp("1970-01-01T00:00:03Z")


def test_reconstruction_can_exclude_stale_books(tmp_path) -> None:
    path = tmp_path / "quotes.csv.gz"
    pd.DataFrame(
        {
            "exchange": ["deribit", "deribit"],
            "symbol": ["STALE", "FRESH"],
            "timestamp": [1_000_000, 3_000_000],
            "local_timestamp": [1_100_000, 3_100_000],
            "ask_amount": [2.0, 2.0],
            "ask_price": [0.02, 0.02],
            "bid_price": [0.019, 0.019],
            "bid_amount": [3.0, 3.0],
        }
    ).to_csv(path, index=False, compression="gzip")

    books = reconstruct_top_of_book(path, max_age=pd.Timedelta(seconds=1))

    assert books["symbol"].tolist() == ["FRESH"]


def test_reconstruction_does_not_let_fresh_ask_mask_stale_bid(tmp_path) -> None:
    path = tmp_path / "quotes.csv.gz"
    pd.DataFrame(
        {
            "exchange": ["deribit", "deribit"],
            "symbol": ["STALE_BID", "STALE_BID"],
            "timestamp": [1_000_000, 10_000_000],
            "local_timestamp": [1_100_000, 10_100_000],
            "ask_amount": [2.0, 4.0],
            "ask_price": [0.02, 0.021],
            "bid_price": [0.019, None],
            "bid_amount": [3.0, None],
        }
    ).to_csv(path, index=False, compression="gzip")

    books = reconstruct_top_of_book(
        path,
        as_of=pd.Timestamp("1970-01-01T00:00:10Z"),
        max_age=pd.Timedelta(seconds=2),
    )

    assert books.empty


def test_reconstruction_rejects_a_missing_book_side(tmp_path) -> None:
    path = tmp_path / "quotes.csv.gz"
    pd.DataFrame(
        {
            "exchange": ["deribit"],
            "symbol": ["BTC-1JAN27-100000-C"],
            "timestamp": [1_000_000],
            "local_timestamp": [1_100_000],
            "ask_amount": [2.0],
            "ask_price": [0.02],
            "bid_price": [None],
            "bid_amount": [None],
        }
    ).to_csv(path, index=False, compression="gzip")

    assert reconstruct_top_of_book(path).empty


def test_as_of_excludes_an_exchange_event_captured_after_decision(tmp_path) -> None:
    path = tmp_path / "quotes.csv.gz"
    pd.DataFrame(
        {
            "exchange": ["deribit", "deribit"],
            "symbol": ["BTC-PERPETUAL", "BTC-PERPETUAL"],
            "timestamp": [1_000_000, 1_900_000],
            "local_timestamp": [1_100_000, 2_100_000],
            "ask_amount": [2.0, 2.0],
            "ask_price": [100.0, 200.0],
            "bid_price": [99.0, 199.0],
            "bid_amount": [3.0, 3.0],
        }
    ).to_csv(path, index=False, compression="gzip")

    books = reconstruct_top_of_book(
        path,
        as_of=pd.Timestamp("1970-01-01T00:00:02Z"),
        max_age=pd.Timedelta(seconds=2),
    )

    assert books.loc[0, "ask_price"] == 100.0
