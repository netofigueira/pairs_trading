import pandas as pd

from quant_pairs.volar_api import VolarClient


def test_client_reads_key_from_dotenv_without_sourcing_it(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("VOLAR_API_KEY='volar_sk_test'\nUNRELATED=$(not-executed)\n")
    calls: list[tuple[str, dict[str, str | int], str]] = []

    def transport(endpoint: str, params: dict[str, str | int], api_key: str) -> dict:
        calls.append((endpoint, params, api_key))
        return {"data": []}

    client = VolarClient.from_environment(dotenv)
    client._transport = transport
    response = client.latest_chain("btc", at="2026-09-01T00:00:00Z")

    assert response == {"data": []}
    assert calls == [
        ("/v1/chains/BTC", {"at": "2026-09-01T00:00:00Z"}, "volar_sk_test")
    ]


def test_chain_snapshot_normalises_envelope_and_contracts() -> None:
    def transport(_: str, __: dict[str, str | int], ___: str) -> dict:
        return {
            "data": {
                "timestamp": "2026-08-25T12:00:00Z",
                "underlying": "BTC",
                "underlying_price": 100_000.0,
                "data": [
                    {
                        "instrument": "BTC-29AUG26-100000-C",
                        "expiry": "2026-08-29T08:00:00Z",
                        "bid_price": 0.01,
                        "ask_price": 0.012,
                        "mark_iv": 0.5,
                        "source": "live_ws",
                    }
                ],
            }
        }

    frame = VolarClient("volar_sk_test", transport=transport).chain_snapshot("btc")

    assert frame.loc[0, "timestamp"] == pd.Timestamp("2026-08-25T12:00:00Z")
    assert frame.loc[0, "underlying"] == "BTC"
    assert frame.loc[0, "ask_price"] == 0.012
