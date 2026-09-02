"""Check read-only access to Volar without exposing the API key."""

from __future__ import annotations

import argparse
import json

from quant_pairs.volar_api import VolarAPIError, VolarClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlying", default="BTC", choices=("BTC",))
    parser.add_argument("--at", help="optional ISO-8601 point-in-time request")
    parser.add_argument("--dotenv", default=".env", help="dotenv path; never printed")
    arguments = parser.parse_args()

    try:
        response = VolarClient.from_environment(arguments.dotenv).latest_chain(
            arguments.underlying, at=arguments.at
        )
    except VolarAPIError as error:
        raise SystemExit(str(error)) from error
    data = response.get("data")
    nested_lengths = (
        {key: len(value) for key, value in data.items() if isinstance(value, list)}
        if isinstance(data, dict)
        else {}
    )
    chain_rows = data.get("data") if isinstance(data, dict) else None
    row_fields = (
        sorted(chain_rows[0])
        if isinstance(chain_rows, list) and chain_rows and isinstance(chain_rows[0], dict)
        else None
    )
    print(
        json.dumps(
            {
                "top_level_fields": sorted(response),
                "data_type": type(data).__name__,
                "data_rows": len(data) if isinstance(data, list) else None,
                "data_fields": sorted(data) if isinstance(data, dict) else None,
                "nested_list_lengths": nested_lengths,
                "chain_row_fields": row_fields,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
