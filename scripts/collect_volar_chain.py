"""Collect one read-only Volar BTC option-chain snapshot into the local data lake."""

from __future__ import annotations

import argparse

from quant_pairs.data_lake import LocalDataLake
from quant_pairs.volar_api import VolarClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--at", help="optional ISO-8601 point-in-time request")
    parser.add_argument("--dotenv", default=".env", help="dotenv path; never printed")
    parser.add_argument("--data-root", default="data", help="local data lake root")
    arguments = parser.parse_args()

    chain = VolarClient.from_environment(arguments.dotenv).chain_snapshot("BTC", at=arguments.at)
    path = LocalDataLake(arguments.data_root).write_option_chain_snapshot("volar", "BTC", chain)
    print(f"snapshot_time={chain['timestamp'].iloc[0]} contracts={len(chain)} path={path}")


if __name__ == "__main__":
    main()
