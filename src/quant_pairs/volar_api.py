"""Authenticated, read-only client for Volar historical options data."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

VOLAR_BASE_URL = "https://api.volardata.com"
Transport = Callable[[str, Mapping[str, str | int], str], dict[str, Any]]


class VolarAPIError(RuntimeError):
    """The Volar API returned an invalid or unusable response."""


class VolarClient:
    """Read historical Volar data; this adapter has no trading capability."""

    def __init__(self, api_key: str, transport: Transport | None = None) -> None:
        if not api_key:
            raise ValueError("VOLAR_API_KEY must not be empty")
        self._api_key = api_key
        self._transport = transport or _http_transport

    @classmethod
    def from_environment(cls, dotenv_path: Path | str = ".env") -> VolarClient:
        """Load the key without sourcing the dotenv file or printing its value."""

        api_key = os.environ.get("VOLAR_API_KEY") or _dotenv_value(dotenv_path, "VOLAR_API_KEY")
        if api_key is None:
            raise VolarAPIError("VOLAR_API_KEY is not set in the environment or dotenv file")
        return cls(api_key)

    def latest_chain(self, underlying: str, *, at: str | None = None) -> dict[str, Any]:
        """Retrieve the current chain, or a point-in-time chain if the plan permits it."""

        params: dict[str, str | int] = {}
        if at is not None:
            params["at"] = at
        payload = self._transport(f"/v1/chains/{underlying.upper()}", params, self._api_key)
        if not isinstance(payload, dict):
            raise VolarAPIError("Volar response is not a JSON object")
        return payload

    def chain_snapshot(self, underlying: str, *, at: str | None = None) -> pd.DataFrame:
        """Return a normalized chain snapshot, preserving exchange provenance."""

        payload = self.latest_chain(underlying, at=at)
        envelope = payload.get("data")
        if not isinstance(envelope, dict) or not isinstance(envelope.get("data"), list):
            raise VolarAPIError("Volar chain response is missing its data rows")
        rows = envelope["data"]
        if not all(isinstance(row, dict) for row in rows):
            raise VolarAPIError("Volar chain rows are not objects")
        frame = pd.DataFrame(rows)
        required = {"instrument", "expiry", "bid_price", "ask_price", "mark_iv", "source"}
        missing = required.difference(frame.columns)
        if missing:
            raise VolarAPIError(f"Volar chain is missing fields: {sorted(missing)}")
        frame.insert(0, "timestamp", pd.to_datetime(envelope.get("timestamp"), utc=True))
        frame.insert(1, "underlying", str(envelope.get("underlying", underlying.upper())).upper())
        frame.insert(
            2,
            "underlying_price",
            pd.to_numeric(envelope.get("underlying_price"), errors="coerce"),
        )
        frame["expiry"] = pd.to_datetime(frame["expiry"], utc=True, errors="raise")
        for column in ("bid_price", "ask_price", "mark_iv"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.sort_values("instrument").reset_index(drop=True)


def _http_transport(
    endpoint: str, params: Mapping[str, str | int], api_key: str
) -> dict[str, Any]:
    suffix = f"?{urlencode(params)}" if params else ""
    request = Request(
        f"{VOLAR_BASE_URL}{endpoint}{suffix}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "quant-pairs-research/0.1 (read-only)",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed official base URL
            payload = json.load(response)
    except HTTPError as error:
        detail = _safe_error_detail(error, api_key)
        message = f"Volar request failed for {endpoint}: HTTP {error.code}"
        if detail:
            message = f"{message} ({detail})"
        raise VolarAPIError(message) from error
    except OSError as error:
        raise VolarAPIError(f"Volar request failed for {endpoint}: {error}") from error
    if not isinstance(payload, dict):
        raise VolarAPIError("Volar response is not a JSON object")
    return payload


def _dotenv_value(path: Path | str, key: str) -> str | None:
    dotenv = Path(path)
    if not dotenv.is_file():
        return None
    prefix = f"{key}="
    for line in dotenv.read_text().splitlines():
        candidate = line.strip()
        if candidate.startswith("export "):
            candidate = candidate.removeprefix("export ").lstrip()
        if candidate.startswith(prefix):
            value = candidate.removeprefix(prefix).strip()
            return value.strip("'\"") or None
    return None


def _safe_error_detail(error: HTTPError, api_key: str) -> str | None:
    """Return an API diagnostic while defensively redacting the bearer token."""

    try:
        payload = json.load(error)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail") or payload.get("message") or payload.get("error")
    if not isinstance(detail, str):
        return None
    return detail.replace(api_key, "[redacted]")[:500]
