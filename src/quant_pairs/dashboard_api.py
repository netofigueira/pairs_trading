"""Private read-only dashboard for quant data health and research summaries."""

# ruff: noqa: E501

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

app = FastAPI(title="Quant research dashboard", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/market/coverage")
def market_coverage() -> list[dict[str, object]]:
    query = """
        SELECT symbol, interval, count(*) AS bars, min(open_time) AS first_bar,
               max(open_time) AS latest_bar, now() - max(close_time) AS staleness
        FROM market.candle
        WHERE venue = 'binance' AND market_type = 'usdm_perpetual'
        GROUP BY symbol, interval
        ORDER BY symbol, interval
    """
    return _rows(query)


@app.get("/api/v1/research/latest")
def latest_research() -> dict[str, object]:
    run = _rows(
        """SELECT id, started_at, completed_at, data_start, data_end, config
           FROM research.formation_run ORDER BY started_at DESC LIMIT 1"""
    )
    if not run:
        return {"run": None, "candidates": []}
    candidates = _rows(
        """SELECT dependent_symbol, independent_symbol, coint_pvalue, half_life_bars,
                  accepted, diagnostics
           FROM research.candidate WHERE formation_run_id = %s
           ORDER BY accepted DESC, coint_pvalue""",
        (run[0]["id"],),
    )
    return {"run": run[0], "candidates": candidates}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


@app.get("/volatility", response_class=HTMLResponse)
def volatility_page() -> str:
    return (Path(__file__).with_name("static") / "volatility.html").read_text(encoding="utf-8")


@app.get("/paper", response_class=HTMLResponse)
def paper_page() -> str:
    return (Path(__file__).with_name("static") / "paper.html").read_text(encoding="utf-8")


@app.get("/api/v1/paper/overview")
def paper_overview() -> dict[str, object]:
    runs = _rows(
        """SELECT id, as_of, created_at, holdout_id, code_version, status,
                  config, input_quality
           FROM research.paper_run ORDER BY as_of DESC LIMIT 1"""
    )
    if not runs:
        return {"run": None, "decisions": [], "positions": [], "pnl": _empty_paper_pnl()}
    decisions = _rows(
        """SELECT id, decided_at, action, status, reason, forecast_rv, bid_iv,
                  ask_iv, forecast_at, quotes_at, details
           FROM research.paper_decision
           WHERE run_id = %s ORDER BY decided_at DESC""",
        (runs[0]["id"],),
    )
    positions = _rows(
        """SELECT p.id, p.instrument_name, p.leg_type, p.side, p.opened_at,
                  p.closed_at, p.contracts, p.entry_price_btc, p.status,
                  m.marked_at, m.unrealized_pnl_btc, m.realized_pnl_btc,
                  m.margin_estimate_btc
           FROM research.paper_position AS p
           LEFT JOIN LATERAL (
               SELECT marked_at, unrealized_pnl_btc, realized_pnl_btc, margin_estimate_btc
               FROM research.paper_mark WHERE position_id = p.id
               ORDER BY marked_at DESC LIMIT 1
           ) AS m ON TRUE
           WHERE p.status = 'open'
           ORDER BY p.opened_at DESC"""
    )
    pnl_rows = _rows(
        """SELECT COALESCE(sum(m.unrealized_pnl_btc), 0) AS unrealized_pnl_btc,
                  COALESCE(sum(m.realized_pnl_btc), 0) AS realized_pnl_btc,
                  COALESCE(sum(m.margin_estimate_btc), 0) AS margin_estimate_btc
           FROM research.paper_position AS p
           LEFT JOIN LATERAL (
               SELECT unrealized_pnl_btc, realized_pnl_btc, margin_estimate_btc
               FROM research.paper_mark WHERE position_id = p.id
               ORDER BY marked_at DESC LIMIT 1
           ) AS m ON TRUE
           WHERE p.status = 'open'"""
    )
    return {
        "run": runs[0],
        "decisions": decisions,
        "positions": positions,
        "pnl": pnl_rows[0] if pnl_rows else _empty_paper_pnl(),
    }


def _empty_paper_pnl() -> dict[str, float]:
    return {"unrealized_pnl_btc": 0.0, "realized_pnl_btc": 0.0, "margin_estimate_btc": 0.0}


@app.get("/api/v1/volatility/research")
def volatility_research() -> dict[str, object]:
    path = Path(
        os.environ.get("QUANT_PAIRS_VOLATILITY_REPORT", "artifacts/volatility-research-v1.json")
    )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail=f"volatility report unavailable: {error}") from error


@app.get("/api/v1/volatility/forecast")
def volatility_forecast() -> dict[str, object]:
    path = Path(
        os.environ.get(
            "QUANT_PAIRS_VOLATILITY_FORECAST", "artifacts/btc-volatility-forecast-v1.json"
        )
    )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail=f"volatility forecast unavailable: {error}") from error


@app.get("/api/v1/volatility/regime-gate")
def volatility_regime_gate() -> dict[str, object]:
    path = Path(
        os.environ.get(
            "QUANT_PAIRS_VOLATILITY_REGIME_GATE",
            "artifacts/volatility-regime-gate-v1.json",
        )
    )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=503, detail=f"volatility regime gate unavailable: {error}"
        ) from error


@app.get("/api/v1/volatility/refit-cadence")
def volatility_refit_cadence() -> dict[str, object]:
    path = Path(
        os.environ.get(
            "QUANT_PAIRS_VOLATILITY_REFIT_CADENCE",
            "artifacts/garch-refit-cadence-v1.json",
        )
    )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=503, detail=f"GARCH refit comparison unavailable: {error}"
        ) from error


def _rows(query: str, parameters: tuple[object, ...] = ()) -> list[dict[str, object]]:
    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            names = [column.name for column in cursor.description]
            return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _database_url() -> str:
    value = os.environ.get("QUANT_PAIRS_DATABASE_URL")
    if not value:
        raise RuntimeError("QUANT_PAIRS_DATABASE_URL is required")
    return value


_PAGE = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Quant Data Health</title>
<style>body{font-family:system-ui;background:#0b1020;color:#e6edf7;margin:3rem}table{border-collapse:collapse;width:100%;background:#121a2d}th,td{padding:.65rem;text-align:left;border-bottom:1px solid #26324d}h1{margin-bottom:.25rem}.muted{color:#9fb0ca}</style></head><body>
<h1>Quant Data Health</h1><p class="muted">Cobertura de candles fechados no TimescaleDB. <a href="/volatility" style="color:#70d6c8">Abrir pesquisa de volatilidade</a></p><table><thead><tr><th>Ativo</th><th>Intervalo</th><th>Barras</th><th>Primeira</th><th>Última</th><th>Atraso</th></tr></thead><tbody id="rows"></tbody></table>
<script>fetch('/api/v1/market/coverage').then(r=>r.json()).then(rows=>document.querySelector('#rows').innerHTML=rows.map(r=>`<tr><td>${r.symbol}</td><td>${r.interval}</td><td>${r.bars}</td><td>${r.first_bar}</td><td>${r.latest_bar}</td><td>${r.staleness}</td></tr>`).join(''))</script></body></html>"""
