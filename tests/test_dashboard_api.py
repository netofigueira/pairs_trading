import json

from quant_pairs import dashboard_api


def test_volatility_endpoint_loads_the_configured_compact_report(tmp_path, monkeypatch) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"schema_version": 1, "carry": {"points": []}}))
    monkeypatch.setenv("QUANT_PAIRS_VOLATILITY_REPORT", str(path))

    assert dashboard_api.volatility_research()["schema_version"] == 1


def test_volatility_page_is_self_contained() -> None:
    page = dashboard_api.volatility_page()

    assert "IV × RV futura" in page
    assert "<svg" in page
    assert "cdn." not in page
    assert "Forecast diário" in page
    assert "Gate long / short / flat congelado" in page
    assert "Walk-forward: frequência de refit" in page


def test_paper_page_is_self_contained() -> None:
    page = dashboard_api.paper_page()

    assert "BTC volatility cockpit" in page
    assert "/api/v1/paper/overview" in page


def test_paper_overview_returns_latest_operational_snapshot(monkeypatch) -> None:
    responses = iter(
        [
            [{"id": "run-1", "status": "blocked", "holdout_id": "volatility-live-v1"}],
            [{"action": "flat", "status": "blocked", "reason": "quotes unavailable"}],
            [],
            [{"unrealized_pnl_btc": 0.0, "realized_pnl_btc": 0.0, "margin_estimate_btc": 0.0}],
        ]
    )
    monkeypatch.setattr(dashboard_api, "_rows", lambda *_args, **_kwargs: next(responses))

    overview = dashboard_api.paper_overview()

    assert overview["run"]["id"] == "run-1"
    assert overview["decisions"][0]["status"] == "blocked"
    assert overview["positions"] == []


def test_volatility_forecast_endpoint_loads_configured_artifact(tmp_path, monkeypatch) -> None:
    path = tmp_path / "forecast.json"
    path.write_text(json.dumps({"schema_version": 1, "horizons": {"30": {}}}))
    monkeypatch.setenv("QUANT_PAIRS_VOLATILITY_FORECAST", str(path))

    assert "30" in dashboard_api.volatility_forecast()["horizons"]


def test_volatility_regime_gate_loads_configured_artifact(tmp_path, monkeypatch) -> None:
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"schema_version": 1, "decision": "do_not_promote"}))
    monkeypatch.setenv("QUANT_PAIRS_VOLATILITY_REGIME_GATE", str(path))

    assert dashboard_api.volatility_regime_gate()["decision"] == "do_not_promote"


def test_volatility_refit_cadence_loads_configured_artifact(tmp_path, monkeypatch) -> None:
    path = tmp_path / "refit.json"
    path.write_text(json.dumps({"schema_version": 1, "selected_refit_cadence": "monthly_refit"}))
    monkeypatch.setenv("QUANT_PAIRS_VOLATILITY_REFIT_CADENCE", str(path))

    assert dashboard_api.volatility_refit_cadence()["selected_refit_cadence"] == "monthly_refit"
