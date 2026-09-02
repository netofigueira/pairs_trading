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


def test_volatility_forecast_endpoint_loads_configured_artifact(tmp_path, monkeypatch) -> None:
    path = tmp_path / "forecast.json"
    path.write_text(json.dumps({"schema_version": 1, "horizons": {"30": {}}}))
    monkeypatch.setenv("QUANT_PAIRS_VOLATILITY_FORECAST", str(path))

    assert "30" in dashboard_api.volatility_forecast()["horizons"]
