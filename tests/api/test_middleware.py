"""Testes para middleware da API (request ID e latência)."""

from __future__ import annotations

import logging
import uuid
from unittest.mock import patch

from fastapi import status
from fastapi.testclient import TestClient

from src.api.logging import LoggingConfig, setup_logging
from src.api.main import app

_HIGH_CHURN_PROB = 0.85

setup_logging(LoggingConfig(json_format=False))

client = TestClient(app)


def _base_payload() -> dict[str, object]:
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
    }


def test_health_endpoint_still_works() -> None:
    """O endpoint de health deve funcionar apos adicionar o middleware."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "healthy"}


@patch("src.api.main.predict_single", return_value=_HIGH_CHURN_PROB)
def test_predict_endpoint_still_works(mock_predict: object) -> None:
    """O endpoint de predict deve funcionar apos adicionar o middleware."""
    payload = _base_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["churn_probability"] == _HIGH_CHURN_PROB
    assert data["churn_prediction"] is True
    mock_predict.assert_called_once()


def test_request_id_returned_in_response_header() -> None:
    """A resposta deve incluir o cabeçalho X-Request-ID."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert "X-Request-ID" in response.headers
    uuid.UUID(response.headers["X-Request-ID"])


def test_request_id_propagated_from_client() -> None:
    """O X-Request-ID fornecido pelo cliente deve ser preservado."""
    custom_id = "my-custom-trace-id-123"
    response = client.get(
        "/health",
        headers={"X-Request-ID": custom_id},
    )
    assert response.headers["X-Request-ID"] == custom_id


def test_latency_middleware_logs_on_request(caplog: object) -> None:
    """LatencyMiddleware deve registrar a conclusão da requisição."""
    with caplog.at_level(logging.INFO):  # type: ignore[union-attr]
        response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert any(
        "Requisição concluída" in r.message or "Violação de SLO" in r.message
        for r in caplog.records  # type: ignore[union-attr]
    )
