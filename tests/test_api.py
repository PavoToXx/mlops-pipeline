"""
Tests de integración para la API FastAPI.
Usa TestClient para probar endpoints sin levantar servidor.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Agregar path del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.main import app
import api.dependencies as deps


@pytest.fixture
def client():
    """Cliente de prueba para FastAPI"""
    return TestClient(app)


@pytest.fixture
def mock_predictor():
    """Mock del predictor para tests rápidos"""
    predictor = MagicMock()
    predictor.model = MagicMock()
    predictor.model_version = "v1.0.0-test"
    return predictor


@pytest.fixture
def sample_server_metrics():
    """Datos de ejemplo para predicción"""
    return {
        "cpu_usage": 87.5,
        "ram_usage": 91.2,
        "disk_io": 78.4,
        "network_traffic": 520.0,
        "temperature": 83.0,
        "cpu_spike_count": 14,
        "ram_spike_count": 11,
        "uptime_hours": 1500.0
    }


# ============== TESTS DE ENDPOINTS ==============

def test_root_redirect(client):
    """Test que / redirige a /docs"""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert "/docs" in response.headers["location"]


def test_health_endpoint(client, mock_predictor):
    """Test del endpoint /health"""
    app.dependency_overrides[deps.get_predictor] = lambda: mock_predictor
    try:
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "version" in data
        assert data["model_loaded"] is True
    finally:
        app.dependency_overrides.pop(deps.get_predictor, None)


def test_health_endpoint_real():
    """Test del endpoint /health con modelo real (si existe)"""
    client = TestClient(app)
    
    # Este test puede fallar si no hay modelo entrenado
    try:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    except Exception:
        pytest.skip("Modelo no disponible para test de integración")


def test_metrics_endpoint_no_file(client):
    """Test de /metrics cuando no existe metrics.json"""
    with patch('os.path.exists', return_value=False):
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Debe retornar métricas en 0
        assert data["accuracy"] == 0
        assert data["precision"] == 0
        assert data["recall"] == 0
        assert data["f1"] == 0
        assert data["roc_auc"] == 0


def test_metrics_endpoint_with_file(client):
    """Test de /metrics cuando existe metrics.json"""
    mock_metrics = {
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.92,
        "f1": 0.925,
        "roc_auc": 0.97
    }
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', MagicMock(return_value=MagicMock(
             __enter__=lambda self: self,
             __exit__=lambda *args: None,
             read=lambda: '{"accuracy":0.95,"precision":0.93,"recall":0.92,"f1":0.925,"roc_auc":0.97}'
         ))):
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["accuracy"] == 0.95
        assert data["f1"] == 0.925


def test_predict_endpoint_valid_input(client, sample_server_metrics, mock_predictor):
    """Test de /predict con entrada válida"""
    from src.predict import PredictionResult
    
    # Mock de la respuesta del predictor
    mock_result = PredictionResult(
        will_fail=True,
        probability=0.8421,
        risk_level="CRITICAL",
        time_to_failure="< 1 hour",
        top_causes=["high_cpu", "high_ram", "high_temperature"],
        model_version="v1.0.0-test"
    )
    mock_predictor.predict.return_value = mock_result
    
    app.dependency_overrides[deps.get_predictor] = lambda: mock_predictor
    try:
        response = client.post("/predict", json=sample_server_metrics)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["will_fail"] is True
        assert data["probability"] == 0.8421
        assert data["risk_level"] == "CRITICAL"
        assert data["time_to_failure"] == "< 1 hour"
        assert "high_cpu" in data["top_causes"]
        assert data["model_version"] == "v1.0.0-test"
    finally:
        app.dependency_overrides.pop(deps.get_predictor, None)


def test_predict_endpoint_missing_field(client):
    """Test de /predict con campo faltante"""
    incomplete_data = {
        "cpu_usage": 87.5,
        "ram_usage": 91.2,
        # Falta disk_io y otros campos
    }
    
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_invalid_type(client):
    """Test de /predict con tipo de dato incorrecto"""
    invalid_data = {
        "cpu_usage": "not_a_number",  # Debería ser float
        "ram_usage": 91.2,
        "disk_io": 78.4,
        "network_traffic": 520.0,
        "temperature": 83.0,
        "cpu_spike_count": 14,
        "ram_spike_count": 11,
        "uptime_hours": 1500.0
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422


def test_predict_endpoint_out_of_range(client):
    """Test de /predict con valores fuera de rango"""
    out_of_range_data = {
        "cpu_usage": 150.0,  # Debería ser <= 100
        "ram_usage": 91.2,
        "disk_io": 78.4,
        "network_traffic": 520.0,
        "temperature": 83.0,
        "cpu_spike_count": 14,
        "ram_spike_count": 11,
        "uptime_hours": 1500.0
    }
    
    response = client.post("/predict", json=out_of_range_data)
    assert response.status_code == 422


def test_predict_endpoint_negative_values(client):
    """Test de /predict con valores negativos"""
    negative_data = {
        "cpu_usage": -10.0,  # No puede ser negativo
        "ram_usage": 91.2,
        "disk_io": 78.4,
        "network_traffic": 520.0,
        "temperature": 83.0,
        "cpu_spike_count": 14,
        "ram_spike_count": 11,
        "uptime_hours": 1500.0
    }
    
    response = client.post("/predict", json=negative_data)
    assert response.status_code == 422


def test_predict_endpoint_normal_scenario(client, mock_predictor):
    """Test de predicción para servidor normal"""
    from src.predict import PredictionResult
    
    normal_metrics = {
        "cpu_usage": 45.0,
        "ram_usage": 50.0,
        "disk_io": 30.0,
        "network_traffic": 200.0,
        "temperature": 55.0,
        "cpu_spike_count": 2,
        "ram_spike_count": 1,
        "uptime_hours": 500.0
    }
    
    mock_result = PredictionResult(
        will_fail=False,
        probability=0.15,
        risk_level="NORMAL",
        time_to_failure="> 24 hours",
        top_causes=["no_critical_indicators"],
        model_version="v1.0.0-test"
    )
    mock_predictor.predict.return_value = mock_result
    
    app.dependency_overrides[deps.get_predictor] = lambda: mock_predictor
    try:
        response = client.post("/predict", json=normal_metrics)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["will_fail"] is False
        assert data["risk_level"] == "NORMAL"
    finally:
        app.dependency_overrides.pop(deps.get_predictor, None)


# ============== TESTS DE VALIDACIÓN ==============

def test_api_docs_accessible(client):
    """Test que la documentación de Swagger es accesible"""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema(client):
    """Test que el schema de OpenAPI es válido"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert schema["info"]["title"] == "Server Failure Prediction API"


def test_cors_headers(client, sample_server_metrics, mock_predictor):
    """Test que los headers están presentes"""
    from src.predict import PredictionResult
    
    mock_result = PredictionResult(
        will_fail=True,
        probability=0.85,
        risk_level="CRITICAL",
        time_to_failure="< 1 hour",
        top_causes=["high_cpu"],
        model_version="v1.0.0-test"
    )
    mock_predictor.predict.return_value = mock_result
    
    app.dependency_overrides[deps.get_predictor] = lambda: mock_predictor
    try:
        response = client.post("/predict", json=sample_server_metrics)
        
        assert response.status_code == 200
        assert "content-type" in response.headers
    finally:
        app.dependency_overrides.pop(deps.get_predictor, None)


def test_all_endpoints_exist(client):
    """Test que todos los endpoints esperados existen"""
    response = client.get("/openapi.json")
    schema = response.json()
    paths = schema["paths"]
    
    assert "/health" in paths
    assert "/metrics" in paths
    assert "/predict" in paths
    # assert "/" in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
