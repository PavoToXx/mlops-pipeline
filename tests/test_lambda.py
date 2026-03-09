import sys
import os
import json
import pytest
import numpy as np
from lambda_folder import lambda_function as lf
from unittest.mock import patch, MagicMock

# Mock boto3 before importing lambda_function so the import succeeds without AWS
sys.modules['boto3'] = MagicMock()


FEATURE_COLS = lf.FEATURE_COLS


def test_build_features_columns(sample_input):
    df = lf.build_features(sample_input)
    assert list(df.columns) == FEATURE_COLS
    assert len(df.columns) == 17


def test_build_features_critical_flags():
    raw = dict(
        cpu_usage=95, ram_usage=92, temperature=82,
        disk_io=50, network_traffic=100,
        cpu_spike_count=3, ram_spike_count=2, uptime_hours=200,
    )
    df = lf.build_features(raw)
    row = df.iloc[0]
    assert row['cpu_critical']  == 1
    assert row['ram_critical']  == 1
    assert row['temp_critical'] == 1
    assert row['multi_critical'] == 3


def test_build_features_normal_flags():
    raw = dict(
        cpu_usage=50, ram_usage=60, temperature=70,
        disk_io=30, network_traffic=100,
        cpu_spike_count=2, ram_spike_count=1, uptime_hours=100,
    )
    df = lf.build_features(raw)
    row = df.iloc[0]
    assert row['cpu_critical']  == 0
    assert row['ram_critical']  == 0
    assert row['temp_critical'] == 0
    assert row['multi_critical'] == 0


def test_build_features_risk_score():
    raw = dict(
        cpu_usage=80, ram_usage=70, temperature=60,
        disk_io=50, network_traffic=100,
        cpu_spike_count=3, ram_spike_count=2, uptime_hours=200,
    )
    df = lf.build_features(raw)
    row = df.iloc[0]
    spike_total = 3 + 2
    expected_risk_score = (
        80 * 0.30 + 70 * 0.25 + 60 * 0.20 + 50 * 0.15 + spike_total * 0.10
    )  # = 61.5
    assert row['risk_score'] == pytest.approx(expected_risk_score)


def test_get_risk_level():
    assert lf.get_risk_level(0.90) == "CRITICAL"
    assert lf.get_risk_level(0.70) == "HIGH"
    assert lf.get_risk_level(0.50) == "MEDIUM"
    assert lf.get_risk_level(0.30) == "LOW"
    assert lf.get_risk_level(0.10) == "NORMAL"


def test_lambda_handler_success(sample_input):
    mock_model  = MagicMock()
    mock_scaler = MagicMock()
    mock_model.predict.return_value       = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
    mock_scaler.transform.return_value    = np.zeros((1, len(FEATURE_COLS)))

    event = {'body': sample_input}

    with patch.object(lf, 'load_models'), \
         patch.object(lf, 'model',  mock_model), \
         patch.object(lf, 'scaler', mock_scaler):
        result = lf.lambda_handler(event, {})

    assert result['statusCode'] == 200
    body = json.loads(result['body'])
    assert 'will_fail'   in body
    assert 'probability' in body
    assert 'risk_level'  in body


def test_lambda_handler_error(sample_input):
    with patch.object(lf, 'load_models', side_effect=RuntimeError("S3 error")):
        result = lf.lambda_handler({'body': sample_input}, {})

    assert result['statusCode'] == 500
    body = json.loads(result['body'])
    assert 'error' in body


def test_lambda_handler_body_string(sample_input):
    mock_model  = MagicMock()
    mock_scaler = MagicMock()
    mock_model.predict.return_value       = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
    mock_scaler.transform.return_value    = np.zeros((1, len(FEATURE_COLS)))

    event = {'body': json.dumps(sample_input)}

    with patch.object(lf, 'load_models'), \
         patch.object(lf, 'model',  mock_model), \
         patch.object(lf, 'scaler', mock_scaler):
        result = lf.lambda_handler(event, {})

    assert result['statusCode'] == 200
    body = json.loads(result['body'])
    assert 'will_fail' in body
