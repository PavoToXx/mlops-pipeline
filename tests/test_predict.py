import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import FailurePredictor, FEATURE_COLS


def make_predictor():
    with patch('predict.os.path.exists', return_value=True), \
         patch('predict.joblib.load', return_value=MagicMock()):
        return FailurePredictor()


def test_build_features_columns(sample_input):
    predictor = make_predictor()
    df = predictor._build_features(sample_input)
    assert list(df.columns) == FEATURE_COLS
    assert len(df.columns) == 12


def test_build_features_calculations(sample_input):
    predictor = make_predictor()
    raw = {
        'cpu_usage': 80, 'ram_usage': 70, 'temperature': 60,
        'disk_io': 50, 'network_traffic': 100,
        'cpu_spike_count': 3, 'ram_spike_count': 2, 'uptime_hours': 200,
    }
    df = predictor._build_features(raw)
    row = df.iloc[0]
    assert row['cpu_ram_ratio']    == pytest.approx(80 / (70 + 1))
    assert row['thermal_pressure'] == pytest.approx(60 * 80 / 100)
    assert row['spike_total']      == pytest.approx(3 + 2)
    assert row['io_network_ratio'] == pytest.approx(50 / (100 + 1))


def test_get_risk_level():
    predictor = make_predictor()
    assert predictor._get_risk_level(0.90) == "CRITICAL"
    assert predictor._get_risk_level(0.70) == "HIGH"
    assert predictor._get_risk_level(0.50) == "MEDIUM"
    assert predictor._get_risk_level(0.30) == "LOW"
    assert predictor._get_risk_level(0.10) == "NORMAL"


def test_get_time_to_failure():
    predictor = make_predictor()
    assert predictor._get_time_to_failure(0.90) == "< 1 hour"
    assert predictor._get_time_to_failure(0.70) == "1-3 hours"
    assert predictor._get_time_to_failure(0.50) == "3-6 hours"
    assert predictor._get_time_to_failure(0.30) == "6-12 hours"
    assert predictor._get_time_to_failure(0.10) == "> 24 hours"


def test_get_top_causes_critical():
    predictor = make_predictor()
    raw = {
        'cpu_usage': 90, 'ram_usage': 90, 'temperature': 80,
        'disk_io': 85, 'network_traffic': 550,
        'cpu_spike_count': 12, 'ram_spike_count': 9, 'uptime_hours': 1600,
    }
    causes = predictor._get_top_causes(raw)
    assert "high_cpu"            in causes
    assert "high_ram"            in causes
    assert "high_temperature"    in causes
    assert "high_disk_io"        in causes
    assert "network_saturation"  in causes
    assert "cpu_spikes"          in causes
    assert "ram_spikes"          in causes
    assert "long_uptime"         in causes


def test_get_top_causes_normal():
    predictor = make_predictor()
    raw = {
        'cpu_usage': 40, 'ram_usage': 50, 'temperature': 55,
        'disk_io': 30, 'network_traffic': 100,
        'cpu_spike_count': 2, 'ram_spike_count': 1, 'uptime_hours': 100,
    }
    causes = predictor._get_top_causes(raw)
    assert causes == ["no_critical_indicators"]


def test_load_raises_when_model_missing():
    with pytest.raises(FileNotFoundError):
        FailurePredictor(
            model_path='/nonexistent/model.pkl',
            scaler_path='/nonexistent/scaler.pkl',
        )
