import sys
import os
import math
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import feature_engineering, clean_data


def make_df(**kwargs):
    defaults = dict(
        cpu_usage=80, ram_usage=70, disk_io=50,
        network_traffic=100, temperature=60,
        cpu_spike_count=3, ram_spike_count=2,
        uptime_hours=200, failure=0,
    )
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


NEW_COLS = [
    'cpu_ram_ratio', 'thermal_pressure', 'spike_total', 'io_network_ratio',
    'risk_score', 'cpu_critical', 'ram_critical', 'temp_critical', 'multi_critical',
]


def test_feature_engineering_columns():
    df = feature_engineering(make_df())
    for col in NEW_COLS:
        assert col in df.columns, f"Missing column: {col}"


def test_feature_engineering_cpu_ram_ratio():
    df = feature_engineering(make_df(cpu_usage=80, ram_usage=70))
    assert df.iloc[0]['cpu_ram_ratio'] == pytest.approx(80 / (70 + 1))


def test_feature_engineering_thermal_pressure():
    df = feature_engineering(make_df(cpu_usage=80, temperature=60))
    assert df.iloc[0]['thermal_pressure'] == pytest.approx(60 * 80 / 100)


def test_feature_engineering_spike_total():
    df = feature_engineering(make_df(cpu_spike_count=3, ram_spike_count=2))
    assert df.iloc[0]['spike_total'] == pytest.approx(3 + 2)


def test_feature_engineering_critical_flags():
    df = feature_engineering(make_df(cpu_usage=95, ram_usage=92, temperature=85))
    row = df.iloc[0]
    assert row['cpu_critical']  == 1
    assert row['ram_critical']  == 1
    assert row['temp_critical'] == 1
    assert row['multi_critical'] == 3


def test_feature_engineering_no_critical():
    df = feature_engineering(make_df(cpu_usage=50, ram_usage=60, temperature=70))
    row = df.iloc[0]
    assert row['cpu_critical']  == 0
    assert row['ram_critical']  == 0
    assert row['temp_critical'] == 0
    assert row['multi_critical'] == 0


def test_clean_data_removes_duplicates():
    row = dict(
        cpu_usage=80, ram_usage=70, disk_io=50,
        network_traffic=100, temperature=60,
        cpu_spike_count=3, ram_spike_count=2,
        uptime_hours=200, failure=0,
    )
    df = pd.DataFrame([row, row])
    result = clean_data(df)
    assert len(result) == 1


def test_clean_data_removes_nulls():
    row1 = dict(
        cpu_usage=80, ram_usage=70, disk_io=50,
        network_traffic=100, temperature=60,
        cpu_spike_count=3, ram_spike_count=2,
        uptime_hours=200, failure=0,
    )
    row2 = dict(row1)
    row2['cpu_usage'] = float('nan')
    df = pd.DataFrame([row1, row2])
    result = clean_data(df)
    assert len(result) == 1
