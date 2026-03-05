import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(path: str = "data/raw/server_metrics.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Elimina duplicados
    before = len(df)
    df = df.drop_duplicates()
    print(f"✅ Duplicados eliminados: {before - len(df)}")

    # Elimina nulos
    df = df.dropna()
    print(f"✅ Filas con nulos eliminadas: {before - len(df)}")

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Risk score compuesto
    df['risk_score'] = (
        df['cpu_usage'] * 0.30 +
        df['ram_usage'] * 0.25 +
        df['temperature'] * 0.20 +
        df['disk_io'] * 0.15 +
        df['spike_total'] * 0.10
    )

    # Flags binarios de umbrales críticos
    df['cpu_critical']   = (df['cpu_usage'] > 90).astype(int)
    df['ram_critical']   = (df['ram_usage'] > 90).astype(int)
    df['temp_critical']  = (df['temperature'] > 80).astype(int)
    df['multi_critical'] = (
        df['cpu_critical'] + df['ram_critical'] + df['temp_critical']
    )

    print(f"✅ Features creados: risk_score, cpu_critical, ram_critical,")
    print(f"   temp_critical, multi_critical")
    return df

def split_and_scale(df: pd.DataFrame):
    feature_cols = [
        'cpu_usage', 'ram_usage', 'disk_io', 'network_traffic',
        'temperature', 'cpu_spike_count', 'ram_spike_count',
        'uptime_hours', 'cpu_ram_ratio', 'thermal_pressure',
        'spike_total', 'io_network_ratio', 'risk_score',
        'cpu_critical', 'ram_critical', 'temp_critical', 'multi_critical'
    ]

    X = df[feature_cols]
    y = df['failure']

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Escala features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols
    )

    print(f"✅ Train: {len(X_train)} filas | Test: {len(X_test)} filas")
    print(f"✅ Clase 0 (normal) en train:  {(y_train == 0).sum()}")
    print(f"✅ Clase 1 (fallo)  en train:  {(y_train == 1).sum()}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_processed_data(X_train, X_test, y_train, y_test, scaler):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv",   index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv",   index=False)

    joblib.dump(scaler, "models/scaler.pkl")

    print(f"✅ Datos guardados en data/processed/")
    print(f"✅ Scaler guardado en models/scaler.pkl")

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    save_processed_data(X_train, X_test, y_train, y_test, scaler)
    print("\n✅ Preprocesamiento completo")