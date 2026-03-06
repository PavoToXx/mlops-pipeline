import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── Features para ENTRENAR el modelo ──────────────────────────────────
# Solo métricas raw + features físicamente interpretables
# SIN risk_score ni multi_critical (evitan que el modelo haga trampa)
TRAIN_FEATURES = [
    'cpu_usage', 'ram_usage', 'disk_io',
    'network_traffic', 'temperature',
    'cpu_spike_count', 'ram_spike_count',
    'uptime_hours',
    'cpu_ram_ratio',       # ratio físico
    'thermal_pressure',    # física real: temp * cpu
    'spike_total',         # suma simple
    'io_network_ratio',    # ratio físico
]

def load_data(path: str = "data/raw/server_metrics.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().dropna()
    print(f"✅ Filas eliminadas: {before - len(df)}")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['cpu_ram_ratio']    = df['cpu_usage'] / (df['ram_usage'] + 1)
    df['thermal_pressure'] = df['temperature'] * df['cpu_usage'] / 100
    df['spike_total']      = df['cpu_spike_count'] + df['ram_spike_count']
    df['io_network_ratio'] = df['disk_io'] / (df['network_traffic'] + 1)

    # Estos se calculan pero NO van al modelo
    # Se usan solo en la API para explicabilidad
    df['risk_score']    = (
        df['cpu_usage']  * 0.30 +
        df['ram_usage']  * 0.25 +
        df['temperature']* 0.20 +
        df['disk_io']    * 0.15 +
        df['spike_total']* 0.10
    )
    df['cpu_critical']   = (df['cpu_usage']   > 90).astype(int)
    df['ram_critical']   = (df['ram_usage']   > 90).astype(int)
    df['temp_critical']  = (df['temperature'] > 80).astype(int)
    df['multi_critical'] = (
        df['cpu_critical'] + df['ram_critical'] + df['temp_critical']
    )

    print(f"✅ Features calculados")
    print(f"   Para entrenamiento: {TRAIN_FEATURES}")
    print(f"   Solo API (excluidos del modelo): "
          f"risk_score, multi_critical, *_critical")
    return df

def split_and_scale(df: pd.DataFrame):
    X = df[TRAIN_FEATURES]   # ← solo los 12 features limpios
    y = df['failure']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=TRAIN_FEATURES
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=TRAIN_FEATURES
    )

    print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"✅ Clase 0: {(y_train==0).sum()} | Clase 1: {(y_train==1).sum()}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_processed_data(X_train, X_test, y_train, y_test, scaler):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv",   index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv",   index=False)
    joblib.dump(scaler, "models/scaler.pkl")

    print(f"✅ Datos guardados | Scaler guardado")

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    save_processed_data(X_train, X_test, y_train, y_test, scaler)
    print("\n✅ Preprocesamiento completo")