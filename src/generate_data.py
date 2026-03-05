mport pandas as pd
import numpy as np
import os

def generate_server_metrics(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Genera dataset realista de métricas de servidores.
    ~20% de los servidores fallan (clase desbalanceada, como en producción real)
    """
    np.random.seed(seed)

    # ── Servidores NORMALES (80%) ──────────────────────────────────────
    n_normal = int(n_samples * 0.80)

    normal = pd.DataFrame({
        'cpu_usage':        np.random.normal(45, 15, n_normal).clip(5, 85),
        'ram_usage':        np.random.normal(50, 18, n_normal).clip(10, 85),
        'disk_io':          np.random.normal(30, 12, n_normal).clip(1, 70),
        'network_traffic':  np.random.normal(200, 80, n_normal).clip(10, 450),
        'temperature':      np.random.normal(55, 8, n_normal).clip(30, 70),
        'cpu_spike_count':  np.random.poisson(2, n_normal).clip(0, 8),
        'ram_spike_count':  np.random.poisson(1, n_normal).clip(0, 6),
        'uptime_hours':     np.random.uniform(1, 500, n_normal),
        'failure':          0
    })

    # ── Servidores EN FALLO (20%) ──────────────────────────────────────
    n_failure = n_samples - n_normal

    failure = pd.DataFrame({
        'cpu_usage':        np.random.normal(88, 8, n_failure).clip(70, 100),
        'ram_usage':        np.random.normal(91, 6, n_failure).clip(75, 100),
        'disk_io':          np.random.normal(85, 10, n_failure).clip(60, 100),
        'network_traffic':  np.random.normal(480, 60, n_failure).clip(300, 700),
        'temperature':      np.random.normal(82, 7, n_failure).clip(70, 100),
        'cpu_spike_count':  np.random.poisson(15, n_failure).clip(8, 30),
        'ram_spike_count':  np.random.poisson(12, n_failure).clip(6, 25),
        'uptime_hours':     np.random.uniform(400, 2000, n_failure),
        'failure':          1
    })

    # ── Combina y mezcla ──────────────────────────────────────────────
    df = pd.concat([normal, failure], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── Features adicionales (feature engineering básico) ─────────────
    df['cpu_ram_ratio']     = df['cpu_usage'] / (df['ram_usage'] + 1)
    df['thermal_pressure']  = df['temperature'] * df['cpu_usage'] / 100
    df['spike_total']       = df['cpu_spike_count'] + df['ram_spike_count']
    df['io_network_ratio']  = df['disk_io'] / (df['network_traffic'] + 1)

    return df


if __name__ == "__main__":
    df = generate_server_metrics(10000)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/server_metrics.csv", index=False)

    print(f"✅ Dataset generado: {len(df)} filas")
    print(f"   Servidores normales: {(df['failure'] == 0).sum()}")
    print(f"   Servidores en fallo: {(df['failure'] == 1).sum()}")
    print(f"   Features: {list(df.columns)}")
    print(f"\n{df.describe().round(2)}")
