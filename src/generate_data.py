import pandas as pd
import numpy as np
import os

def generate_server_metrics(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    n_normal  = int(n_samples * 0.80)
    n_failure = n_samples - n_normal
    n_each    = n_failure // 5   # 5 patrones ahora

    # ── NORMALES ───────────────────────────────────────────────────────
    normal = pd.DataFrame({
        'cpu_usage':       np.random.normal(50, 15, n_normal).clip(10, 75),
        'ram_usage':       np.random.normal(55, 18, n_normal).clip(10, 78),
        'disk_io':         np.random.normal(35, 15, n_normal).clip(5,  70),
        'network_traffic': np.random.normal(230, 90, n_normal).clip(20, 480),
        'temperature':     np.random.normal(58, 10, n_normal).clip(30, 74),
        'cpu_spike_count': np.random.poisson(3, n_normal).clip(0, 10),
        'ram_spike_count': np.random.poisson(2, n_normal).clip(0, 8),
        'uptime_hours':    np.random.uniform(1, 1500, n_normal),
        'failure':         0
    })

    # ── PATRÓN 1: CPU + RAM saturados ─────────────────────────────────
    p1 = pd.DataFrame({
        'cpu_usage':       np.random.normal(88, 7,  n_each).clip(78, 100),
        'ram_usage':       np.random.normal(90, 6,  n_each).clip(80, 100),
        'disk_io':         np.random.normal(45, 15, n_each).clip(10, 80),
        'network_traffic': np.random.normal(280, 80,n_each).clip(50, 550),
        'temperature':     np.random.normal(68, 10, n_each).clip(50, 85),
        'cpu_spike_count': np.random.poisson(8, n_each).clip(3, 18),
        'ram_spike_count': np.random.poisson(6, n_each).clip(2, 16),
        'uptime_hours':    np.random.uniform(100, 2000, n_each),
        'failure':         1
    })

    # ── PATRÓN 2: Temperatura + Disk I/O críticos ──────────────────────
    p2 = pd.DataFrame({
        'cpu_usage':       np.random.normal(60, 15, n_each).clip(30, 85),
        'ram_usage':       np.random.normal(62, 15, n_each).clip(30, 85),
        'disk_io':         np.random.normal(88, 7,  n_each).clip(75, 100),
        'network_traffic': np.random.normal(350, 90,n_each).clip(80, 600),
        'temperature':     np.random.normal(87, 5,  n_each).clip(80, 100),
        'cpu_spike_count': np.random.poisson(4, n_each).clip(0, 12),
        'ram_spike_count': np.random.poisson(3, n_each).clip(0, 10),
        'uptime_hours':    np.random.uniform(500, 2000, n_each),
        'failure':         1
    })

    # ── PATRÓN 3: RAM sola crítica (memoria leak) ──────────────────────
    p3 = pd.DataFrame({
        'cpu_usage':       np.random.normal(48, 15, n_each).clip(15, 75),
        'ram_usage':       np.random.normal(93, 5,  n_each).clip(85, 100),
        'disk_io':         np.random.normal(38, 15, n_each).clip(5,  70),
        'network_traffic': np.random.normal(240, 80,n_each).clip(30, 500),
        'temperature':     np.random.normal(62, 10, n_each).clip(40, 80),
        'cpu_spike_count': np.random.poisson(3, n_each).clip(0, 9),
        'ram_spike_count': np.random.poisson(3, n_each).clip(0, 9),
        'uptime_hours':    np.random.uniform(800, 3000, n_each),
        'failure':         1
    })

    # ── PATRÓN 4: Network saturado + uptime alto ───────────────────────
    p4 = pd.DataFrame({
        'cpu_usage':       np.random.normal(65, 18, n_each).clip(35, 88),
        'ram_usage':       np.random.normal(85, 8,  n_each).clip(70, 100),
        'disk_io':         np.random.normal(60, 18, n_each).clip(20, 90),
        'network_traffic': np.random.normal(580, 70,n_each).clip(430, 800),
        'temperature':     np.random.normal(72, 10, n_each).clip(55, 90),
        'cpu_spike_count': np.random.poisson(5, n_each).clip(1, 14),
        'ram_spike_count': np.random.poisson(4, n_each).clip(1, 12),
        'uptime_hours':    np.random.uniform(1400, 3000, n_each),
        'failure':         1
    })

    # ── PATRÓN 5: Degradación progresiva general ───────────────────────
    n_p5 = n_failure - (n_each * 4)
    p5 = pd.DataFrame({
        'cpu_usage':       np.random.normal(76, 10, n_p5).clip(60, 95),
        'ram_usage':       np.random.normal(79, 10, n_p5).clip(62, 97),
        'disk_io':         np.random.normal(70, 12, n_p5).clip(50, 95),
        'network_traffic': np.random.normal(430, 80,n_p5).clip(250, 680),
        'temperature':     np.random.normal(75, 8,  n_p5).clip(60, 95),
        'cpu_spike_count': np.random.poisson(6, n_p5).clip(2, 16),
        'ram_spike_count': np.random.poisson(5, n_p5).clip(2, 14),
        'uptime_hours':    np.random.uniform(600, 2500, n_p5),
        'failure':         1
    })

    # ── Combina + mezcla ───────────────────────────────────────────────
    df = pd.concat([normal,p1,p2,p3,p4,p5], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── Ruido gaussiano ────────────────────────────────────────────────
    for col in ['cpu_usage','ram_usage','disk_io',
                'network_traffic','temperature']:
        df[col] += np.random.normal(0, 2, len(df))
        df[col]  = df[col].clip(0, 100)

    # ── Feature engineering ────────────────────────────────────────────
    df['cpu_ram_ratio']    = df['cpu_usage'] / (df['ram_usage'] + 1)
    df['thermal_pressure'] = df['temperature'] * df['cpu_usage'] / 100
    df['spike_total']      = df['cpu_spike_count'] + df['ram_spike_count']
    df['io_network_ratio'] = df['disk_io'] / (df['network_traffic'] + 1)

    return df


if __name__ == "__main__":
    df = generate_server_metrics(10000)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/server_metrics.csv", index=False)

    print(f"✅ Dataset: {len(df)} filas")
    print(f"   Normales: {(df['failure']==0).sum()}")
    print(f"   Fallos:   {(df['failure']==1).sum()}")
    print(f"\n📊 Medias por clase:")
    cols = ['cpu_usage','ram_usage','temperature','disk_io','spike_total']
    print(df.groupby('failure')[cols].mean().round(1).to_string())