import joblib
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass

# ← Mismo orden exacto que TRAIN_FEATURES en preprocess.py
FEATURE_COLS = [
    'cpu_usage', 'ram_usage', 'disk_io',
    'network_traffic', 'temperature',
    'cpu_spike_count', 'ram_spike_count',
    'uptime_hours',
    'cpu_ram_ratio', 'thermal_pressure',
    'spike_total', 'io_network_ratio',
]

@dataclass
class PredictionResult:
    will_fail:        bool
    probability:      float
    risk_level:       str
    time_to_failure:  str
    top_causes:       list
    model_version:    str

class FailurePredictor:
    def __init__(
        self,
        model_path:  str = "models/model.pkl",
        scaler_path: str = "models/scaler.pkl",
        version:     str = "v1.0.0"
    ):
        self.model_path    = model_path
        self.scaler_path   = scaler_path
        self.model_version = version
        self.model  = None
        self.scaler = None
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler no encontrado: {self.scaler_path}")
        self.model  = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        print(f"✅ Modelo cargado: {self.model_path}")
        print(f"✅ Scaler cargado: {self.scaler_path}")

    def _build_features(self, raw: dict) -> pd.DataFrame:
        cpu   = raw['cpu_usage']
        ram   = raw['ram_usage']
        temp  = raw['temperature']
        net   = raw['network_traffic']
        disk  = raw['disk_io']
        cpu_s = raw['cpu_spike_count']
        ram_s = raw['ram_spike_count']

        features = {
            'cpu_usage':        cpu,
            'ram_usage':        ram,
            'disk_io':          disk,
            'network_traffic':  net,
            'temperature':      temp,
            'cpu_spike_count':  cpu_s,
            'ram_spike_count':  ram_s,
            'uptime_hours':     raw['uptime_hours'],
            'cpu_ram_ratio':    cpu / (ram + 1),
            'thermal_pressure': temp * cpu / 100,
            'spike_total':      cpu_s + ram_s,
            'io_network_ratio': disk / (net + 1),
        }
        return pd.DataFrame([features])[FEATURE_COLS]

    def _get_risk_level(self, p: float) -> str:
        if p >= 0.80:   return "CRITICAL"
        elif p >= 0.60: return "HIGH"
        elif p >= 0.40: return "MEDIUM"
        elif p >= 0.20: return "LOW"
        else:           return "NORMAL"

    def _get_time_to_failure(self, p: float) -> str:
        if p >= 0.80:   return "< 1 hour"
        elif p >= 0.60: return "1-3 hours"
        elif p >= 0.40: return "3-6 hours"
        elif p >= 0.20: return "6-12 hours"
        else:           return "> 24 hours"

    def _get_top_causes(self, raw: dict) -> list:
        causes = []
        if raw['cpu_usage']       > 85:  causes.append("high_cpu")
        if raw['ram_usage']       > 85:  causes.append("high_ram")
        if raw['temperature']     > 78:  causes.append("high_temperature")
        if raw['disk_io']         > 80:  causes.append("high_disk_io")
        if raw['network_traffic'] > 500: causes.append("network_saturation")
        if raw['cpu_spike_count'] > 10:  causes.append("cpu_spikes")
        if raw['ram_spike_count'] > 8:   causes.append("ram_spikes")
        if raw['uptime_hours']    > 1500: causes.append("long_uptime")
        return causes if causes else ["no_critical_indicators"]

    def predict(self, raw: dict) -> PredictionResult:
        df        = self._build_features(raw)
        df_scaled = pd.DataFrame(
            self.scaler.transform(df), columns=FEATURE_COLS
        )
        will_fail   = bool(self.model.predict(df_scaled)[0])
        probability = round(float(
            self.model.predict_proba(df_scaled)[0][1]
        ), 4)

        return PredictionResult(
            will_fail       = will_fail,
            probability     = probability,
            risk_level      = self._get_risk_level(probability),
            time_to_failure = self._get_time_to_failure(probability),
            top_causes      = self._get_top_causes(raw),
            model_version   = self.model_version
        )

if __name__ == "__main__":
    predictor = FailurePredictor()

    casos = {
        "Normal":           {"cpu_usage":42, "ram_usage":55, "disk_io":30,
                             "network_traffic":180, "temperature":52,
                             "cpu_spike_count":3, "ram_spike_count":2,
                             "uptime_hours":240},
        "Bajo estrés":      {"cpu_usage":75, "ram_usage":78, "disk_io":65,
                             "network_traffic":380, "temperature":70,
                             "cpu_spike_count":8, "ram_spike_count":7,
                             "uptime_hours":900},
        "RAM crítica sola": {"cpu_usage":45, "ram_usage":92, "disk_io":35,
                             "network_traffic":200, "temperature":58,
                             "cpu_spike_count":4, "ram_spike_count":3,
                             "uptime_hours":400},
        "Temp alta sola":   {"cpu_usage":50, "ram_usage":60, "disk_io":40,
                             "network_traffic":220, "temperature":83,
                             "cpu_spike_count":5, "ram_spike_count":4,
                             "uptime_hours":600},
        "CRÍTICO":          {"cpu_usage":94, "ram_usage":96, "disk_io":88,
                             "network_traffic":560, "temperature":84,
                             "cpu_spike_count":18, "ram_spike_count":15,
                             "uptime_hours":1800},
    }

    print("\n" + "="*50)
    for nombre, caso in casos.items():
        r = predictor.predict(caso)
        print(f"\n🔶 {nombre}:")
        print(f"   probability: {r.probability} | {r.risk_level}")
        print(f"   will_fail:   {r.will_fail}")