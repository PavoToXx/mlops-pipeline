import joblib
import pandas as pd
import os
from dataclasses import dataclass
import warnings

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
        version:     str = "v1.0.0",
        load_on_init: bool = True
    ):
        self.model_path    = model_path
        self.scaler_path   = scaler_path
        self.model_version = version
        self.model  = None
        self.scaler = None
        # By default, load_on_init preserves previous behaviour for tests
        # that expect immediate loading. When set to False the predictor
        # will be created without attempting to read files (useful for
        # app import time or CI environments where a model lives in S3).
        self.load_on_init = load_on_init
        # Do not automatically call _load here; tests or callers may request
        # immediate load by passing load_on_init=False to skip, or True to load.
        # Backwards-compatible default: load on init.
        if self.load_on_init:
            try:
                self._load()
            except Exception:
                # Re-raise to preserve prior behaviour when load_on_init=True
                raise

    def _load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler no encontrado: {self.scaler_path}")
        # Suppress sklearn InconsistentVersionWarning when loading older-serialized estimators
        warnings.filterwarnings(
            "ignore",
            message=r"Trying to unpickle estimator StandardScaler from version .* when using version .*")
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
        if raw['cpu_usage']       > 85:   causes.append("high_cpu")
        if raw['ram_usage']       > 85:   causes.append("high_ram")
        if raw['temperature']     > 78:   causes.append("high_temperature")
        if raw['disk_io']         > 80:   causes.append("high_disk_io")
        if raw['network_traffic'] > 500:  causes.append("network_saturation")
        if raw['cpu_spike_count'] > 10:   causes.append("cpu_spikes")
        if raw['ram_spike_count'] > 8:    causes.append("ram_spikes")
        if raw['uptime_hours']    > 1500: causes.append("long_uptime")
        return causes if causes else ["no_critical_indicators"]

    def predict(self, raw: dict) -> PredictionResult:
        # Ensure model and scaler are loaded before predicting
        if self.model is None or self.scaler is None:
            self._load()

        # For static type checkers: guarantee these are not None beyond this point
        assert self.model is not None and self.scaler is not None, "Modelo o scaler no cargado"

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