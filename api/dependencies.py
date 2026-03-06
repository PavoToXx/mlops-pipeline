from functools import lru_cache
from src.predict import FailurePredictor

@lru_cache(maxsize=1)
def get_predictor() -> FailurePredictor:
    """
    Carga el modelo una sola vez en memoria.
    lru_cache garantiza que no se recarga en cada request.
    """
    return FailurePredictor(
        model_path  = "models/model.pkl",
        scaler_path = "models/scaler.pkl",
        version     = "v1.0.0"
    )