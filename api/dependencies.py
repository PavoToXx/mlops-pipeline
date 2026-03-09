from functools import lru_cache
from src.predict import FailurePredictor

@lru_cache(maxsize=1)
def get_predictor() -> FailurePredictor:
    """
    Carga el modelo una sola vez en memoria.
    lru_cache garantiza que no se recarga en cada request.
    """
    # Instantiate without loading files immediately so app import and
    # schema validation do not require model artifacts to exist locally.
    return FailurePredictor(
        model_path  = "models/model.pkl",
        scaler_path = "models/scaler.pkl",
        version     = "v1.0.0",
        load_on_init=False
    )