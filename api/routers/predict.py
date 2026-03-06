from fastapi import APIRouter, Depends, HTTPException
from api.schemas import ServerMetrics, PredictionResponse
from api.dependencies import get_predictor
from src.predict import FailurePredictor

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict(
    metrics:   ServerMetrics      = ...,
    predictor: FailurePredictor   = Depends(get_predictor)
):
    try:
        result = predictor.predict(metrics.model_dump())
        return PredictionResponse(
            will_fail       = result.will_fail,
            probability     = result.probability,
            risk_level      = result.risk_level,
            time_to_failure = result.time_to_failure,
            top_causes      = result.top_causes,
            model_version   = result.model_version
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))