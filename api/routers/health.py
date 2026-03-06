from fastapi import APIRouter, Depends
from api.schemas import HealthResponse
from api.dependencies import get_predictor

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health(predictor=Depends(get_predictor)):
    return HealthResponse(
        status       = "ok",
        model_loaded = predictor.model is not None,
        version      = predictor.model_version
    )