from fastapi import APIRouter
from api.schemas import MetricsResponse
import json, os

router = APIRouter()

@router.get("/metrics", response_model=MetricsResponse)
def metrics():
    path = "reports/metrics.json"
    if not os.path.exists(path):
        return MetricsResponse(
            accuracy=0, precision=0,
            recall=0, f1=0, roc_auc=0
        )
    with open(path) as f:
        m = json.load(f)
    return MetricsResponse(**m)