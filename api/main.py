from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from api.routers import predict, health, metrics

app = FastAPI(
    title       = "Server Failure Prediction API",
    description = "Predicts if a server will fail based on metrics",
    version     = "1.0.0",
)

app.include_router(health.router,  tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(metrics.router, tags=["Metrics"])

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")  # ← redirige / a /docs