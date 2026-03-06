from fastapi import FastAPI
<<<<<<< HEAD
from fastapi.responses import RedirectResponse
=======
>>>>>>> 585b3f1 (Data local commit)
from api.routers import predict, health, metrics

app = FastAPI(
    title       = "Server Failure Prediction API",
<<<<<<< HEAD
    description = "Predicts if a server will fail based on metrics",
=======
    description = "Predicts if a server will fail in the next hours based on metrics",
>>>>>>> 585b3f1 (Data local commit)
    version     = "1.0.0",
)

app.include_router(health.router,  tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
<<<<<<< HEAD
app.include_router(metrics.router, tags=["Metrics"])

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")  # ← redirige / a /docs
=======
app.include_router(metrics.router, tags=["Metrics"])
>>>>>>> 585b3f1 (Data local commit)
