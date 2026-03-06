FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Solo instala lo necesario para la API
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    xgboost \
    scikit-learn \
    pandas \
    numpy \
    joblib \
    pydantic

COPY src/       ./src/
COPY api/       ./api/
COPY models/    ./models/
COPY reports/   ./reports/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
