# MLOps Pipeline - Server Failure Prediction

Proyecto de MLOps para entrenar, evaluar y servir un modelo de clasificacion que predice fallas de servidor a partir de metricas operativas.

## Que incluye

- Pipeline de datos y entrenamiento (`src/`)
- API REST con FastAPI (`api/`)
- Empaquetado con Docker (`Dockerfile`, `docker-compose.yml`)
- Variante de inferencia en AWS Lambda (`lambda/`)
- CI y CD con GitHub Actions (`.github/workflows/ci.yml`, `.github/workflows/cd.yml`)

## Objetivo

Predecir si un servidor fallara en las proximas horas, entregando:
- Probabilidad de falla
- Nivel de riesgo
- Tiempo estimado a falla
- Causas probables

## Estructura del proyecto

```text
api/                # FastAPI app, dependencias y routers
src/                # Generacion de datos, preprocesamiento, entrenamiento y evaluacion
data/               # Datasets raw y procesados
models/             # Artefactos del modelo (no versionados)
reports/            # Metricas y reportes
lambda/             # Funcion y contenedor para AWS Lambda
monitoring/         # Espacio para monitoreo/observabilidad
tests/              # Tests unitarios del pipeline, predictor y lambda
```

## Stack tecnologico

- Python 3.11+
- FastAPI + Uvicorn
- scikit-learn + XGBoost
- pandas + numpy
- Docker / Docker Compose
- AWS Lambda + ECR + S3

## Instalacion local

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Flujo ML (entrenamiento y evaluacion)

Ejecuta desde la raiz del proyecto:

```bash
python src/generate_data.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

Artefactos generados:
- `data/raw/server_metrics.csv`
- `data/processed/X_train.csv`, `data/processed/X_test.csv`
- `data/processed/y_train.csv`, `data/processed/y_test.csv`
- `models/model.pkl`, `models/scaler.pkl`
- `reports/metrics.json`

## Ejecutar API

### Opcion A: Uvicorn (local)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Opcion B: Docker Compose

```bash
docker compose up --build
```

Accesos:
- Swagger UI: `http://localhost:8000/docs`
- Healthcheck: `http://localhost:8000/health`
- Metricas: `http://localhost:8000/metrics`

## Endpoints

### `GET /health`

Valida estado del servicio y carga del modelo.

### `GET /metrics`

Retorna metricas almacenadas en `reports/metrics.json`.

### `POST /predict`

Request:

```json
{
  "cpu_usage": 87.5,
  "ram_usage": 91.2,
  "disk_io": 78.4,
  "network_traffic": 520.0,
  "temperature": 83.0,
  "cpu_spike_count": 14,
  "ram_spike_count": 11,
  "uptime_hours": 1500.0
}
```

Response ejemplo:

```json
{
  "will_fail": true,
  "probability": 0.8421,
  "risk_level": "CRITICAL",
  "time_to_failure": "< 1 hour",
  "top_causes": ["high_cpu", "high_ram", "high_temperature"],
  "model_version": "v1.0.0"
}
```

## Tests

El proyecto incluye pruebas en:
- `tests/test_preprocess.py`
- `tests/test_predict.py`
- `tests/test_lambda.py`

Ejecutar tests:

```bash
pytest tests/ -v
```

## CI/CD

- `ci.yml`: ejecuta tests en `push` y `pull_request` hacia `main`.
- `cd.yml`: construye imagen Docker de `lambda/`, publica en ECR y actualiza Lambda cuando hay cambios en `lambda/**`.

## Seguridad y publicacion responsable

Para evitar filtrar informacion sensible:
- No publiques secretos (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, tokens, API keys).
- No subas artefactos pesados o sensibles (`models/*.pkl`, `data/raw/*.csv`, `data/processed/*.csv`).
- Usa `GitHub Secrets` para credenciales de despliegue.
- Si publicas logs, revisa que no incluyan IDs internos, rutas privadas o datos operativos sensibles.
- Antes de abrir el repositorio, rota cualquier credencial usada en entornos de prueba.

## Licencia

Este proyecto se distribuye bajo la licencia incluida en `LICENSE`.
