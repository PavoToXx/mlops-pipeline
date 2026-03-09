# MLOps Pipeline - Server Failure Prediction

Proyecto de MLOps para entrenar, evaluar y servir un modelo de clasificacion que predice fallas de servidor a partir de metricas operativas.

## Que incluye

- Pipeline de datos y entrenamiento (`src/`)
- **Tracking de experimentos con MLflow** (`mlruns/`)
- API REST con FastAPI (`api/`)
- Empaquetado con Docker (`Dockerfile`, `docker-compose.yml`)
- Variante de inferencia en AWS Lambda (`lambda/`)
- CI con tests automatizados (`.github/workflows/ci.yml`)
- CI de entrenamiento automático (`.github/workflows/train.yml`)
- CD con despliegue a Lambda (`.github/workflows/cd.yml`)
- **Monitoreo con CloudWatch** (`monitoring/`)

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
mlruns/             # Experimentos MLflow (no versionado)
monitoring/         # Scripts de configuracion de CloudWatch
tests/              # Tests unitarios (pipeline, API, Lambda)
.github/workflows/  # CI/CD con GitHub Actions
```

## Stack tecnologico

- Python 3.11+
- FastAPI + Uvicorn
- scikit-learn + XGBoost
- **MLflow** (experiment tracking)
- pandas + numpy
- Docker / Docker Compose
- AWS Lambda + ECR + S3
- **AWS CloudWatch** (logs y metricas)
- **GitHub Actions** (CI/CD)

## Instalacion local

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt

Note: This project has been tested with Python 3.14 for the Lambda container. If you run the Lambda container locally, the `lambda_folder/Dockerfile` uses Python 3.14. Local development and tests run under your venv Python (3.11+ is supported), but ensure `scikit-learn` and `cloudpickle` versions match the model serialization to avoid deserialization warnings.
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
- `mlruns/` (experimentos MLflow)

## MLflow Tracking

El entrenamiento registra automaticamente en MLflow:
- Hiperparametros del modelo
- Metricas de evaluacion
- Modelo entrenado
- Tags (tipo de modelo, aprobacion)

Ver experimentos:

```bash
mlflow ui
```

Accede a `http://localhost:5000` para explorar runs, comparar metricas y descargar modelos.

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
- `tests/test_preprocess.py` - Tests de feature engineering
- `tests/test_predict.py` - Tests del predictor
- `tests/test_lambda.py` - Tests de la funcion Lambda
- `tests/test_api.py` - Tests de endpoints FastAPI

Ejecutar tests:

```bash
pytest tests/ -v

# Con coverage
pytest tests/ --cov=api --cov=src --cov=lambda -v

Tips for running tests and Lambda locally:

- The tests mock AWS where needed; to run Lambda handler manually use `invoke_lambda_local.py` at project root which calls `lambda_folder.lambda_function.lambda_handler` and will load local `models/model.pkl` and `models/scaler.pkl` if present.
- To build the Lambda container (uses Python 3.14):

```bash
docker build -t ml-lambda:local lambda_folder
docker run --rm -v $(pwd):/var/task -w /var/task ml-lambda:local \
  python -c "from lambda_folder import lambda_function as lf; print(lf.lambda_handler({'body': {'cpu_usage':50,'ram_usage':60,'temperature':70,'disk_io':30,'network_traffic':100,'cpu_spike_count':1,'ram_spike_count':0,'uptime_hours':10}}, None))"
```

- If you see `InconsistentVersionWarning` when loading models, either align `scikit-learn` versions between training and serving (recommended), or reserialize the model with the target sklearn version. Tests suppress this warning during loading.
```

## CI/CD

### Workflows de GitHub Actions

- **`ci.yml`**: Ejecuta tests en `push` y `pull_request` hacia `main`
- **`train.yml`**: Pipeline completo de entrenamiento
  - Genera datos, preprocesa, entrena y evalua
  - Registra experimentos en MLflow
  - Valida metricas contra threshold (F1 >= 0.85)
  - Sube artefactos (modelo, metricas, MLflow runs)
  - Comenta metricas en PRs
- **`cd.yml`**: Despliegue continuo a Lambda
  - Construye imagen Docker de `lambda/`
  - Publica en Amazon ECR
  - Actualiza funcion Lambda cuando hay cambios en `lambda/**`

## Monitoreo

Scripts en `monitoring/` para configurar CloudWatch:

```bash
# Configurar alarmas y dashboard
python monitoring/setup_cloudwatch.py

# Con parametros personalizados
python monitoring/setup_cloudwatch.py --function-name tu-lambda --sns-topic arn:aws:sns:...
```

Componentes de monitoreo:
- **Alarmas**: Latencia alta, errores, predicciones criticas, throttles
- **Dashboard**: Metricas visuales de predicciones, latencia, errores
- **Logs estructurados**: JSON con eventos, metricas y traces
- **Metricas custom**: Namespace `MLOps/ServerFailure`

Ver documentacion completa en [`monitoring/README.md`](monitoring/README.md).

## Seguridad y publicacion responsable

Para evitar filtrar informacion sensible:
- No publiques secretos (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, tokens, API keys).
- No subas artefactos pesados o sensibles (`models/*.pkl`, `data/raw/*.csv`, `data/processed/*.csv`, `mlruns/`).
- Usa `GitHub Secrets` para credenciales de despliegue.
- Si publicas logs, revisa que no incluyan IDs internos, rutas privadas o datos operativos sensibles.
- Antes de abrir el repositorio, rota cualquier credencial usada en entornos de prueba.
- Los experimentos de MLflow pueden contener datos sensibles; no los versiones en Git.

## Licencia

Este proyecto se distribuye bajo la licencia incluida en `LICENSE`.
