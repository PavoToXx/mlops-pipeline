# MLOps Pipeline - Server Failure Prediction

Proyecto de MLOps para entrenar, evaluar y servir un modelo de clasificacion que predice fallas de servidor a partir de metricas operativas.

Incluye:
- Pipeline de datos y entrenamiento (`src/`)
- API REST con FastAPI (`api/`)
- Empaquetado con Docker (`Dockerfile`, `docker-compose.yml`)
- Variante de inferencia en AWS Lambda (`lambda/`)
- Workflow de despliegue en GitHub Actions (`.github/workflows/ci.yml`)

## Objetivo

Predecir si un servidor fallara en las proximas horas, entregando:
- Probabilidad de falla
- Nivel de riesgo
- Tiempo estimado a falla
- Causas probables

## Estructura del proyecto

```text
api/                # FastAPI app y routers
src/                # Generacion de datos, preprocesamiento, entrenamiento y evaluacion
data/               # Datasets raw y procesados
models/             # Artefactos del modelo (no versionados)
reports/            # Metricas y reportes de evaluacion
lambda/             # Funcion y contenedor para AWS Lambda
monitoring/         # Espacio para monitoreo/observabilidad
tests/              # Pruebas (actualmente vacio)
```

## Stack tecnologico

- Python 3.11+
- FastAPI + Uvicorn
- scikit-learn + XGBoost
- pandas + numpy
- Docker / Docker Compose
- AWS Lambda + ECR + S3 (flujo cloud)

## Instalacion local

1. Clonar repositorio.
2. Crear y activar entorno virtual.
3. Instalar dependencias.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Ejecucion del pipeline ML

Orden recomendado desde la raiz del proyecto:

```bash
python src/generate_data.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

Resultados esperados:
- `data/raw/server_metrics.csv`
- `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- `models/model.pkl`, `models/scaler.pkl`
- `reports/metrics.json`

## Levantar la API

### Opcion A: local con Uvicorn

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Opcion B: Docker Compose

```bash
docker compose up --build
```

API disponible en:
- Swagger UI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Endpoints principales

### `GET /health`

Valida estado del servicio y carga de modelo.

### `GET /metrics`

Retorna metricas desde `reports/metrics.json`.

### `POST /predict`

Entrada esperada:

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

Respuesta ejemplo:

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

## Despliegue cloud (resumen)

El repositorio contiene un workflow de GitHub Actions en `.github/workflows/ci.yml` que:
- Construye imagen Docker de `lambda/`
- Publica en Amazon ECR
- Actualiza una funcion AWS Lambda basada en imagen

Importante:
- Las credenciales AWS se consumen desde `GitHub Secrets`.
- No incluir claves, tokens ni archivos `.env` en el repositorio.

## Seguridad y publicacion responsable

Para evitar filtrar informacion sensible:
- No publiques secretos (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, tokens, API keys).
- No subas artefactos pesados o sensibles (`models/*.pkl`, `data/raw/*.csv`, `data/processed/*.csv`).
- Usa variables de entorno para configuracion de infraestructura (bucket, regiones, nombres de funciones).
- Revisa logs antes de compartirlos: podrian contener rutas internas o identificadores.
- Si vas a abrir el repositorio, rota credenciales previamente usadas en pruebas.

## Pruebas

`tests/` existe pero actualmente no contiene pruebas automatizadas.

## Licencia

Este proyecto se distribuye bajo la licencia incluida en `LICENSE`.
