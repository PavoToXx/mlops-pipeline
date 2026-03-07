# Monitoreo CloudWatch

Scripts para configurar y gestionar el monitoreo del pipeline MLOps en AWS CloudWatch.

## `setup_cloudwatch.py`

Configura alarmas, dashboards y filtros de métricas para monitorear la función Lambda y el rendimiento del modelo.

### Requisitos

```bash
pip install boto3
```

### Configuración de credenciales AWS

Asegúrate de tener configuradas tus credenciales AWS:

```bash
aws configure
```

O configura las variables de entorno:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### Uso básico

```bash
python monitoring/setup_cloudwatch.py
```

### Uso con parámetros

```bash
# Especificar nombre de función Lambda
python monitoring/setup_cloudwatch.py --function-name tu-funcion-lambda

# Agregar notificaciones SNS
python monitoring/setup_cloudwatch.py --sns-topic arn:aws:sns:us-east-1:123456789012:your-topic
```

## Componentes creados

### 1. Alarmas

- **MLOps-HighPredictionLatency**: Alerta si la latencia promedio supera 2 segundos
- **MLOps-HighErrorRate**: Alerta si hay más de 5 errores en 5 minutos
- **MLOps-HighCriticalRiskRate**: Alerta si hay muchas predicciones de riesgo crítico
- **MLOps-LambdaErrors**: Alerta por errores de Lambda
- **MLOps-LambdaThrottles**: Alerta por throttling de Lambda

### 2. Dashboard

Dashboard visual con:
- Latencia de predicciones (promedio y p99)
- Total de predicciones vs fallos predichos
- Errores de predicción
- Predicciones de riesgo crítico
- Duración de Lambda
- Invocaciones y throttles de Lambda
- Últimas predicciones (query de logs)

Para acceder:
```
https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=MLOps-ServerFailure-Pipeline
```

### 3. Métricas custom

Namespace: `MLOps/ServerFailure`

Métricas enviadas desde Lambda:
- `PredictionLatency`: Tiempo de inferencia en ms
- `PredictionsTotal`: Total de predicciones
- `FailurePredicted`: Predicciones de fallo
- `CriticalRiskPredictions`: Predicciones de riesgo crítico
- `PredictionErrors`: Errores durante predicción
- `ModelLoadTime`: Tiempo de carga del modelo

### 4. Filtros de métricas desde logs

Extrae métricas automáticamente de los logs estructurados JSON:
- `ModelLoadTime`: Tiempo de carga desde S3
- `PredictionErrorsFromLogs`: Conteo de errores

## Estructura de logs

Los logs de Lambda están en formato JSON estructurado:

```json
{
  "event": "prediction_success",
  "request_id": "abc-123",
  "will_fail": true,
  "probability": 0.8421,
  "risk_level": "CRITICAL",
  "latency_seconds": 0.123,
  "timestamp": "2026-03-07T10:30:00Z"
}
```

## Queries útiles de CloudWatch Logs Insights

### Predicciones exitosas en la última hora

```
fields @timestamp, probability, risk_level, latency_seconds
| filter event = "prediction_success"
| sort @timestamp desc
| limit 100
```

### Errores con detalles

```
fields @timestamp, error, error_type
| filter event = "prediction_error"
| sort @timestamp desc
```

### Latencia promedio por hora

```
fields @timestamp, event, latency_seconds
| filter event = "prediction_success"
| stats avg(latency_seconds) as avg_latency by bin(5m)
```

### Distribución de niveles de riesgo

```
fields risk_level
| filter event = "prediction_success"
| stats count() by risk_level
```

## Eliminación de recursos

Para eliminar las alarmas y dashboard creados:

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Eliminar alarmas
alarms = [
    'MLOps-HighPredictionLatency',
    'MLOps-HighErrorRate',
    'MLOps-HighCriticalRiskRate',
    'MLOps-LambdaErrors',
    'MLOps-LambdaThrottles'
]
cloudwatch.delete_alarms(AlarmNames=alarms)

# Eliminar dashboard
cloudwatch.delete_dashboards(DashboardNames=['MLOps-ServerFailure-Pipeline'])
```

## Costos

- CloudWatch Logs: ~$0.50/GB ingested
- CloudWatch Metrics: Primeras 10 métricas custom gratis, luego $0.30 por métrica
- CloudWatch Alarms: $0.10 por alarma/mes
- CloudWatch Dashboards: Primeros 3 gratis, luego $3/mes cada uno

Para este pipeline básico, el costo mensual estimado es < $5 USD.
