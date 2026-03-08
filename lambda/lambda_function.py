import json
import boto3
import joblib
import numpy as np
import pandas as pd
import os
import tempfile
import logging
import time
from datetime import datetime

# Configurar logging estructurado para CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

BUCKET     = "joseph-mlops-pipeline-models"
MODEL_KEY  = "models/model.pkl"
SCALER_KEY = "models/scaler.pkl"
S3_BUCKET  = "joseph-mlops-pipeline-models"
S3_PREFIX  = "joseph-mlops-pipeline-models/predictions"

FEATURE_COLS = [
    'cpu_usage', 'ram_usage', 'disk_io',
    'network_traffic', 'temperature',
    'cpu_spike_count', 'ram_spike_count',
    'uptime_hours',
    'cpu_ram_ratio', 'thermal_pressure',
    'spike_total', 'io_network_ratio',
    'risk_score', 'cpu_critical',
    'ram_critical', 'temp_critical',
    'multi_critical',
]

model  = None
scaler = None

# AWS Clients reutilizables
cloudwatch = boto3.client('cloudwatch')
_s3 = boto3.client("s3")
_cloudwatch = cloudwatch


# -----------------------------------------
# MÉTRICAS CLOUDWATCH
# -----------------------------------------
def log_metric(metric_name, value, unit='None'):
    """Envia métricas custom a CloudWatch"""
    try:
        cloudwatch.put_metric_data(
            Namespace='MLOps/ServerFailure',
            MetricData=[{
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.utcnow()
            }]
        )
    except Exception as e:
        logger.warning(f"Failed to log metric {metric_name}: {e}")


# -----------------------------------------
# LOG AVANZADO DE PREDICCIONES
# -----------------------------------------
def _log_prediction(body_raw: dict, probability: float, will_fail: bool,
                    model_version: str = "v1",
                    s3_prefix: str = S3_PREFIX,
                    s3_bucket: str = S3_BUCKET):
    """
    Log estructurado a CloudWatch Logs,
    enviar métrica a CloudWatch,
    opcionalmente guardar en S3 como jsonl
    """

    ts = int(time.time())

    record = {
        "ts": ts,
        "model_version": model_version,
        "probability": float(probability),
        "will_fail": bool(will_fail),
        "input": body_raw
    }

    try:

        # CloudWatch Logs
        print(json.dumps(record, ensure_ascii=False))

        # CloudWatch Metrics
        try:
            _cloudwatch.put_metric_data(
                Namespace="mlops/predictions",
                MetricData=[{
                    "MetricName": "prediction_probability",
                    "Dimensions": [
                        {"Name": "model_version", "Value": str(model_version)}
                    ],
                    "Value": float(probability),
                    "Unit": "None"
                }]
            )
        except Exception as e:
            logger.warning("CloudWatch metric failed: %s", e)

        # Guardar predicción en S3 (opcional)
        if s3_bucket and s3_prefix:
            try:
                key = f"{s3_prefix.rstrip('/')}/{time.strftime('%Y/%m/%d')}/preds-{ts}.jsonl"

                _s3.put_object(
                    Bucket=s3_bucket,
                    Key=key,
                    Body=json.dumps(record, ensure_ascii=False) + "\n"
                )

            except Exception as e:
                logger.warning("S3 logging failed: %s", e)

    except Exception as e:
        logger.error("Prediction logging error: %s", e)


# -----------------------------------------
# CARGA MODELOS
# -----------------------------------------
def load_models():

    global model, scaler

    if model is not None:
        logger.info(json.dumps({
            "event": "model_cache_hit",
            "timestamp": datetime.utcnow().isoformat()
        }))
        return

    start_time = time.time()

    s3  = boto3.client('s3')
    tmp = tempfile.gettempdir()

    try:

        logger.info(json.dumps({
            "event": "model_loading_start",
            "bucket": BUCKET,
            "model_key": MODEL_KEY,
            "scaler_key": SCALER_KEY
        }))

        s3.download_file(BUCKET, MODEL_KEY,  f"{tmp}/model.pkl")
        s3.download_file(BUCKET, SCALER_KEY, f"{tmp}/scaler.pkl")

        model  = joblib.load(f"{tmp}/model.pkl")
        scaler = joblib.load(f"{tmp}/scaler.pkl")

        load_time = time.time() - start_time

        logger.info(json.dumps({
            "event": "model_loaded",
            "load_time_seconds": round(load_time, 3),
            "timestamp": datetime.utcnow().isoformat()
        }))

        log_metric('ModelLoadTime', load_time, 'Seconds')

    except Exception as e:

        logger.error(json.dumps({
            "event": "model_load_error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }))

        raise


# -----------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------
def build_features(raw: dict) -> pd.DataFrame:

    cpu   = raw['cpu_usage']
    ram   = raw['ram_usage']
    temp  = raw['temperature']
    net   = raw['network_traffic']
    disk  = raw['disk_io']
    cpu_s = raw['cpu_spike_count']
    ram_s = raw['ram_spike_count']

    spike_total = cpu_s + ram_s

    features = {

        'cpu_usage': cpu,
        'ram_usage': ram,
        'disk_io': disk,
        'network_traffic': net,
        'temperature': temp,
        'cpu_spike_count': cpu_s,
        'ram_spike_count': ram_s,
        'uptime_hours': raw['uptime_hours'],

        'cpu_ram_ratio': cpu / (ram + 1),

        'thermal_pressure': temp * cpu / 100,

        'spike_total': spike_total,

        'io_network_ratio': disk / (net + 1),

        'risk_score': cpu * 0.30 + ram * 0.25 + temp * 0.20 + disk * 0.15 + spike_total * 0.10,

        'cpu_critical': int(cpu > 90),

        'ram_critical': int(ram > 90),

        'temp_critical': int(temp > 80),

        'multi_critical': int(cpu > 90) + int(ram > 90) + int(temp > 80),

    }

    return pd.DataFrame([features])[FEATURE_COLS]


# -----------------------------------------
# RISK LEVEL
# -----------------------------------------
def get_risk_level(p: float):

    if p >= 0.80:
        return "CRITICAL"

    elif p >= 0.60:
        return "HIGH"

    elif p >= 0.40:
        return "MEDIUM"

    elif p >= 0.20:
        return "LOW"

    else:
        return "NORMAL"


# -----------------------------------------
# LAMBDA HANDLER
# -----------------------------------------
def lambda_handler(event, context):

    start_time = time.time()

    request_id = context.request_id if context else "local"

    logger.info(json.dumps({
        "event": "prediction_request_start",
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat()
    }))

    try:

        load_models()

        if isinstance(event.get('body'), str):

            body = json.loads(event['body'])

        else:

            body = event.get('body', event)

        logger.info(json.dumps({
            "event": "input_received",
            "request_id": request_id,
            "input_keys": list(body.keys())
        }))

        df = build_features(body)

        if scaler is None:
            raise RuntimeError("Scaler not loaded properly")

        df_scaled = pd.DataFrame(
            scaler.transform(df),
            columns=FEATURE_COLS
        )

        will_fail = bool(model.predict(df_scaled)[0])

        probability = round(float(
            model.predict_proba(df_scaled)[0][1]
        ), 4)

        risk_level = get_risk_level(probability)

        latency = time.time() - start_time

        logger.info(json.dumps({
            "event": "prediction_success",
            "request_id": request_id,
            "will_fail": will_fail,
            "probability": probability,
            "risk_level": risk_level,
            "latency_seconds": round(latency, 3),
            "timestamp": datetime.utcnow().isoformat()
        }))

        # Métricas
        log_metric('PredictionLatency', latency * 1000, 'Milliseconds')
        log_metric('PredictionProbability', probability, 'None')
        log_metric('PredictionsTotal', 1, 'Count')

        if will_fail:
            log_metric('FailurePredicted', 1, 'Count')

        if risk_level == 'CRITICAL':
            log_metric('CriticalRiskPredictions', 1, 'Count')

        # NUEVO: LOG DE PREDICCIÓN PARA MLOPS
        _log_prediction(
            body_raw=body,
            probability=probability,
            will_fail=will_fail,
            model_version="v1",
            s3_prefix= S3_PREFIX,
            s3_bucket= S3_BUCKET
        )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "will_fail": will_fail,
                "probability": probability,
                "risk_level": risk_level
            })
        }

    except KeyError as e:

        logger.error(json.dumps({
            "event": "missing_field_error",
            "request_id": request_id,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }))

        log_metric('PredictionErrors', 1, 'Count')

        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": f"Missing required field: {str(e)}"
            })
        }

    except Exception as e:

        logger.error(json.dumps({
            "event": "prediction_error",
            "request_id": request_id,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }))

        log_metric('PredictionErrors', 1, 'Count')

        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": str(e)
            })
        }