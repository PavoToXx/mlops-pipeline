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
cloudwatch = boto3.client('cloudwatch')

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
        'cpu_usage':        cpu,
        'ram_usage':        ram,
        'disk_io':          disk,
        'network_traffic':  net,
        'temperature':      temp,
        'cpu_spike_count':  cpu_s,
        'ram_spike_count':  ram_s,
        'uptime_hours':     raw['uptime_hours'],
        'cpu_ram_ratio':    cpu / (ram + 1),
        'thermal_pressure': temp * cpu / 100,
        'spike_total':      spike_total,
        'io_network_ratio': disk / (net + 1),
        # Features adicionales que el modelo necesita
        'risk_score':    cpu * 0.30 + ram * 0.25 + temp * 0.20 + disk * 0.15 + spike_total * 0.10,
        'cpu_critical':  int(cpu  > 90),
        'ram_critical':  int(ram  > 90),
        'temp_critical': int(temp > 80),
        'multi_critical': int(cpu > 90) + int(ram > 90) + int(temp > 80),
    }
    return pd.DataFrame([features])[FEATURE_COLS]

def get_risk_level(p: float) -> str:
    if p >= 0.80:   return "CRITICAL"
    elif p >= 0.60: return "HIGH"
    elif p >= 0.40: return "MEDIUM"
    elif p >= 0.20: return "LOW"
    else:           return "NORMAL"

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

        # Feature engineering y predicción
        df        = build_features(body)
        df_scaled = pd.DataFrame(
            scaler.transform(df), columns=FEATURE_COLS
        )

        will_fail   = bool(model.predict(df_scaled)[0])
        probability = round(float(
            model.predict_proba(df_scaled)[0][1]
        ), 4)
        risk_level = get_risk_level(probability)
        
        # Calcular latencia
        latency = time.time() - start_time
        
        # Log de resultado
        logger.info(json.dumps({
            "event": "prediction_success",
            "request_id": request_id,
            "will_fail": will_fail,
            "probability": probability,
            "risk_level": risk_level,
            "latency_seconds": round(latency, 3),
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        # Métricas custom a CloudWatch
        log_metric('PredictionLatency', latency * 1000, 'Milliseconds')
        log_metric('PredictionProbability', probability, 'None')
        log_metric('PredictionsTotal', 1, 'Count')
        
        if will_fail:
            log_metric('FailurePredicted', 1, 'Count')
            
        if risk_level == 'CRITICAL':
            log_metric('CriticalRiskPredictions', 1, 'Count')

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "will_fail":   will_fail,
                "probability": probability,
                "risk_level":  risk_level,
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
            "body": json.dumps({"error": f"Missing required field: {str(e)}"})
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
            "body": json.dumps({"error": str(e)})
        }