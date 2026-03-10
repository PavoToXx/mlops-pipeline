import json
import boto3
import joblib
import numpy as np
import pandas as pd
import os
import tempfile
import logging
import time
from datetime import datetime, timezone
import warnings

# -----------------------
# Config / Defaults (override with env vars)
# -----------------------
BUCKET        = os.getenv("MODELS_S3_BUCKET", "joseph-mlops-pipeline-models")
MODEL_KEY     = os.getenv("MODEL_S3_KEY", "models/model.pkl")
SCALER_KEY    = os.getenv("SCALER_S3_KEY", "models/scaler.pkl")

# Where prediction events will be stored (optional)
PREDICTIONS_S3_BUCKET = os.getenv("PREDICTIONS_S3_BUCKET", BUCKET)
PREDICTIONS_S3_PREFIX = os.getenv("PREDICTIONS_S3_PREFIX", "predictions")

# Default model version (override in env if you version your models)
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# Features con las que se entrena el modelo (deben coincidir con preprocess.py)
MODEL_FEATURE_COLS = [
    'cpu_usage', 'ram_usage', 'disk_io',
    'network_traffic', 'temperature',
    'cpu_spike_count', 'ram_spike_count',
    'uptime_hours',
    'cpu_ram_ratio', 'thermal_pressure',
    'spike_total', 'io_network_ratio',
    'risk_score', 'cpu_critical', 'ram_critical', 'temp_critical', 'multi_critical'
]

FEATURE_COLS = MODEL_FEATURE_COLS

# -----------------------
# Logging
# -----------------------
logger = logging.getLogger()
if not logger.handlers:
    # If running outside Lambda, basicConfig helps local debugging
    logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# -----------------------
# Clients (reusable)
# -----------------------
_s3 = boto3.client("s3")
_cloudwatch = boto3.client("cloudwatch")

# -----------------------
# Globals for cached models
# -----------------------
model = None
scaler = None

# -----------------------
# Utilities: CloudWatch metrics
# -----------------------
def log_metric(metric_name: str, value: float, unit: str = 'None', model_version: str = 'None'):
    """Send a custom metric datapoint to CloudWatch (non-blocking)."""
    try:
        dims = []
        if model_version:
            dims = [{"Name": "model_version", "Value": str(model_version)}]
        _cloudwatch.put_metric_data(
            Namespace='MLOps/ServerFailure',
            MetricData=[{
                'MetricName': metric_name,
                'Value': float(value),
                'Unit': unit,
                'Dimensions': dims
            }]
        )
    except Exception as e:
        logger.warning("Failed to log metric %s: %s", metric_name, e)


# -----------------------
# Utilities: structured prediction logging
# -----------------------
def _log_prediction(body_raw: dict, probability: float, will_fail: bool,
                    model_version: str = MODEL_VERSION,
                    s3_prefix: str = PREDICTIONS_S3_PREFIX,
                    s3_bucket: str = PREDICTIONS_S3_BUCKET):
    """
    - Print JSON (goes to CloudWatch Logs)
    - Put a datapoint to CloudWatch (namespace: mlops/predictions)
    - Optionally write a small JSONL object to S3 under the given prefix
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
        # 1) Structured log -> stdout -> CloudWatch Logs
        print(json.dumps(record, ensure_ascii=False))

        # 2) CloudWatch metric
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
            logger.warning("CloudWatch put_metric_data failed: %s", e)

        # 3) Optional: Upload single-line JSON to S3 for historical storage / monitoring pipeline
        if s3_bucket and s3_prefix:
            try:
                # Use timestamped key so writes are idempotent and cheap
                key = f"{s3_prefix.rstrip('/')}/{time.strftime('%Y/%m/%d')}/preds-{ts}.jsonl"
                _s3.put_object(
                    Bucket=s3_bucket,
                    Key=key,
                    Body=json.dumps(record, ensure_ascii=False) + "\n"
                )
            except Exception as e:
                logger.warning("S3 put_object failed: %s", e)

    except Exception as e:
        logger.error("Prediction logging error: %s", e)


# -----------------------
# Model loading
# -----------------------
def load_models():
    """Download model/scaler from S3 once and cache in module-level globals."""
    global model, scaler

    if model is not None and scaler is not None:
        logger.info(json.dumps({
            "event": "model_cache_hit",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))
        return

    start_time = time.time()
    tmp = tempfile.gettempdir()

    try:
        # Suppress sklearn InconsistentVersionWarning when unpickling models
        warnings.filterwarnings(
            "ignore",
            message=r"Trying to unpickle estimator StandardScaler from version .* when using version .*")

        logger.info(json.dumps({
            "event": "model_loading_start",
            "bucket": BUCKET,
            "model_key": MODEL_KEY,
            "scaler_key": SCALER_KEY,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))
        # Prefer local copies (helpful for local dev/tests) if present under repository `models/`
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        local_model_path = os.path.join(repo_root, 'models', 'model.pkl')
        local_scaler_path = os.path.join(repo_root, 'models', 'scaler.pkl')

        if os.path.exists(local_model_path) and os.path.exists(local_scaler_path):
            logger.info(json.dumps({
                "event": "model_loading_local",
                "model_path": local_model_path,
                "scaler_path": local_scaler_path,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }))
            loaded_model = joblib.load(local_model_path)
            loaded_scaler = joblib.load(local_scaler_path)
        else:
            # Download artifacts to tmp (overwrites if exists)
            _s3.download_file(BUCKET, MODEL_KEY, f"{tmp}/model.pkl")
            _s3.download_file(BUCKET, SCALER_KEY, f"{tmp}/scaler.pkl")

            model_local_path = f"{tmp}/model.pkl"
            scaler_local_path = f"{tmp}/scaler.pkl"

            # Load with joblib
            loaded_model = joblib.load(model_local_path)
            loaded_scaler = joblib.load(scaler_local_path)

        model = loaded_model
        scaler = loaded_scaler

        load_time = time.time() - start_time
        logger.info(json.dumps({
            "event": "model_loaded",
            "load_time_seconds": round(load_time, 3),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))

        log_metric('ModelLoadTime', load_time, 'Seconds', model_version=MODEL_VERSION)

    except Exception as e:
        logger.error(json.dumps({
            "event": "model_load_error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))
        raise


# -----------------------
# Feature engineering (same as training pipeline)
# -----------------------
def build_features(raw: dict) -> pd.DataFrame:
    cpu = raw['cpu_usage']
    ram = raw['ram_usage']
    temp = raw['temperature']
    net = raw['network_traffic']
    disk = raw['disk_io']
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

# -----------------------
# Risk level helper
# -----------------------
def get_risk_level(p: float) -> str:
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


# -----------------------
# Lambda handler
# -----------------------
def lambda_handler(event, context):
    """
    Lambda entry point. Expects event with body either JSON string or dict with required fields.
    Returns JSON response with will_fail, probability, risk_level.
    """
    start_time = time.time()
    request_id = getattr(context, "aws_request_id", "local")

    logger.info(json.dumps({
        "event": "prediction_request_start",
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }))

    try:
        # Ensure model/scaler loaded (cached across invocations while the container lives)
        load_models()

        # Parse input
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)

        logger.info(json.dumps({
            "event": "input_received",
            "request_id": request_id,
            "input_keys": list(body.keys()) if isinstance(body, dict) else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))

        # Build features and scale
        df = build_features(body)

        if scaler is None:
            raise RuntimeError("Scaler not loaded properly")

        if model is None:
            raise RuntimeError("Model not loaded properly")

        # Asegurarse de pasar solo las columnas usadas por el modelo al scaler
        df_model_input = df[MODEL_FEATURE_COLS]

        df_scaled = pd.DataFrame(
            scaler.transform(df_model_input),
            columns=MODEL_FEATURE_COLS
        )
        will_fail = bool(model.predict(df_scaled)[0])
        probability = round(float(model.predict_proba(df_scaled)[0][1]), 4)
        risk_level = get_risk_level(probability)

        latency = time.time() - start_time

        logger.info(json.dumps({
            "event": "prediction_success",
            "request_id": request_id,
            "will_fail": will_fail,
            "probability": probability,
            "risk_level": risk_level,
            "latency_seconds": round(latency, 3),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))

        # CloudWatch metrics (per-call)
        log_metric('PredictionLatencyMs', latency * 1000, 'Milliseconds', model_version=MODEL_VERSION)
        log_metric('PredictionProbability', probability, 'None', model_version=MODEL_VERSION)
        log_metric('PredictionsTotal', 1, 'Count', model_version=MODEL_VERSION)
        if will_fail:
            log_metric('FailurePredicted', 1, 'Count', model_version=MODEL_VERSION)
        if risk_level == 'CRITICAL':
            log_metric('CriticalRiskPredictions', 1, 'Count', model_version=MODEL_VERSION)

        # Structured log + CloudWatch metric + optional S3 upload
        try:
            _log_prediction(
                body_raw=body,
                probability=probability,
                will_fail=will_fail,
                model_version=MODEL_VERSION,
                s3_prefix=PREDICTIONS_S3_PREFIX,
                s3_bucket=PREDICTIONS_S3_BUCKET
            )
        except Exception as e:
            logger.warning("Prediction logging helper failed: %s", e)

        # Response
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))
        log_metric('PredictionErrors', 1, 'Count', model_version=MODEL_VERSION)
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))
        log_metric('PredictionErrors', 1, 'Count', model_version=MODEL_VERSION)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": str(e)
            })
        }