#!/usr/bin/env python3
"""
Monitor básico de calidad de predicciones.
- Lee archivos JSONL desde s3://{bucket}/{prefix}/YYYY/MM/DD/*.jsonl
- Calcula estadísticas: count, mean(probability), pct_prob_above_0.8
- Compara contra baseline (archivo local reports/metrics.json o valores en código)
Exit codes:
  0 -> OK
  2 -> Alert (degradación detectada)
"""

import os
import json
import boto3
from datetime import datetime, timedelta
from statistics import mean

s3 = boto3.client("s3")
sns = boto3.client("sns")
secrets = boto3.client("secretsmanager")

# ENV
BUCKET = os.getenv("PREDICTIONS_S3_BUCKET")
PREFIX = os.getenv("PREDICTIONS_S3_PREFIX", "predictions")
SNS_ARN = os.getenv("MONITOR_SNS_ARN")
LOOKBACK_DAYS = int(os.getenv("MONITOR_LOOKBACK_DAYS", "1"))
THRESHOLD_DROP = float(os.getenv("MONITOR_PCT_DROP_THRESHOLD", "0.10"))
BASELINE_SECRET_ARN = os.getenv("MONITOR_BASELINE_SECRET_ARN")  # opcional para baseline dinámico

def list_keys_for_day(bucket, prefix, day_str):
    paginator = s3.get_paginator("list_objects_v2")
    day_prefix = f"{prefix.rstrip('/')}/{day_str}/"
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=day_prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    return keys

def read_jsonl_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    for line in obj["Body"].read().decode("utf-8").splitlines():
        if line.strip():
            yield json.loads(line)

def collect_recent_records(bucket, prefix, days):
    records = []
    for d in range(days):
        day = (datetime.utcnow() - timedelta(days=d)).strftime("%Y/%m/%d")
        keys = list_keys_for_day(bucket, prefix, day)
        for k in keys:
            try:
                for rec in read_jsonl_from_s3(bucket, k):
                    records.append(rec)
            except Exception:
                pass
    return records

def load_baseline():
    # try secret first, then fallback to S3 file or hard-coded
    if BASELINE_SECRET_ARN:
        try:
            sec = secrets.get_secret_value(SecretId=BASELINE_SECRET_ARN)
            return json.loads(sec["SecretString"])
        except Exception:
            pass
    # fallback simple baseline (tune o guarda en S3)
    return {"mean_prob": 0.60}

def publish_alert(subject, body):
    if SNS_ARN:
        sns.publish(TopicArn=SNS_ARN, Subject=subject, Message=body)
    else:
        print("ALERT:", subject, body)

def lambda_handler(event, context):
    if not BUCKET:
        return {"statusCode": 500, "body": "PREDICTIONS_S3_BUCKET not set"}

    recs = collect_recent_records(BUCKET, PREFIX, LOOKBACK_DAYS)
    if not recs:
        return {"statusCode": 200, "body": "No predictions found"}

    probs = [float(r.get("probability", 0.0)) for r in recs]
    stats = {"count": len(probs), "mean_prob": mean(probs), "pct_ge_0_8": sum(1 for p in probs if p>=0.8) / len(probs)}

    baseline = load_baseline()
    baseline_mean = float(baseline.get("mean_prob", 0.6))
    drop = (baseline_mean - stats["mean_prob"]) / baseline_mean if baseline_mean>0 else 0.0

    body = {"stats": stats, "baseline_mean": baseline_mean, "drop": drop}
    print(json.dumps(body))

    if drop >= THRESHOLD_DROP:
        publish_alert("ML model degradation detected", json.dumps(body))
        # opcional: crear issue en GitHub si tienes token (usar requests o GitHub API)
        return {"statusCode": 200, "body": "ALERT_SENT"}
    else:
        return {"statusCode": 200, "body": "OK"}