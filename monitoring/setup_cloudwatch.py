"""
Script para configurar monitoreo de CloudWatch para el pipeline MLOps.
Crea alarmas y dashboards para Lambda y métricas del modelo.
"""

import boto3
import json
from typing import List, Dict

# Cliente de CloudWatch
cloudwatch = boto3.client('cloudwatch')
logs = boto3.client('logs')

# Configuración
LAMBDA_FUNCTION_NAME = "mlops-pipeline-function"
SNS_TOPIC_ARN = None  # Configurar con tu ARN de SNS si quieres notificaciones
NAMESPACE = "MLOps/ServerFailure"


def create_alarm(
    alarm_name: str,
    metric_name: str,
    threshold: float,
    comparison_operator: str,
    statistic: str = 'Average',
    period: int = 300,
    evaluation_periods: int = 2,
    namespace: str = NAMESPACE
):
    """Crea una alarma de CloudWatch"""
    try:
        alarm_config = {
            'AlarmName': alarm_name,
            'ComparisonOperator': comparison_operator,
            'EvaluationPeriods': evaluation_periods,
            'MetricName': metric_name,
            'Namespace': namespace,
            'Period': period,
            'Statistic': statistic,
            'Threshold': threshold,
            'ActionsEnabled': SNS_TOPIC_ARN is not None,
            'AlarmDescription': f'Alarma para {metric_name}',
        }
        
        if SNS_TOPIC_ARN:
            alarm_config['AlarmActions'] = [SNS_TOPIC_ARN]
        
        cloudwatch.put_metric_alarm(**alarm_config)
        print(f"✅ Alarma creada: {alarm_name}")
        
    except Exception as e:
        print(f"❌ Error creando alarma {alarm_name}: {e}")


def create_standard_alarms():
    """Crea alarmas estándar para el pipeline"""
    
    # Alarma: Alta latencia de predicción
    create_alarm(
        alarm_name='MLOps-HighPredictionLatency',
        metric_name='PredictionLatency',
        threshold=2000,  # 2 segundos
        comparison_operator='GreaterThanThreshold',
        statistic='Average',
        period=300
    )
    
    # Alarma: Alta tasa de errores
    create_alarm(
        alarm_name='MLOps-HighErrorRate',
        metric_name='PredictionErrors',
        threshold=5,
        comparison_operator='GreaterThanThreshold',
        statistic='Sum',
        period=300
    )
    
    # Alarma: Muchas predicciones de riesgo crítico
    create_alarm(
        alarm_name='MLOps-HighCriticalRiskRate',
        metric_name='CriticalRiskPredictions',
        threshold=10,
        comparison_operator='GreaterThanThreshold',
        statistic='Sum',
        period=300
    )
    
    # Alarma Lambda: Errores de función
    create_alarm(
        alarm_name='MLOps-LambdaErrors',
        metric_name='Errors',
        threshold=5,
        comparison_operator='GreaterThanThreshold',
        namespace='AWS/Lambda',
        period=300
    )
    
    # Alarma Lambda: Throttles
    create_alarm(
        alarm_name='MLOps-LambdaThrottles',
        metric_name='Throttles',
        threshold=1,
        comparison_operator='GreaterThanThreshold',
        namespace='AWS/Lambda',
        period=300
    )


def create_dashboard():
    """Crea un dashboard de CloudWatch para el pipeline MLOps"""
    
    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        [NAMESPACE, "PredictionLatency", {"stat": "Average"}],
                        ["...", {"stat": "p99"}]
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Latencia de Predicciones",
                    "yAxis": {"left": {"label": "ms"}}
                }
            },
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        [NAMESPACE, "PredictionsTotal", {"stat": "Sum"}],
                        [".", "FailurePredicted", {"stat": "Sum"}]
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "Predicciones Totales vs Fallos",
                    "yAxis": {"left": {"label": "Count"}}
                }
            },
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        [NAMESPACE, "PredictionErrors", {"stat": "Sum"}]
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "Errores de Predicción",
                    "yAxis": {"left": {"label": "Count"}}
                }
            },
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        [NAMESPACE, "CriticalRiskPredictions", {"stat": "Sum"}]
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "Predicciones de Riesgo Crítico",
                    "yAxis": {"left": {"label": "Count"}}
                }
            },
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["AWS/Lambda", "Duration", {"stat": "Average", "dimensions": {"FunctionName": LAMBDA_FUNCTION_NAME}}],
                        ["...", {"stat": "p99"}]
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Duración de Lambda",
                    "yAxis": {"left": {"label": "ms"}}
                }
            },
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["AWS/Lambda", "Invocations", {"stat": "Sum", "dimensions": {"FunctionName": LAMBDA_FUNCTION_NAME}}],
                        [".", "Errors", {"stat": "Sum", "...": "..."}],
                        [".", "Throttles", {"stat": "Sum", "...": "...""}]
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "Invocaciones Lambda",
                    "yAxis": {"left": {"label": "Count"}}
                }
            },
            {
                "type": "log",
                "properties": {
                    "query": f"SOURCE '/aws/lambda/{LAMBDA_FUNCTION_NAME}'\n| fields @timestamp, event, probability, risk_level, latency_seconds\n| filter event = 'prediction_success'\n| sort @timestamp desc\n| limit 20",
                    "region": "us-east-1",
                    "title": "Últimas Predicciones"
                }
            }
        ]
    }
    
    try:
        cloudwatch.put_dashboard(
            DashboardName='MLOps-ServerFailure-Pipeline',
            DashboardBody=json.dumps(dashboard_body)
        )
        print("✅ Dashboard creado: MLOps-ServerFailure-Pipeline")
    except Exception as e:
        print(f"❌ Error creando dashboard: {e}")


def create_log_metric_filters():
    """Crea filtros de métricas desde los logs de Lambda"""
    
    log_group_name = f'/aws/lambda/{LAMBDA_FUNCTION_NAME}'
    
    filters = [
        {
            'filterName': 'ModelLoadTime',
            'filterPattern': '{ $.event = "model_loaded" }',
            'metricName': 'ModelLoadTime',
            'metricValue': '$.load_time_seconds',
            'unit': 'Seconds'
        },
        {
            'filterName': 'PredictionErrors',
            'filterPattern': '{ $.event = "prediction_error" }',
            'metricName': 'PredictionErrorsFromLogs',
            'metricValue': '1',
            'unit': 'Count'
        }
    ]
    
    for filter_config in filters:
        try:
            logs.put_metric_filter(
                logGroupName=log_group_name,
                filterName=filter_config['filterName'],
                filterPattern=filter_config['filterPattern'],
                metricTransformations=[{
                    'metricName': filter_config['metricName'],
                    'metricNamespace': NAMESPACE,
                    'metricValue': filter_config['metricValue'],
                    'unit': filter_config.get('unit', 'None')
                }]
            )
            print(f"✅ Filtro de métrica creado: {filter_config['filterName']}")
        except Exception as e:
            print(f"❌ Error creando filtro {filter_config['filterName']}: {e}")


def setup_monitoring():
    """Configura todo el monitoreo de CloudWatch"""
    print("🚀 Configurando monitoreo de CloudWatch...\n")
    
    print("📊 Creando alarmas...")
    create_standard_alarms()
    
    print("\n📈 Creando dashboard...")
    create_dashboard()
    
    print("\n📝 Creando filtros de métricas en logs...")
    create_log_metric_filters()
    
    print("\n✅ Monitoreo configurado exitosamente")
    print(f"\n🔗 Dashboard: https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=MLOps-ServerFailure-Pipeline")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--function-name':
        LAMBDA_FUNCTION_NAME = sys.argv[2]
    
    if len(sys.argv) > 1 and sys.argv[1] == '--sns-topic':
        SNS_TOPIC_ARN = sys.argv[2]
    
    setup_monitoring()
