import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from mlflow.models.signature import infer_signature

# Opcional: skops para persistencia más segura
USE_SKOPS = os.getenv("USE_SKOPS", "0") == "1"
if USE_SKOPS:
    try:
        import skops.io as skops_io
    except Exception:
        USE_SKOPS = False
        print("⚠️ skops no disponible, USE_SKOPS desactivado. Instala 'skops' si quieres usarlo.")

def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()
    print(f"✅ Datos cargados:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test, params):
    # REMOVI: use_label_encoder (deprecated/unused en versiones recientes)
    model = XGBClassifier(
        n_estimators      = params.get("n_estimators", 300),
        max_depth         = params.get("max_depth", 6),
        learning_rate     = params.get("learning_rate", 0.05),
        subsample         = params.get("subsample", 0.8),
        colsample_bytree  = params.get("colsample_bytree", 0.8),
        eval_metric       = "logloss",
        random_state      = params.get("random_state", 42),
        early_stopping_rounds = params.get("early_stopping_rounds", 20),
        # puedes controlar threads si quieres limitar CPU:
        # n_jobs = params.get("n_jobs", 1)
    )
    model.fit(
        X_train, y_train,
        eval_set        = [(X_test, y_test)],
        verbose         = 50,
    )
    return model

def evaluate_model(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)),  4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall":    round(float(recall_score(y_test, y_pred)),    4),
        "f1":        round(float(f1_score(y_test, y_pred)),        4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_proba)),  4),
    }

    print("\n📊 Métricas del modelo:")
    for k, v in metrics.items():
        print(f"   {k:12}: {v}")
    return metrics

def save_model_local(model, out_path="models/model.pkl"):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, out_path)
    print(f"\n✅ Modelo guardado en {out_path}")

def print_feature_importance(model, feature_names):
    importance = pd.DataFrame({
        'feature':    feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🔍 Top 5 features más importantes:")
    print(importance.head(5).to_string(index=False))

if __name__ == "__main__":
    print("🚀 Iniciando entrenamiento...\n")

    # cargar datos
    X_train, X_test, y_train, y_test = load_data()

    # Hiperparámetros (puedes ajustar via env vars o pasar por CLI más adelante)
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "early_stopping_rounds": 20,
        # "n_jobs": 1,  # descomenta para limitar uso CPU
    }

    # Configurar MLflow tracking URI si está en env
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
    if MLFLOW_URI:
        mlflow.set_tracking_uri(MLFLOW_URI)
        print(f"MLflow tracking URI: {MLFLOW_URI}")
    else:
        print("MLflow tracking URI not set — using default local (environment)")

    # Nombre del experimento (puedes parametrizar)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "mlops-pipeline-experiment")
    mlflow.set_experiment(experiment_name)

    # Run MLflow
    with mlflow.start_run(run_name=f"train-{os.getenv('GITHUB_SHA','local')}") as run:
        # log params
        mlflow.log_params(params)

        # Entrenar
        model = train_model(X_train, X_test, y_train, y_test, params)

        # Evaluar
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log feature names as tag/artifact
        mlflow.set_tag("model.flavor", "xgboost")
        mlflow.log_text("\n".join(X_train.columns.tolist()), "feature_names.txt")

        # Guardar modelo local también (pickle)
        save_model_local(model, out_path="models/model.pkl")

        # Intentar inferir signature y input_example para mejor trazabilidad
        try:
            signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])
            input_example = X_train.head(1)
        except Exception:
            signature = None
            input_example = None

        # Logueo del modelo en MLflow: intentar usar 'name' si está disponible (nueva API),
        # si no, caer en artifact_path para compatibilidad.
        try:
            # Preferimos 'name' en MLflow >= 2.x (si existe)
            mlflow.sklearn.log_model(sk_model=model, name="model", signature=signature, input_example=input_example)
        except TypeError:
            # Fallback para versiones antiguas que usan artifact_path
            mlflow.sklearn.log_model(sk_model=model, artifact_path="model", signature=signature, input_example=input_example)

        # Opcional: guardar con skops (serialización más segura) y subir como artifact
        if USE_SKOPS:
            skops_path = "models/model.skops"
            try:
                skops_io.dump(model, skops_path)
                mlflow.log_artifact(skops_path, artifact_path="model_skops")
                print(f"✅ Modelo guardado con skops en {skops_path} y subido a MLflow artifacts/model_skops")
            except Exception as e:
                print("⚠️ Error guardando con skops:", e)

        run_id = run.info.run_id
        print(f"\n✅ MLflow run finished: {run_id}")

        # Optional: register model if registry available and env var set
        registry_name = os.getenv("MLFLOW_REGISTER_NAME")
        if registry_name:
            model_uri = f"runs:/{run_id}/model"
            try:
                mv = mlflow.register_model(model_uri, registry_name)
                print(f"✅ Registered model {registry_name} version: {mv.version}")
            except Exception as e:
                print("⚠️ Model registry error:", e)

    print_feature_importance(model, X_train.columns.tolist())

    print(f"\n{'='*45}")
    if metrics['f1'] >= 0.85:
        print(f"✅ MODELO APROBADO  — F1: {metrics['f1']}")