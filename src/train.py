import os
import joblib
import pandas as pd

# ── MLflow: importar submódulos explícitamente para que Pylance los reconozca
import mlflow
import mlflow.xgboost as mlflow_xgboost
import mlflow.sklearn as mlflow_sklearn
import mlflow.pyfunc as mlflow_pyfunc

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from mlflow.models.signature import infer_signature

# ── skops: import a nivel de módulo con fallback, evita "possibly unbound" ───
try:
    import skops.io as skops_io
    _SKOPS_AVAILABLE = True
except ImportError:
    skops_io = None          # type: ignore[assignment]
    _SKOPS_AVAILABLE = False

# Leer env var DESPUÉS de intentar el import, para poder cruzar ambos estados
USE_SKOPS = str(os.getenv("USE_SKOPS", "0")).strip().lower() in ("1", "true", "yes")
if USE_SKOPS and not _SKOPS_AVAILABLE:
    USE_SKOPS = False
    print("⚠️  skops no disponible — USE_SKOPS desactivado. Instala 'skops' para usarlo.")
print("USE_SKOPS:", USE_SKOPS)

# ─── Funciones ────────────────────────────────────────────────────────────────

def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()
    print(f"✅ Datos cargados — X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, params):
    model = XGBClassifier(
        n_estimators          = params["n_estimators"],
        max_depth             = params["max_depth"],
        learning_rate         = params["learning_rate"],
        subsample             = params["subsample"],
        colsample_bytree      = params["colsample_bytree"],
        eval_metric           = "logloss",
        random_state          = params["random_state"],
        early_stopping_rounds = params["early_stopping_rounds"],
    )
    model.fit(
        X_train, y_train,
        eval_set = [(X_test, y_test)],
        verbose  = 50,
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
    """Guarda el modelo en disco con joblib (una sola vez)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)
    print(f"✅ Modelo guardado localmente en {out_path}")
    return out_path


def build_signature(X_train, model):
    """Intenta inferir la firma del modelo; retorna None si falla."""
    try:
        return infer_signature(X_train, model.predict_proba(X_train)[:, 1])
    except Exception as e:
        print(f"⚠️  No se pudo inferir la firma del modelo: {e}")
        return None


def log_model_to_mlflow(model, signature, input_example):
    """
    Loguea el modelo en MLflow usando la mejor estrategia disponible:
      1. skops  (si USE_SKOPS=true y skops está instalado)
      2. mlflow.xgboost  (formato nativo XGBoost, sin pickle) ← recomendado
      3. mlflow.sklearn  (fallback con pickle)
    """
    if USE_SKOPS and skops_io is not None:
        skops_path = "models/model.skops"
        python_model_path = "src/mlflow_models/skops_model_from_code.py"
        try:
            skops_io.dump(model, skops_path)
            print(f"✅ Modelo guardado con skops en {skops_path}")

            log_kwargs = {
                "name": "model",
                "python_model": python_model_path,
                "artifacts": {"skops_model": skops_path},
                "pip_requirements": ["scikit-learn", "xgboost", "skops"],
            }
            if signature is not None:
                log_kwargs["signature"] = signature
            # input_example omitido intencionalmente con skops
            # (puede fallar al validar tipos de XGBoost durante el logueo)

            mlflow.pyfunc.log_model(**log_kwargs)
            print("✅ Modelo pyfunc (skops) logueado en MLflow")
            return

        except Exception as e:
            print(f"⚠️  Error con skops: {e} — usando fallback xgboost nativo")

    # ── Formato nativo XGBoost (recomendado, sin pickle) ─────────────────────
    try:
        try:
            mlflow_xgboost.log_model(
                xgb_model     = model,
                name          = "model",
                signature     = signature,
                input_example = input_example,
            )
        except TypeError:
            # Versiones antiguas de MLflow usan artifact_path en lugar de name
            mlflow_xgboost.log_model(
                xgb_model     = model,
                artifact_path = "model",
                signature     = signature,
                input_example = input_example,
            )
        print("✅ Modelo XGBoost nativo logueado en MLflow")
        return

    except Exception as e:
        print(f"⚠️  Error con mlflow.xgboost: {e} — usando fallback sklearn/pickle")

    # ── Fallback: sklearn / pickle ────────────────────────────────────────────
    try:
        try:
            mlflow_sklearn.log_model(
                sk_model      = model,
                name          = "model",
                signature     = signature,
                input_example = input_example,
            )
        except TypeError:
            mlflow_sklearn.log_model(
                sk_model      = model,
                artifact_path = "model",
                signature     = signature,
                input_example = input_example,
            )
        print("✅ Modelo sklearn (pickle) logueado en MLflow [fallback]")

    except Exception as e:
        print(f"❌ No se pudo loguear el modelo en MLflow: {e}")


def print_feature_importance(model, feature_names, top_n=5):
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    print(f"\n🔍 Top {top_n} features más importantes:")
    print(importance.head(top_n).to_string(index=False))


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Iniciando entrenamiento...\n")

    X_train, X_test, y_train, y_test = load_data()

    params = {
        "n_estimators":         300,
        "max_depth":            6,
        "learning_rate":        0.05,
        "subsample":            0.8,
        "colsample_bytree":     0.8,
        "random_state":         42,
        "early_stopping_rounds": 20,
    }

    # ── MLflow setup ─────────────────────────────────────────────────────────
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        print(f"MLflow tracking URI: {mlflow_uri}")
    else:
        print("MLflow tracking URI no configurado — usando entorno local por defecto")

    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "mlops-pipeline-experiment")
    mlflow.set_experiment(experiment_name)

    run_name = f"train-{os.getenv('GITHUB_SHA', 'local')}"

    # ── Run ──────────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)

        model   = train_model(X_train, X_test, y_train, y_test, params)
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        mlflow.set_tag("model.flavor", "xgboost")
        mlflow.log_text("\n".join(X_train.columns.tolist()), "feature_names.txt")

        # Guardar localmente (una sola vez)
        save_model_local(model, out_path="models/model.pkl")

        # Inferir firma (una sola vez)
        signature     = build_signature(X_train, model)
        input_example = X_train.head(1)

        # Loguear en MLflow
        log_model_to_mlflow(model, signature, input_example)

        run_id = run.info.run_id
        print(f"\n✅ MLflow run finalizado: {run_id}")

        # Registro opcional en Model Registry
        registry_name = os.getenv("MLFLOW_REGISTER_NAME")
        if registry_name:
            model_uri = f"runs:/{run_id}/model"
            try:
                mv = mlflow.register_model(model_uri, registry_name)
                print(f"✅ Modelo registrado: {registry_name} v{mv.version}")
            except Exception as e:
                print(f"⚠️  Error en Model Registry: {e}")

    # ── Post-run ─────────────────────────────────────────────────────────────
    print_feature_importance(model, X_train.columns.tolist())

    print(f"\n{'='*45}")
    if metrics["f1"] >= 0.85:
        print(f"✅ MODELO APROBADO  — F1: {metrics['f1']}")
    else:
        print(f"❌ MODELO RECHAZADO — F1: {metrics['f1']} (mínimo requerido: 0.85)")