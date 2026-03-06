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

def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()
    print(f"✅ Datos cargados:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    model = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = 42,
        early_stopping_rounds = 20,
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

def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("\n✅ Modelo guardado en models/model.pkl")

def print_feature_importance(model, feature_names):
    importance = pd.DataFrame({
        'feature':    feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🔍 Top 5 features más importantes:")
    print(importance.head(5).to_string(index=False))

if __name__ == "__main__":
    print("🚀 Iniciando entrenamiento...\n")

    X_train, X_test, y_train, y_test = load_data()
    model   = train_model(X_train, X_test, y_train, y_test)
    metrics = evaluate_model(model, X_test, y_test)

    print_feature_importance(model, X_train.columns.tolist())
    save_model(model)

    print(f"\n{'='*45}")
    if metrics['f1'] >= 0.85:
        print(f"✅ MODELO APROBADO  — F1: {metrics['f1']}")
        print(f"   Listo para deploy a producción")
    else:
        print(f"❌ MODELO RECHAZADO — F1: {metrics['f1']}")
        print(f"   Requiere ajuste")
    print(f"{'='*45}")