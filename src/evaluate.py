import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Sin pantalla (para CI/CD)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

def load_artifacts():
    model  = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    print("✅ Artefactos cargados")
    return model, scaler, X_test, y_test

def full_evaluation(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)),  4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall":    round(float(recall_score(y_test, y_pred)),    4),
        "f1":        round(float(f1_score(y_test, y_pred)),        4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_proba)),  4),
    }

    print("\n📊 Reporte completo:")
    print(classification_report(y_test, y_pred,
          target_names=["Normal", "Fallo"]))

    return metrics, y_pred, y_proba

def plot_confusion_matrix(y_test, y_pred, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n🔢 Confusion Matrix:")
    print(f"   Verdaderos Negativos (TN): {tn}  → Normal predicho como Normal")
    print(f"   Falsos Positivos     (FP): {fp}  → Normal predicho como Fallo")
    print(f"   Falsos Negativos     (FN): {fn}  → Fallo predicho como Normal ⚠️")
    print(f"   Verdaderos Positivos (TP): {tp}  → Fallo predicho como Fallo")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)

    classes = ['Normal', 'Fallo']
    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicción', ylabel='Real',
           title='Confusion Matrix — Server Failure Prediction')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]),
                    ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black',
                    fontsize=16, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix guardada: {path}")
    return path

def plot_roc_curve(y_test, y_proba, output_dir):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#2196F3', lw=2,
            label=f'ROC AUC = {auc:.4f}')
    ax.plot([0,1], [0,1], 'k--', lw=1, label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    ax.set(xlabel='False Positive Rate',
           ylabel='True Positive Rate',
           title='ROC Curve — Server Failure Prediction',
           xlim=[0,1], ylim=[0,1.02])
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ ROC curve guardada: {path}")
    return path

def plot_feature_importance(model, feature_names, output_dir):
    importance = pd.DataFrame({
        'feature':    feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(importance['feature'], importance['importance'],
                   color='#2196F3', edgecolor='white')
    ax.set(xlabel='Importance', title='Top 10 Feature Importance')
    ax.bar_label(bars, fmt='%.3f', padding=3)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Feature importance guardada: {path}")
    return path

def save_metrics_json(metrics, output_dir):
    path = os.path.join(output_dir, "metrics.json")
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Métricas guardadas: {path}")
    return path

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)

    # 1. Carga
    model, scaler, X_test, y_test = load_artifacts()

    # 2. Evalúa
    metrics, y_pred, y_proba = full_evaluation(model, X_test, y_test)

    # 3. Gráficas
    plot_confusion_matrix(y_test, y_pred,   "reports")
    plot_roc_curve(y_test, y_proba,          "reports")
    plot_feature_importance(model, X_test.columns.tolist(), "reports")

    # 4. JSON con métricas
    save_metrics_json(metrics, "reports")

    # 5. Resumen final
    print(f"\n{'='*50}")
    print(f"📋 RESUMEN EVALUACIÓN:")
    for k, v in metrics.items():
        status = "✅" if v >= 0.90 else "⚠️"
        print(f"   {status} {k:12}: {v}")
    print(f"\n📁 Reportes guardados en reports/")
    print(f"{'='*50}")