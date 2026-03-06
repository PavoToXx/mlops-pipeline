import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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
    print(f"   TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)
    classes = ['Normal', 'Fallo']
    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicción', ylabel='Real',
           title='Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black',
                    fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Guardada: {path}")

def plot_roc_curve(y_test, y_proba, output_dir):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC AUC = {auc:.4f}')
    ax.plot([0,1], [0,1], 'k--', lw=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    ax.set(xlabel='FPR', ylabel='TPR', title='ROC Curve',
           xlim=[0,1], ylim=[0,1.02])
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Guardada: {path}")

def save_metrics_json(metrics, output_dir):
    path = os.path.join(output_dir, "metrics.json")
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Métricas guardadas: {path}")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    model, scaler, X_test, y_test = load_artifacts()
    metrics, y_pred, y_proba = full_evaluation(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, "reports")
    plot_roc_curve(y_test, y_proba, "reports")
    save_metrics_json(metrics, "reports")
    print(f"\n{'='*50}")
    print("📋 RESUMEN:")
    for k, v in metrics.items():
        print(f"   ✅ {k:12}: {v}")
    print(f"{'='*50}")