"""
train.py
Script pelatihan Gaussian Naïve Bayes untuk klasifikasi 4 kelas kondisi
keuangan UMKM (Critical, Elite, Growth, Struggling) dengan 5 fitur hasil
Sequential Backward Selection. Jalankan sekali sebelum menjalankan app.py.

Penggunaan:
    python train.py

Sesuai naskah skripsi:
  - Split stratified 80/20, random_state=42 (test = 30.000 baris)
  - Cross-validation 10-fold dihitung pada seluruh dataset
    (menghasilkan Acc 88,84% ± 0,30%, Macro F1 85,17% ± 0,31%)
  - Model final tanpa scaler (GaussianNB murni)
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_dataset, select_features, split_data, KELAS, LABEL_TEKS
from src.model import build_pipeline, train_pipeline, save_pipeline
from src.evaluation import (
    compute_metrics,
    compute_confusion_matrix,
    compute_roc_curves,
    cross_validate,
)

DATASET_PATH = os.path.join("data", "raw", "dataset_umkm.csv")
MODEL_PATH = os.path.join("models", "gnb_pipeline.pkl")
METRICS_PATH = os.path.join("models", "metrics.json")
ROC_POINTS = 250  # downsample titik kurva ROC agar metrics.json ringkas


def _downsample(roc: dict) -> dict:
    out = {}
    for nama, d in roc.items():
        fpr = np.asarray(d["fpr"]); tpr = np.asarray(d["tpr"])
        if len(fpr) > ROC_POINTS:
            idx = np.unique(np.linspace(0, len(fpr) - 1, ROC_POINTS).astype(int))
            fpr, tpr = fpr[idx], tpr[idx]
        out[nama] = {"fpr": [round(float(v), 5) for v in fpr],
                     "tpr": [round(float(v), 5) for v in tpr],
                     "auc": d["auc"]}
    return out


def main():
    print("=" * 60)
    print("  TRAINING — KLASIFIKASI 4 KELAS KEUANGAN UMKM (GNB)")
    print("=" * 60)
    t_start = time.time()

    print(f"\n[1/5] Memuat dataset dari: {DATASET_PATH}")
    df = load_dataset(DATASET_PATH)
    print(f"      Jumlah baris  : {len(df):,}")
    print(f"      Jumlah kolom  : {len(df.columns)}")

    print("\n[2/5] Seleksi 5 fitur SBS & pemetaan label 4 kelas...")
    X, y = select_features(df)
    for nama in KELAS:
        kode = [k for k, v in LABEL_TEKS.items() if v == nama][0]
        print(f"      {nama:<11}: {int((y == kode).sum()):,} baris")

    print("\n[3/5] Stratified split 80:20 (random_state=42)...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"      Train: {len(X_train):,} baris | Test: {len(X_test):,} baris")

    print("\n[4/5] 10-Fold Stratified Cross-Validation (seluruh dataset, sesuai naskah)...")
    cv_results = cross_validate(X, y, n_splits=10, random_state=42)
    print(f"      CV Accuracy : {cv_results['mean_acc']:.4f} ± {cv_results['std_acc']:.4f}")
    print(f"      CV Macro F1 : {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")

    print("\n[5/5] Training model final (GaussianNB, tanpa scaler)...")
    pipeline = build_pipeline()
    pipeline = train_pipeline(pipeline, X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    metrics = compute_metrics(y_test.values, y_pred, y_proba)
    cm = compute_confusion_matrix(y_test.values, y_pred)
    roc = compute_roc_curves(y_test.values, y_proba)

    print("\n  Hasil Evaluasi pada Test Set (30.000 baris):")
    print(f"    Accuracy        : {metrics['accuracy']:.4f}")
    print(f"    Macro Precision : {metrics['precision_macro']:.4f}")
    print(f"    Macro Recall    : {metrics['recall_macro']:.4f}")
    print(f"    Macro F1        : {metrics['f1_macro']:.4f}")
    print(f"    Weighted F1     : {metrics['f1_weighted']:.4f}")
    print(f"    ROC-AUC (OvR)   : {metrics['roc_auc']:.4f}")
    print("\n  Confusion Matrix (baris=aktual, kolom=prediksi):")
    print(f"    Urutan kelas: {KELAS}")
    for i, nama in enumerate(KELAS):
        print(f"    {nama:<11}: {cm[i].tolist()}")

    save_pipeline(pipeline, MODEL_PATH)
    print(f"\n  Model disimpan ke: {MODEL_PATH}")

    os.makedirs("models", exist_ok=True)
    all_metrics = {
        **metrics,
        **cv_results,
        "confusion_matrix": cm.tolist(),
        "classes": KELAS,
        "roc_curves": _downsample(roc),
        "total_data": len(X),
        "total_train": len(X_train),
        "total_test": len(X_test),
        "training_time_seconds": round(time.time() - t_start, 2),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Metrik disimpan ke: {METRICS_PATH}")

    elapsed = round(time.time() - t_start, 2)
    print(f"\n{'=' * 60}")
    print(f"  Training selesai dalam {elapsed} detik.")
    print("  Jalankan: python app.py  →  http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
