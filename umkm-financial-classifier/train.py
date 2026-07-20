"""
train.py
Script pelatihan pipeline Gaussian Naïve Bayes untuk klasifikasi kondisi
keuangan UMKM. Jalankan sekali sebelum menjalankan app.py.

Penggunaan:
    python train.py
"""

import os
import sys
import time
import json
import numpy as np

# Pastikan src bisa diimport dari direktori proyek
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_dataset, select_features, split_data
from src.model import build_pipeline, train_pipeline, save_pipeline
from src.evaluation import (
    compute_metrics,
    compute_confusion_matrix,
    compute_roc_curve,
    cross_validate,
)

# ── Konfigurasi ──────────────────────────────────────────────────────────────

DATASET_PATH  = os.path.join("data", "raw", "dataset_umkm.csv")
MODEL_PATH    = os.path.join("models", "gnb_pipeline.pkl")
METRICS_PATH  = os.path.join("models", "metrics.json")


def main():
    print("=" * 60)
    print("  TRAINING PIPELINE — SISTEM KLASIFIKASI KEUANGAN UMKM")
    print("=" * 60)
    t_start = time.time()

    # ── 1. Load Dataset ──────────────────────────────────────────
    print(f"\n[1/5] Memuat dataset dari: {DATASET_PATH}")
    df = load_dataset(DATASET_PATH)
    print(f"      Jumlah baris  : {len(df):,}")
    print(f"      Jumlah kolom  : {len(df.columns)}")

    # ── 2. Seleksi Fitur & Binarisasi Label ─────────────────────
    print("\n[2/5] Seleksi fitur & binarisasi label...")
    X, y = select_features(df)
    print(f"      Distribusi kelas: Sehat={int(y.sum()):,} | Tidak Sehat={int((y==0).sum()):,}")

    # ── 3. Stratified Split (SEBELUM normalisasi) ────────────────
    print("\n[3/5] Stratified split 80:20 (sebelum normalisasi)...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"      Train: {len(X_train):,} baris | Test: {len(X_test):,} baris")
    print(f"      Proporsi kelas train: {y_train.mean():.4f} | test: {y_test.mean():.4f}")

    # ── 4. Cross-Validation (pada data latih saja) ───────────────
    print("\n[4/5] 10-Fold Stratified Cross-Validation (data latih)...")
    cv_results = cross_validate(X_train, y_train, n_splits=10, random_state=42)
    print(f"      CV Accuracy : {cv_results['mean_acc']:.4f} ± {cv_results['std_acc']:.4f}")
    print(f"      CV F1-Score : {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
    print(f"      Skor per fold: {[f'{s:.4f}' for s in cv_results['scores']]}")

    # ── 5. Training Pipeline Final & Evaluasi ───────────────────
    print("\n[5/5] Training pipeline final (MinMaxScaler + GaussianNB)...")
    pipeline = build_pipeline()
    pipeline = train_pipeline(pipeline, X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    metrics = compute_metrics(y_test.values, y_pred, y_proba)
    cm      = compute_confusion_matrix(y_test.values, y_pred)
    fpr, tpr, auc_score = compute_roc_curve(y_test.values, y_proba[:, 1])

    print("\n  Hasil Evaluasi pada Test Set:")
    print(f"    Accuracy  : {metrics['accuracy']:.4f}")
    print(f"    Precision : {metrics['precision']:.4f}")
    print(f"    Recall    : {metrics['recall']:.4f}")
    print(f"    F1-Score  : {metrics['f1_weighted']:.4f}")
    print(f"    ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    # ── Simpan Model ─────────────────────────────────────────────
    save_pipeline(pipeline, MODEL_PATH)
    print(f"\n  Model disimpan ke: {MODEL_PATH}")

    # ── Simpan Metrik ────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    all_metrics = {
        **metrics,
        **cv_results,
        "confusion_matrix": cm.tolist(),
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "total_train": len(X_train),
        "total_test":  len(X_test),
        "training_time_seconds": round(time.time() - t_start, 2),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Metrik disimpan ke: {METRICS_PATH}")

    elapsed = round(time.time() - t_start, 2)
    print(f"\n{'=' * 60}")
    print(f"  Training selesai dalam {elapsed} detik.")
    print(f"  Jalankan: python app.py  →  http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
