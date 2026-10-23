"""
evaluation.py
Menghitung metrik evaluasi multiclass (4 kelas) dan cross-validation:
accuracy, macro/weighted precision-recall-F1, ROC-AUC One-vs-Rest,
confusion matrix 4x4, dan metrik per kelas.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize

from src.preprocessing import KELAS, LABEL_TEKS

LABEL_ORDER = sorted(LABEL_TEKS.keys())  # [0, 1, 2, 3]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Hitung metrik evaluasi multiclass.
    y_proba: array shape (n, 4) mengikuti urutan classes_ model
    (indeks kelas 0..3 = Critical, Elite, Growth, Struggling).
    Return dict metrik agregat + per kelas + ROC-AUC OvR.
    """
    per_class = {}
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    _, support = np.unique(y_true, return_counts=True)
    # urutan unik y_true harus sejajar LABEL_ORDER bila semua kelas hadir
    present = sorted(np.unique(y_true).tolist())
    for i, k in enumerate(present):
        nama = LABEL_TEKS[int(k)]
        per_class[nama] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    roc_per_class = {}
    try:
        aucs = roc_auc_score(y_true, y_proba, multi_class="ovr", average=None)
        for i, k in enumerate(LABEL_ORDER):
            roc_per_class[LABEL_TEKS[k]] = round(float(aucs[i]), 4)
        roc_macro = round(float(roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro")), 4)
    except ValueError:
        roc_macro = 0.0

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(
            y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(
            y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0)), 4),
        # alias agar konsumen lama (app.py/dashboard) tetap terbaca
        "precision": round(float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1": round(float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "roc_auc": roc_macro,
        "roc_auc_per_class": roc_per_class,
        "per_class": per_class,
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return confusion matrix shape (4, 4) dengan urutan LABEL_ORDER."""
    return confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)


def compute_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Kurva ROC One-vs-Rest per kelas.
    Return: {nama_kelas: {"fpr": [...], "tpr": [...], "auc": float}}
    """
    y_bin = label_binarize(y_true, classes=LABEL_ORDER)
    out = {}
    for i, k in enumerate(LABEL_ORDER):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        out[LABEL_TEKS[k]] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(float(roc_auc_score(y_bin[:, i], y_proba[:, i])), 4),
        }
    return out


def compute_roc_curve(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
):
    """Legacy biner — dipertahankan untuk kompatibilitas; gunakan compute_roc_curves."""
    from sklearn.metrics import auc
    fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
    return fpr, tpr, round(float(auc(fpr, tpr)), 4)


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 10,
    random_state: int = 42,
) -> dict:
    """
    Stratified K-Fold CV. Setiap fold membuat Pipeline GaussianNB baru.
    Sesuai naskah, CV dijalankan pada seluruh dataset (dipanggil dari train.py).
    Return: {scores, mean_acc, std_acc, f1_scores, mean_f1, std_f1, n_splits}
    (mean_f1/std_f1 = macro F1).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_scores = []
    f1_scores = []

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    for train_idx, val_idx in skf.split(X_arr, y_arr):
        Xf_train, Xf_val = X_arr[train_idx], X_arr[val_idx]
        yf_train, yf_val = y_arr[train_idx], y_arr[val_idx]

        fold_pipe = Pipeline([("gnb", GaussianNB())])
        fold_pipe.fit(Xf_train, yf_train)
        y_pred = fold_pipe.predict(Xf_val)

        acc_scores.append(accuracy_score(yf_val, y_pred))
        f1_scores.append(f1_score(yf_val, y_pred, average="macro", zero_division=0))

    return {
        "scores": [round(s, 4) for s in acc_scores],
        "mean_acc": round(float(np.mean(acc_scores)), 4),
        "std_acc": round(float(np.std(acc_scores)), 4),
        "f1_scores": [round(s, 4) for s in f1_scores],
        "mean_f1": round(float(np.mean(f1_scores)), 4),
        "std_f1": round(float(np.std(f1_scores)), 4),
        "n_splits": n_splits,
    }
