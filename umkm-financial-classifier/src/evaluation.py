"""
evaluation.py
Menghitung semua metrik evaluasi dan cross-validation.
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
    auc,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Hitung metrik evaluasi model.
    y_proba: array shape (n, 2) — kolom 1 adalah P(Sehat)
    Return: {accuracy, precision, recall, f1_weighted, roc_auc}
    """
    return {
        "accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
        "precision":   round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall":      round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "roc_auc":     round(float(roc_auc_score(y_true, y_proba[:, 1])), 4),
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Return confusion matrix shape (2, 2): [[TN, FP], [FN, TP]]
    """
    return confusion_matrix(y_true, y_pred)


def compute_roc_curve(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
) -> tuple:
    """
    y_proba_pos: probabilitas kelas positif (Sehat = 1)
    Return: (fpr_array, tpr_array, auc_score)
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
    auc_score   = round(float(auc(fpr, tpr)), 4)
    return fpr, tpr, auc_score


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 10,
    random_state: int = 42,
) -> dict:
    """
    10-Fold Stratified CV pada data latih saja.
    Setiap fold membuat Pipeline baru (MinMaxScaler + GaussianNB)
    untuk mencegah leakage antar fold.
    Return: {mean_acc, std_acc, scores_per_fold, mean_f1, std_f1}
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_scores = []
    f1_scores  = []

    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series)    else y

    for train_idx, val_idx in skf.split(X_arr, y_arr):
        Xf_train, Xf_val = X_arr[train_idx], X_arr[val_idx]
        yf_train, yf_val = y_arr[train_idx], y_arr[val_idx]

        # Pipeline baru per fold untuk mencegah leakage
        fold_pipe = Pipeline([
            ("scaler", MinMaxScaler()),
            ("gnb",    GaussianNB()),
        ])
        fold_pipe.fit(Xf_train, yf_train)
        y_pred = fold_pipe.predict(Xf_val)

        acc_scores.append(accuracy_score(yf_val, y_pred))
        f1_scores.append(f1_score(yf_val, y_pred, average="weighted", zero_division=0))

    return {
        "scores":       [round(s, 4) for s in acc_scores],
        "mean_acc":     round(float(np.mean(acc_scores)), 4),
        "std_acc":      round(float(np.std(acc_scores)), 4),
        "mean_f1":      round(float(np.mean(f1_scores)), 4),
        "std_f1":       round(float(np.std(f1_scores)), 4),
        "n_splits":     n_splits,
    }
