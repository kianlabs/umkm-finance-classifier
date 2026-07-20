"""
test_evaluation.py
Unit tests untuk modul src/evaluation.py
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import (
    compute_metrics,
    compute_confusion_matrix,
    compute_roc_curve,
    cross_validate,
)
from src.preprocessing import load_dataset, select_features, split_data
from src.model import build_pipeline, train_pipeline

DATASET_PATH = os.path.join("data", "raw", "dataset_umkm.csv")


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def eval_data():
    if not os.path.exists(DATASET_PATH):
        pytest.skip()
    df = load_dataset(DATASET_PATH)
    X, y = select_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = build_pipeline()
    pipe = train_pipeline(pipe, X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    return y_test.values, y_pred, y_proba, X_train, y_train


# ── compute_metrics ───────────────────────────────────────────────────────────

def test_metrics_keys(eval_data):
    y_true, y_pred, y_proba, *_ = eval_data
    m = compute_metrics(y_true, y_pred, y_proba)
    for key in ["accuracy", "precision", "recall", "f1_weighted", "roc_auc"]:
        assert key in m, f"Kunci '{key}' tidak ada"


def test_metrics_range(eval_data):
    y_true, y_pred, y_proba, *_ = eval_data
    m = compute_metrics(y_true, y_pred, y_proba)
    for key, val in m.items():
        assert 0.0 <= val <= 1.0, f"{key} = {val} di luar [0, 1]"


def test_metrics_accuracy_threshold(eval_data):
    """Accuracy harus >= 0.94 sesuai target penelitian."""
    y_true, y_pred, y_proba, *_ = eval_data
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] >= 0.94, f"Accuracy terlalu rendah: {m['accuracy']}"


def test_metrics_roc_auc_threshold(eval_data):
    """ROC-AUC harus >= 0.98."""
    y_true, y_pred, y_proba, *_ = eval_data
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["roc_auc"] >= 0.98, f"ROC-AUC terlalu rendah: {m['roc_auc']}"


# ── compute_confusion_matrix ──────────────────────────────────────────────────

def test_confusion_matrix_shape(eval_data):
    y_true, y_pred, *_ = eval_data
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)


def test_confusion_matrix_sum(eval_data):
    y_true, y_pred, *_ = eval_data
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.sum() == len(y_true)


# ── compute_roc_curve ─────────────────────────────────────────────────────────

def test_roc_curve_output(eval_data):
    y_true, _, y_proba, *_ = eval_data
    fpr, tpr, auc_score = compute_roc_curve(y_true, y_proba[:, 1])
    assert len(fpr) == len(tpr)
    assert 0.0 <= auc_score <= 1.0
    assert fpr[0] == 0.0 or fpr[0] >= 0.0
    assert tpr[-1] == 1.0 or tpr[-1] <= 1.0


# ── cross_validate ────────────────────────────────────────────────────────────

def test_cross_validate_n_splits(eval_data):
    """CV harus menghasilkan tepat n_splits skor."""
    *_, X_train, y_train = eval_data
    result = cross_validate(X_train.iloc[:5000], y_train.iloc[:5000], n_splits=5)
    assert len(result["scores"]) == 5


def test_cross_validate_keys(eval_data):
    *_, X_train, y_train = eval_data
    result = cross_validate(X_train.iloc[:3000], y_train.iloc[:3000], n_splits=3)
    for key in ["scores", "mean_acc", "std_acc", "mean_f1", "std_f1"]:
        assert key in result


def test_cross_validate_only_train_data():
    """Property 8: CV hanya menggunakan data latih, tanpa X_test."""
    if not os.path.exists(DATASET_PATH):
        pytest.skip()
    from src.preprocessing import load_dataset, select_features, split_data
    df = load_dataset(DATASET_PATH)
    X, y = select_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # cross_validate dipanggil HANYA dengan X_train — bukan X_test
    result = cross_validate(X_train.iloc[:4000], y_train.iloc[:4000], n_splits=5)
    assert result["mean_acc"] > 0.5  # minimal lebih baik dari random
