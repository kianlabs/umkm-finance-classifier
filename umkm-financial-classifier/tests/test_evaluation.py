"""
test_evaluation.py
Unit tests untuk modul src/evaluation.py (metrik multiclass 4 kelas)
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import (
    compute_metrics,
    compute_confusion_matrix,
    compute_roc_curves,
    cross_validate,
)
from src.preprocessing import load_dataset, select_features, split_data, KELAS
from src.model import build_pipeline, train_pipeline

DATASET_PATH = os.path.join("data", "raw", "dataset_umkm.csv")


@pytest.fixture(scope="module")
def eval_data():
    if not os.path.exists(DATASET_PATH):
        pytest.skip()
    df = load_dataset(DATASET_PATH)
    X, y = select_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = build_pipeline()
    pipe = train_pipeline(pipe, X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    return y_test.values, y_pred, y_proba, X_train, y_train


# ── compute_metrics ───────────────────────────────────────────────────────────

def test_metrics_keys(eval_data):
    y_true, y_pred, y_proba, *_ = eval_data
    m = compute_metrics(y_true, y_pred, y_proba)
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                "f1_weighted", "roc_auc", "roc_auc_per_class", "per_class"]:
        assert key in m, f"Kunci '{key}' tidak ada"


def test_metrics_range(eval_data):
    y_true, y_pred, y_proba, *_ = eval_data
    m = compute_metrics(y_true, y_pred, y_proba)
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                "f1_weighted", "roc_auc"]:
        assert 0.0 <= m[key] <= 1.0, f"{key} = {m[key]} di luar [0, 1]"


def test_metrics_accuracy_threshold(eval_data):
    """Accuracy 4 kelas harus >= 0,88 (naskah: 0,888767)."""
    y_true, y_pred, y_proba, *_ = eval_data
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] >= 0.88, f"Accuracy terlalu rendah: {m['accuracy']}"


def test_metrics_roc_auc_threshold(eval_data):
    """Macro ROC-AUC OvR harus >= 0,97 (naskah: 0,9791)."""
    y_true, y_pred, y_proba, *_ = eval_data
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["roc_auc"] >= 0.97, f"ROC-AUC terlalu rendah: {m['roc_auc']}"


def test_metrics_per_class(eval_data):
    y_true, y_pred, y_proba, *_ = eval_data
    m = compute_metrics(y_true, y_pred, y_proba)
    assert set(m["per_class"].keys()) == set(KELAS)
    total_support = sum(v["support"] for v in m["per_class"].values())
    assert total_support == len(y_true)


# ── compute_confusion_matrix ──────────────────────────────────────────────────

def test_confusion_matrix_shape(eval_data):
    y_true, y_pred, *_ = eval_data
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (4, 4)


def test_confusion_matrix_sum(eval_data):
    y_true, y_pred, *_ = eval_data
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.sum() == len(y_true)


# ── compute_roc_curves ────────────────────────────────────────────────────────

def test_roc_curves_per_class(eval_data):
    y_true, _, y_proba, *_ = eval_data
    roc = compute_roc_curves(y_true, y_proba)
    assert set(roc.keys()) == set(KELAS)
    for nama, d in roc.items():
        assert len(d["fpr"]) == len(d["tpr"])
        assert 0.0 <= d["auc"] <= 1.0


# ── cross_validate ────────────────────────────────────────────────────────────

def test_cross_validate_n_splits(eval_data):
    *_, X_train, y_train = eval_data
    result = cross_validate(X_train.iloc[:5000], y_train.iloc[:5000], n_splits=5)
    assert len(result["scores"]) == 5
    assert len(result["f1_scores"]) == 5


def test_cross_validate_keys(eval_data):
    *_, X_train, y_train = eval_data
    result = cross_validate(X_train.iloc[:3000], y_train.iloc[:3000], n_splits=3)
    for key in ["scores", "mean_acc", "std_acc", "mean_f1", "std_f1"]:
        assert key in result
