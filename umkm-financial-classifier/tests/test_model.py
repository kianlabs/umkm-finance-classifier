"""
test_model.py
Unit tests untuk modul src/model.py (GaussianNB 4 kelas, tanpa scaler)
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import load_dataset, select_features, split_data, KELAS, LABEL_TEKS
from src.model import (
    build_pipeline,
    train_pipeline,
    save_pipeline,
    load_pipeline,
    predict_single,
    predict_dataframe,
)

DATASET_PATH = os.path.join("data", "raw", "dataset_umkm.csv")


@pytest.fixture(scope="module")
def trained_pipeline():
    if not os.path.exists(DATASET_PATH):
        pytest.skip(f"Dataset tidak ditemukan: {DATASET_PATH}")
    df = load_dataset(DATASET_PATH)
    X, y = select_features(df)
    X_train, _, y_train, _ = split_data(X, y)
    pipe = build_pipeline()
    return train_pipeline(pipe, X_train, y_train)


@pytest.fixture(scope="module")
def test_data():
    if not os.path.exists(DATASET_PATH):
        pytest.skip(f"Dataset tidak ditemukan: {DATASET_PATH}")
    df = load_dataset(DATASET_PATH)
    X, y = select_features(df)
    _, X_test, _, y_test = split_data(X, y)
    return X_test, y_test


# ── Test build_pipeline ───────────────────────────────────────────────────────

def test_build_pipeline_hanya_gnb():
    """Konfigurasi final naskah: GaussianNB murni, TANPA scaler."""
    pipe = build_pipeline()
    assert "gnb" in pipe.named_steps
    assert "scaler" not in pipe.named_steps


def test_build_pipeline_gnb_type():
    from sklearn.naive_bayes import GaussianNB
    pipe = build_pipeline()
    assert isinstance(pipe.named_steps["gnb"], GaussianNB)


# ── Test train_pipeline ───────────────────────────────────────────────────────

def test_pipeline_fitted(trained_pipeline):
    gnb = trained_pipeline.named_steps["gnb"]
    assert hasattr(gnb, "theta_"), "GNB belum di-fit"


def test_gnb_theta_shape(trained_pipeline):
    gnb = trained_pipeline.named_steps["gnb"]
    assert gnb.theta_.shape == (4, 5), f"Shape theta_ salah: {gnb.theta_.shape}"


def test_gnb_classes(trained_pipeline):
    gnb = trained_pipeline.named_steps["gnb"]
    assert list(gnb.classes_) == [0, 1, 2, 3]


# ── Test predict_proba ────────────────────────────────────────────────────────

def test_predict_proba_sum_to_one(trained_pipeline, test_data):
    X_test, _ = test_data
    proba = trained_pipeline.predict_proba(X_test.head(100))
    assert proba.shape == (100, 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_label_consistent_with_argmax(trained_pipeline, test_data):
    X_test, _ = test_data
    sample = X_test.head(100)
    labels = trained_pipeline.predict(sample)
    probas = trained_pipeline.predict_proba(sample)
    assert np.array_equal(labels, np.argmax(probas, axis=1))


# ── Test save / load ──────────────────────────────────────────────────────────

def test_save_load_pipeline(trained_pipeline, test_data, tmp_path):
    X_test, _ = test_data
    path = str(tmp_path / "test_pipeline.pkl")

    save_pipeline(trained_pipeline, path)
    loaded = load_pipeline(path)

    sample = X_test.head(200)
    assert np.array_equal(trained_pipeline.predict(sample), loaded.predict(sample))
    assert np.allclose(
        trained_pipeline.predict_proba(sample),
        loaded.predict_proba(sample),
        atol=1e-10,
    )


def test_load_pipeline_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_pipeline("path/tidak/ada.pkl")


# ── Test predict_single ───────────────────────────────────────────────────────

SAMPLE_INPUT = {
    "Net_Profit_Margin": 12.5,
    "Burn_Rate_Ratio": 0.82,
    "Avg_Historical_Rating": 4.3,
    "Business_Tenure_Months": 24,
    "Repeat_Order_Rate": 35.0,
}


def test_predict_single_keys(trained_pipeline):
    result = predict_single(trained_pipeline, SAMPLE_INPUT)
    for key in ["label", "label_teks", "probabilitas", "prob_max", "rekomendasi"]:
        assert key in result, f"Kunci '{key}' tidak ada"


def test_predict_single_label_four_class(trained_pipeline):
    result = predict_single(trained_pipeline, SAMPLE_INPUT)
    assert result["label"] in {0, 1, 2, 3}
    assert result["label_teks"] in KELAS


def test_predict_single_prob_sum(trained_pipeline):
    result = predict_single(trained_pipeline, SAMPLE_INPUT)
    total = sum(result["probabilitas"].values())
    assert abs(total - 1.0) < 1e-3
    assert set(result["probabilitas"].keys()) == set(KELAS)


def test_predict_single_label_teks_konsisten(trained_pipeline):
    result = predict_single(trained_pipeline, SAMPLE_INPUT)
    assert result["label_teks"] == LABEL_TEKS[result["label"]]
    assert result["prob_max"] == max(result["probabilitas"].values())


# ── Test predict_dataframe ────────────────────────────────────────────────────

def test_predict_dataframe_row_count(trained_pipeline, test_data):
    X_test, _ = test_data
    sample = X_test.head(500)
    result = predict_dataframe(trained_pipeline, sample)
    assert len(result) == len(sample)


def test_predict_dataframe_columns(trained_pipeline, test_data):
    X_test, _ = test_data
    result = predict_dataframe(trained_pipeline, X_test.head(10))
    for col in ["Prediksi_Label", "Prediksi_Teks",
                "Prob_Critical", "Prob_Elite", "Prob_Growth", "Prob_Struggling", "Prob_Max"]:
        assert col in result.columns, f"Kolom '{col}' tidak ada"
    assert set(result["Prediksi_Teks"].unique()).issubset(set(KELAS))
