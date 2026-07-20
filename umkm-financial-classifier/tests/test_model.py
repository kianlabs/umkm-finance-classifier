"""
test_model.py
Unit tests untuk modul src/model.py
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import load_dataset, select_features, split_data
from src.model import (
    build_pipeline,
    train_pipeline,
    save_pipeline,
    load_pipeline,
    predict_single,
    predict_dataframe,
)

DATASET_PATH = os.path.join("data", "raw", "dataset_umkm.csv")
MODEL_PATH   = os.path.join("models", "gnb_pipeline.pkl")


# ── Fixture ───────────────────────────────────────────────────────────────────

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

def test_build_pipeline_steps():
    pipe = build_pipeline()
    assert "scaler" in pipe.named_steps
    assert "gnb"    in pipe.named_steps


def test_build_pipeline_scaler_type():
    from sklearn.preprocessing import MinMaxScaler
    pipe = build_pipeline()
    assert isinstance(pipe.named_steps["scaler"], MinMaxScaler)


def test_build_pipeline_gnb_type():
    from sklearn.naive_bayes import GaussianNB
    pipe = build_pipeline()
    assert isinstance(pipe.named_steps["gnb"], GaussianNB)


# ── Test train_pipeline ───────────────────────────────────────────────────────

def test_pipeline_fitted(trained_pipeline):
    scaler = trained_pipeline.named_steps["scaler"]
    gnb    = trained_pipeline.named_steps["gnb"]
    assert hasattr(scaler, "data_min_"), "Scaler belum di-fit"
    assert hasattr(gnb, "theta_"),       "GNB belum di-fit"


def test_pipeline_no_leakage(trained_pipeline):
    """
    Property 3: Scaler data_min_ dan data_max_ hanya dari X_train.
    """
    if not os.path.exists(DATASET_PATH):
        pytest.skip()
    df = load_dataset(DATASET_PATH)
    X, y = select_features(df)
    X_train, _, _, _ = split_data(X, y)

    scaler = trained_pipeline.named_steps["scaler"]
    assert np.allclose(scaler.data_min_, X_train.min().values, atol=1e-4)
    assert np.allclose(scaler.data_max_, X_train.max().values, atol=1e-4)


def test_gnb_theta_shape(trained_pipeline):
    gnb = trained_pipeline.named_steps["gnb"]
    assert gnb.theta_.shape == (2, 6), f"Shape theta_ salah: {gnb.theta_.shape}"


# ── Test predict_proba ────────────────────────────────────────────────────────

def test_predict_proba_sum_to_one(trained_pipeline, test_data):
    """Property 1: P(Sehat) + P(Tidak Sehat) == 1.0."""
    X_test, _ = test_data
    proba = trained_pipeline.predict_proba(X_test.head(100).values)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_label_consistent_with_argmax(trained_pipeline, test_data):
    """Property 2: label == argmax(predict_proba)."""
    X_test, _ = test_data
    sample = X_test.head(100).values
    labels = trained_pipeline.predict(sample)
    probas = trained_pipeline.predict_proba(sample)
    assert np.array_equal(labels, np.argmax(probas, axis=1))


# ── Test save / load ──────────────────────────────────────────────────────────

def test_save_load_pipeline(trained_pipeline, test_data, tmp_path):
    """Property 7: Prediksi identik setelah save + load."""
    X_test, _ = test_data
    path = str(tmp_path / "test_pipeline.pkl")

    save_pipeline(trained_pipeline, path)
    loaded = load_pipeline(path)

    sample = X_test.head(200).values
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
    "Monthly_Revenue":        6500000.0,
    "Net_Profit_Margin":      12.5,
    "Burn_Rate_Ratio":        0.82,
    "Transaction_Count":      120,
    "Repeat_Order_Rate":      35.0,
    "Business_Tenure_Months": 24,
}


def test_predict_single_keys(trained_pipeline):
    result = predict_single(trained_pipeline, SAMPLE_INPUT)
    assert "label"            in result
    assert "label_teks"       in result
    assert "prob_sehat"       in result
    assert "prob_tidak_sehat" in result
    assert "rekomendasi"      in result


def test_predict_single_label_binary(trained_pipeline):
    result = predict_single(trained_pipeline, SAMPLE_INPUT)
    assert result["label"] in {0, 1}


def test_predict_single_prob_sum(trained_pipeline):
    result = predict_single(trained_pipeline, SAMPLE_INPUT)
    assert abs(result["prob_sehat"] + result["prob_tidak_sehat"] - 1.0) < 1e-4


def test_predict_single_label_teks(trained_pipeline):
    result = predict_single(trained_pipeline, SAMPLE_INPUT)
    expected = "Sehat" if result["label"] == 1 else "Tidak Sehat"
    assert result["label_teks"] == expected


# ── Test predict_dataframe ────────────────────────────────────────────────────

def test_predict_dataframe_row_count(trained_pipeline, test_data):
    """Property 5: len(predict_dataframe(df)) == len(df)."""
    X_test, _ = test_data
    sample = X_test.head(500)
    result = predict_dataframe(trained_pipeline, sample)
    assert len(result) == len(sample)


def test_predict_dataframe_columns(trained_pipeline, test_data):
    X_test, _ = test_data
    result = predict_dataframe(trained_pipeline, X_test.head(10))
    assert "Prediksi_Label"   in result.columns
    assert "Prediksi_Teks"    in result.columns
    assert "Prob_Sehat"       in result.columns
    assert "Prob_Tidak_Sehat" in result.columns
