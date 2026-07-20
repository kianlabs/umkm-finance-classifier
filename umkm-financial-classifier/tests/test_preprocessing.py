"""
test_preprocessing.py
Unit tests untuk modul src/preprocessing.py
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import (
    FITUR_TERPILIH,
    LABEL_KOLOM,
    LABEL_MAP,
    load_dataset,
    select_features,
    split_data,
    validate_input,
    validate_csv_columns,
)

DATASET_PATH = os.path.join("data", "raw", "dataset_umkm.csv")


# ── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df():
    if not os.path.exists(DATASET_PATH):
        pytest.skip(f"Dataset tidak ditemukan: {DATASET_PATH}")
    return load_dataset(DATASET_PATH)


@pytest.fixture(scope="module")
def X_y(df):
    return select_features(df)


# ── Test load_dataset ─────────────────────────────────────────────────────────

def test_load_dataset_returns_dataframe(df):
    assert isinstance(df, pd.DataFrame)


def test_load_dataset_has_required_columns(df):
    for col in FITUR_TERPILIH:
        assert col in df.columns, f"Kolom '{col}' tidak ada"
    assert LABEL_KOLOM in df.columns


def test_load_dataset_not_empty(df):
    assert len(df) > 0


def test_load_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("path/yang/tidak/ada.csv")


# ── Test select_features ──────────────────────────────────────────────────────

def test_select_features_columns(X_y):
    X, y = X_y
    assert list(X.columns) == FITUR_TERPILIH


def test_select_features_binary_labels(X_y):
    X, y = X_y
    assert set(y.unique()).issubset({0, 1}), f"Label tidak biner: {y.unique()}"


def test_binarisasi_elite_growth_ke_1():
    """Elite dan Growth harus dipetakan ke 1."""
    df_test = pd.DataFrame({LABEL_KOLOM: ["Elite", "Growth", "Struggling", "Critical"]})
    for col in FITUR_TERPILIH:
        df_test[col] = 1.0
    X, y = select_features(df_test)
    assert list(y) == [1, 1, 0, 0]


def test_select_features_no_nan(X_y):
    X, y = X_y
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0


# ── Test split_data ───────────────────────────────────────────────────────────

def test_split_data_sizes(X_y):
    X, y = X_y
    X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.2)
    total = len(X_tr) + len(X_te)
    assert total == len(X)
    assert abs(len(X_te) / total - 0.2) < 0.01


def test_split_data_stratified(X_y):
    """Proporsi kelas train vs test harus selisih < 1%."""
    X, y = X_y
    _, _, y_tr, y_te = split_data(X, y, test_size=0.2)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01


def test_split_data_no_overlap(X_y):
    """Index train dan test tidak boleh tumpang tindih."""
    X, y = X_y
    X_tr, X_te, _, _ = split_data(X, y)
    assert len(set(X_tr.index) & set(X_te.index)) == 0


# ── Test validate_input ───────────────────────────────────────────────────────

VALID_INPUT = {
    "Monthly_Revenue": 50000.0,
    "Net_Profit_Margin": 10.0,
    "Burn_Rate_Ratio": 0.8,
    "Transaction_Count": 100,
    "Repeat_Order_Rate": 30.0,
    "Business_Tenure_Months": 24,
}


def test_validate_input_valid():
    ok, errs = validate_input(VALID_INPUT)
    assert ok is True
    assert errs == []


def test_validate_input_missing_field():
    data = {k: v for k, v in VALID_INPUT.items() if k != "Net_Profit_Margin"}
    ok, errs = validate_input(data)
    assert ok is False
    assert any("Net_Profit_Margin" in e for e in errs)


def test_validate_input_out_of_range_transaction():
    data = {**VALID_INPUT, "Transaction_Count": -5}
    ok, errs = validate_input(data)
    assert ok is False
    assert len(errs) == 1


def test_validate_input_burn_rate_too_high():
    data = {**VALID_INPUT, "Burn_Rate_Ratio": 3.0}
    ok, errs = validate_input(data)
    assert ok is False


def test_validate_input_non_numeric():
    data = {**VALID_INPUT, "Monthly_Revenue": "abc"}
    ok, errs = validate_input(data)
    assert ok is False


def test_validate_input_no_mutation():
    """validate_input tidak boleh mengubah dict input."""
    original = dict(VALID_INPUT)
    validate_input(VALID_INPUT)
    assert VALID_INPUT == original


# ── Test validate_csv_columns ─────────────────────────────────────────────────

def test_validate_csv_columns_valid():
    df_ok = pd.DataFrame(columns=FITUR_TERPILIH)
    ok, missing = validate_csv_columns(df_ok)
    assert ok is True
    assert missing == []


def test_validate_csv_columns_missing():
    df_bad = pd.DataFrame(columns=["Monthly_Revenue"])
    ok, missing = validate_csv_columns(df_bad)
    assert ok is False
    assert len(missing) == len(FITUR_TERPILIH) - 1
