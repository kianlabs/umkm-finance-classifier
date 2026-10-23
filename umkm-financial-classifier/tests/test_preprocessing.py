"""
test_preprocessing.py
Unit tests untuk modul src/preprocessing.py (4 kelas, 5 fitur SBS)
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
    KELAS,
    load_dataset,
    select_features,
    split_data,
    validate_input,
    validate_csv_columns,
)

DATASET_PATH = os.path.join("data", "raw", "dataset_umkm.csv")


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


def test_fitur_terpilih_lima_sbs():
    """5 fitur SBS: tanpa Monthly_Revenue/Transaction_Count, dengan Avg_Historical_Rating."""
    assert len(FITUR_TERPILIH) == 5
    assert "Avg_Historical_Rating" in FITUR_TERPILIH
    assert "Monthly_Revenue" not in FITUR_TERPILIH
    assert "Transaction_Count" not in FITUR_TERPILIH


# ── Test select_features ──────────────────────────────────────────────────────

def test_select_features_columns(X_y):
    X, y = X_y
    assert list(X.columns) == FITUR_TERPILIH


def test_select_features_four_class_labels(X_y):
    X, y = X_y
    assert set(y.unique()) == {0, 1, 2, 3}, f"Label bukan 4 kelas: {y.unique()}"


def test_label_map_empat_kelas():
    df_test = pd.DataFrame({LABEL_KOLOM: ["Critical", "Elite", "Growth", "Struggling"]})
    for col in FITUR_TERPILIH:
        df_test[col] = 1.0
    X, y = select_features(df_test)
    assert list(y) == [LABEL_MAP["Critical"], LABEL_MAP["Elite"],
                       LABEL_MAP["Growth"], LABEL_MAP["Struggling"]]
    assert list(y) == [0, 1, 2, 3]


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
    assert len(X_te) == 30000


def test_split_data_stratified(X_y):
    """Proporsi tiap kelas train vs test harus selisih < 1%."""
    X, y = X_y
    _, _, y_tr, y_te = split_data(X, y, test_size=0.2)
    p_tr = y_tr.value_counts(normalize=True).sort_index()
    p_te = y_te.value_counts(normalize=True).sort_index()
    assert (abs(p_tr - p_te) < 0.01).all()


def test_split_data_no_overlap(X_y):
    X, y = X_y
    X_tr, X_te, _, _ = split_data(X, y)
    assert len(set(X_tr.index) & set(X_te.index)) == 0


# ── Test validate_input ───────────────────────────────────────────────────────

VALID_INPUT = {
    "Net_Profit_Margin": 10.0,
    "Burn_Rate_Ratio": 0.8,
    "Avg_Historical_Rating": 4.2,
    "Business_Tenure_Months": 24,
    "Repeat_Order_Rate": 30.0,
}


def test_validate_input_valid():
    ok, errs = validate_input(VALID_INPUT)
    assert ok is True
    assert errs == []


def test_validate_input_missing_field():
    data = {k: v for k, v in VALID_INPUT.items() if k != "Avg_Historical_Rating"}
    ok, errs = validate_input(data)
    assert ok is False
    assert any("Avg_Historical_Rating" in e for e in errs)


def test_validate_input_rating_out_of_range():
    data = {**VALID_INPUT, "Avg_Historical_Rating": 7.5}
    ok, errs = validate_input(data)
    assert ok is False
    assert len(errs) == 1


def test_validate_input_burn_rate_too_high():
    data = {**VALID_INPUT, "Burn_Rate_Ratio": 3.0}
    ok, errs = validate_input(data)
    assert ok is False


def test_validate_input_non_numeric():
    data = {**VALID_INPUT, "Burn_Rate_Ratio": "abc"}
    ok, errs = validate_input(data)
    assert ok is False


def test_validate_input_no_mutation():
    original = dict(VALID_INPUT)
    validate_input(VALID_INPUT)
    assert VALID_INPUT == original


# ── Test validate_csv_columns ─────────────────────────────────────────────────

def test_validate_csv_columns_valid():
    df_ok = pd.DataFrame(columns=FITUR_TERPILIH)
    ok, missing = validate_csv_columns(df_ok)
    assert ok is True
    assert missing == []


def test_validate_csv_columns_raw_names():
    """Nama mentah CSV (dengan satuan) juga diterima."""
    df_raw = pd.DataFrame(columns=[
        "Net_Profit_Margin (%)", "Burn_Rate_Ratio", "Avg_Historical_Rating",
        "Business_Tenure_Months", "Repeat_Order_Rate (%)",
    ])
    ok, missing = validate_csv_columns(df_raw)
    assert ok is True
    assert missing == []


def test_validate_csv_columns_missing():
    df_bad = pd.DataFrame(columns=["Net_Profit_Margin"])
    ok, missing = validate_csv_columns(df_bad)
    assert ok is False
    assert len(missing) == len(FITUR_TERPILIH) - 1
