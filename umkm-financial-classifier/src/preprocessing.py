"""
preprocessing.py
Memuat dataset, menyeleksi 6 fitur, binarisasi label,
stratified split, dan validasi input.

Dataset asli memiliki:
  - Kolom label: "Class" (Elite, Growth, Struggling, Critical)
  - Kolom fitur dengan nama: "Monthly_Revenue", "Net_Profit_Margin (%)",
    "Burn_Rate_Ratio", "Transaction_Count", "Business_Tenure_Months",
    "Repeat_Order_Rate (%)"
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Konstanta ────────────────────────────────────────────────────────────────

# Nama kolom di dataset CSV asli
COL_RAW_LABEL = "Class"
COL_RAW_MAP = {
    "Monthly_Revenue":        "Monthly_Revenue",
    "Net_Profit_Margin":      "Net_Profit_Margin (%)",
    "Burn_Rate_Ratio":        "Burn_Rate_Ratio",
    "Transaction_Count":      "Transaction_Count",
    "Repeat_Order_Rate":      "Repeat_Order_Rate (%)",
    "Business_Tenure_Months": "Business_Tenure_Months",
}

# Nama fitur canonical (dipakai di seluruh pipeline)
FITUR_TERPILIH: list = [
    "Monthly_Revenue",
    "Net_Profit_Margin",
    "Burn_Rate_Ratio",
    "Transaction_Count",
    "Repeat_Order_Rate",
    "Business_Tenure_Months",
]

LABEL_KOLOM: str = "Financial_Condition"

LABEL_MAP: dict = {
    "Elite":      1,
    "Growth":     1,
    "Struggling": 0,
    "Critical":   0,
}

BATAS_VALIDASI: dict = {
    "Monthly_Revenue":        {"min": 0.01,  "max": float("inf"), "tipe": float,
                               "label": "Pendapatan Bulanan"},
    "Net_Profit_Margin":      {"min": -35.0, "max": 45.0,         "tipe": float,
                               "label": "Net Profit Margin (%)"},
    "Burn_Rate_Ratio":        {"min": 0.44,  "max": 1.55,         "tipe": float,
                               "label": "Burn Rate Ratio"},
    "Transaction_Count":      {"min": 9,     "max": 285,          "tipe": int,
                               "label": "Jumlah Transaksi"},
    "Repeat_Order_Rate":      {"min": 2.0,   "max": 54.0,         "tipe": float,
                               "label": "Repeat Order Rate (%)"},
    "Business_Tenure_Months": {"min": 3,     "max": 179,          "tipe": int,
                               "label": "Lama Usaha (bulan)"},
}


# ── Fungsi Utama ─────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """
    Muat dataset dari CSV.
    Rename kolom agar sesuai dengan nama canonical FITUR_TERPILIH.
    Raise FileNotFoundError jika file tidak ada.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}\n"
            "Pastikan file CSV sudah ada di path tersebut."
        )
    df = pd.read_csv(path)

    # Rename kolom berspace/tanda kurung ke nama canonical
    rename_map = {v: k for k, v in COL_RAW_MAP.items()}
    df = df.rename(columns=rename_map)

    # Rename kolom label
    if COL_RAW_LABEL in df.columns and LABEL_KOLOM not in df.columns:
        df = df.rename(columns={COL_RAW_LABEL: LABEL_KOLOM})

    return df


def select_features(df: pd.DataFrame) -> tuple:
    """
    Pilih FITUR_TERPILIH sebagai X dan binarisasi LABEL_KOLOM sebagai y.
    Return: (X: DataFrame, y: Series)
    """
    missing = [c for c in FITUR_TERPILIH if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom fitur tidak ditemukan: {missing}")
    if LABEL_KOLOM not in df.columns:
        raise ValueError(f"Kolom label '{LABEL_KOLOM}' tidak ditemukan.")

    X = df[FITUR_TERPILIH].copy()
    # Pastikan tipe numerik
    for col in FITUR_TERPILIH:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Binarisasi label
    y = df[LABEL_KOLOM].map(LABEL_MAP)
    if y.isnull().sum() > 0:
        unknown = df[LABEL_KOLOM][y.isnull()].unique().tolist()
        raise ValueError(f"Nilai label tidak dikenal: {unknown}")

    # Hapus baris dengan NaN setelah konversi
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Stratified train-test split.
    WAJIB dipanggil SEBELUM normalisasi (mencegah data leakage).
    Return: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def validate_input(data: dict) -> tuple:
    """
    Validasi dict input dari form/API.
    Return: (is_valid: bool, errors: list[str])
    """
    errors = []

    for fitur, aturan in BATAS_VALIDASI.items():
        if fitur not in data:
            errors.append(f"Field '{fitur}' ({aturan['label']}) tidak ditemukan.")
            continue

        nilai = data[fitur]

        # Cek bisa dikonversi ke tipe yang benar
        try:
            if aturan["tipe"] == int:
                nilai_konversi = int(float(nilai))
            else:
                nilai_konversi = float(nilai)
        except (ValueError, TypeError):
            errors.append(
                f"'{fitur}' ({aturan['label']}) harus berupa angka, "
                f"diterima: '{nilai}'."
            )
            continue

        # Cek rentang
        batas_min = aturan["min"]
        batas_max = aturan["max"]
        if batas_max == float("inf"):
            if nilai_konversi < batas_min:
                errors.append(
                    f"'{fitur}' ({aturan['label']}) harus ≥ {batas_min}, "
                    f"diterima: {nilai_konversi}."
                )
        else:
            if not (batas_min <= nilai_konversi <= batas_max):
                errors.append(
                    f"'{fitur}' ({aturan['label']}) harus antara "
                    f"{batas_min} s.d. {batas_max}, diterima: {nilai_konversi}."
                )

    return (len(errors) == 0, errors)


def validate_csv_columns(df: pd.DataFrame) -> tuple:
    """
    Pastikan df memiliki semua kolom FITUR_TERPILIH.
    Return: (is_valid: bool, missing_columns: list[str])
    """
    missing = [c for c in FITUR_TERPILIH if c not in df.columns]
    return (len(missing) == 0, missing)
