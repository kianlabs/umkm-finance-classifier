"""
preprocessing.py
Memuat dataset, menyeleksi 5 fitur hasil Sequential Backward Selection (SBS),
memetakan label 4 kelas, stratified split, dan validasi input.

Sesuai naskah skripsi:
  - Judul: Penerapan Metode Naïve Bayes untuk Klasifikasi Kondisi Keuangan
    UMKM Berdasarkan Laporan Keuangan
  - Dataset: 150.000 baris sintetis (Kaggle: zkyfauzi/umkm-dataset),
    11 kandidat fitur numerik → 5 fitur terpilih via SBS.
  - Label asli: kolom "Class" dengan 4 kelas:
    Critical, Elite, Growth, Struggling (TIDAK dibinarisasi).

Kolom fitur mentah di CSV:
  "Net_Profit_Margin (%)", "Burn_Rate_Ratio", "Avg_Historical_Rating",
  "Business_Tenure_Months", "Repeat_Order_Rate (%)"
Fitur yang tereliminasi SBS (tidak dipakai model):
  Monthly_Revenue, Transaction_Count, Review_Volatility,
  Digital_Adoption_Score, Location_Competitiveness, Sentiment_Score.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Konstanta ────────────────────────────────────────────────────────────────

COL_RAW_LABEL = "Class"

# Kandidat fitur numerik awal (11) sebelum seleksi SBS — untuk dokumentasi/EDA
KANDIDAT_FITUR: list = [
    "Monthly_Revenue",
    "Net_Profit_Margin",
    "Burn_Rate_Ratio",
    "Transaction_Count",
    "Avg_Historical_Rating",
    "Review_Volatility",
    "Business_Tenure_Months",
    "Repeat_Order_Rate",
    "Digital_Adoption_Score",
    "Location_Competitiveness",
    "Sentiment_Score",
]

COL_RAW_MAP = {
    "Net_Profit_Margin":      "Net_Profit_Margin (%)",
    "Burn_Rate_Ratio":        "Burn_Rate_Ratio",
    "Avg_Historical_Rating":  "Avg_Historical_Rating",
    "Business_Tenure_Months": "Business_Tenure_Months",
    "Repeat_Order_Rate":      "Repeat_Order_Rate (%)",
    # Kandidat lain (untuk EDA/korelasi dataset mentah)
    "Monthly_Revenue":        "Monthly_Revenue",
    "Transaction_Count":      "Transaction_Count",
    "Review_Volatility":      "Review_Volatility",
    "Digital_Adoption_Score": "Digital_Adoption_Score",
    "Location_Competitiveness": "Location_Competitiveness",
    "Sentiment_Score":        "Sentiment_Score",
}

# 5 fitur terpilih hasil Sequential Backward Selection (urutan sesuai naskah)
FITUR_TERPILIH: list = [
    "Net_Profit_Margin",
    "Burn_Rate_Ratio",
    "Avg_Historical_Rating",
    "Business_Tenure_Months",
    "Repeat_Order_Rate",
]

LABEL_KOLOM: str = "Financial_Condition"

# 4 kelas — urutan alfabetis (sama dengan LabelEncoder sklearn)
KELAS: list = ["Critical", "Elite", "Growth", "Struggling"]

LABEL_MAP: dict = {nama: i for i, nama in enumerate(KELAS)}   # nama -> indeks
LABEL_TEKS: dict = {i: nama for nama, i in LABEL_MAP.items()}  # indeks -> nama

# Rentang validasi diturunkan dari nilai min/max aktual dataset 150.000 baris:
#   Net_Profit_Margin -35,0..45,0 | Burn_Rate_Ratio 0,437..1,55
#   Avg_Historical_Rating 1,5..5,0 (skala rating 1..5) | Tenure 3..179
#   Repeat_Order_Rate 2,0..54,06
BATAS_VALIDASI: dict = {
    "Net_Profit_Margin":      {"min": -35.0, "max": 45.0,  "tipe": float,
                               "label": "Net Profit Margin (%)"},
    "Burn_Rate_Ratio":        {"min": 0.43,  "max": 1.55,  "tipe": float,
                               "label": "Burn Rate Ratio"},
    "Avg_Historical_Rating":  {"min": 1.0,   "max": 5.0,   "tipe": float,
                               "label": "Rating Historis Rata-rata"},
    "Business_Tenure_Months": {"min": 3,     "max": 179,   "tipe": int,
                               "label": "Lama Usaha (bulan)"},
    "Repeat_Order_Rate":      {"min": 2.0,   "max": 54.06, "tipe": float,
                               "label": "Repeat Order Rate (%)"},
}


# ── Fungsi Utama ─────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """
    Muat dataset dari CSV.
    Rename kolom agar sesuai dengan nama canonical.
    Raise FileNotFoundError jika file tidak ada.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}\n"
            "Pastikan file CSV sudah ada di path tersebut."
        )
    df = pd.read_csv(path)

    rename_map = {v: k for k, v in COL_RAW_MAP.items()}
    df = df.rename(columns=rename_map)

    if COL_RAW_LABEL in df.columns and LABEL_KOLOM not in df.columns:
        df = df.rename(columns={COL_RAW_LABEL: LABEL_KOLOM})

    return df


def select_features(df: pd.DataFrame) -> tuple:
    """
    Pilih FITUR_TERPILIH (5 fitur SBS) sebagai X dan petakan LABEL_KOLOM
    ke indeks 4 kelas (Critical=0, Elite=1, Growth=2, Struggling=3) sebagai y.
    Return: (X: DataFrame, y: Series)
    """
    missing = [c for c in FITUR_TERPILIH if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom fitur tidak ditemukan: {missing}")
    if LABEL_KOLOM not in df.columns:
        raise ValueError(f"Kolom label '{LABEL_KOLOM}' tidak ditemukan.")

    X = df[FITUR_TERPILIH].copy()
    for col in FITUR_TERPILIH:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Label 4 kelas — TANPA binarisasi
    y = df[LABEL_KOLOM].map(LABEL_MAP)
    if y.isnull().sum() > 0:
        unknown = df[LABEL_KOLOM][y.isnull()].unique().tolist()
        raise ValueError(f"Nilai label tidak dikenal: {unknown}")
    y = y.astype(int)

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
    Stratified train-test split 80/20, random_state=42.
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
    Validasi dict input dari form/API terhadap BATAS_VALIDASI (5 fitur SBS).
    Return: (is_valid: bool, errors: list[str])
    """
    errors = []

    for fitur, aturan in BATAS_VALIDASI.items():
        if fitur not in data:
            errors.append(f"Field '{fitur}' ({aturan['label']}) tidak ditemukan.")
            continue

        nilai = data[fitur]

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

        batas_min = aturan["min"]
        batas_max = aturan["max"]
        if not (batas_min <= nilai_konversi <= batas_max):
            errors.append(
                f"'{fitur}' ({aturan['label']}) harus antara "
                f"{batas_min} s.d. {batas_max}, diterima: {nilai_konversi}."
            )

    return (len(errors) == 0, errors)


def validate_csv_columns(df: pd.DataFrame) -> tuple:
    """
    Pastikan df memiliki semua kolom FITUR_TERPILIH.
    Menerima nama canonical maupun nama mentah CSV (dengan satuan dalam kurung).
    Return: (is_valid: bool, missing_columns: list[str])
    """
    rename_map = {v: k for k, v in COL_RAW_MAP.items()}
    cols = {rename_map.get(c, c) for c in df.columns}
    missing = [c for c in FITUR_TERPILIH if c not in cols]
    return (len(missing) == 0, missing)
