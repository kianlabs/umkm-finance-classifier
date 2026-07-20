"""
model.py
Membangun, melatih, menyimpan, memuat, dan menjalankan inferensi
Gaussian Naïve Bayes Pipeline (MinMaxScaler + GaussianNB).
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB

from src.preprocessing import FITUR_TERPILIH, BATAS_VALIDASI


# ── Rekomendasi Kontekstual ──────────────────────────────────────────────────

def generate_rekomendasi(label: int, probas: np.ndarray) -> str:
    """
    Buat teks rekomendasi berdasarkan label prediksi dan probabilitas.
    label: 0 = Tidak Sehat, 1 = Sehat
    probas: array [P(Tidak Sehat), P(Sehat)]
    """
    prob_sehat = probas[1]

    if label == 1:
        if prob_sehat >= 0.90:
            return (
                "UMKM Anda berada dalam kondisi keuangan yang sangat sehat. "
                "Pertahankan strategi saat ini, eksplorasi peluang ekspansi pasar, "
                "dan pertimbangkan reinvestasi keuntungan untuk pertumbuhan jangka panjang."
            )
        elif prob_sehat >= 0.70:
            return (
                "UMKM Anda dalam kondisi keuangan yang sehat. "
                "Pantau terus rasio burn rate dan tingkatkan repeat order rate "
                "untuk memperkuat posisi keuangan."
            )
        else:
            return (
                "UMKM Anda terklasifikasi sehat namun dengan probabilitas moderat. "
                "Perhatikan efisiensi operasional dan pertahankan margin profit. "
                "Lakukan evaluasi bulanan terhadap arus kas."
            )
    else:
        prob_tidak_sehat = probas[0]
        if prob_tidak_sehat >= 0.90:
            return (
                "UMKM Anda menghadapi tekanan keuangan yang serius. "
                "Segera lakukan efisiensi biaya operasional, evaluasi burn rate, "
                "dan cari sumber pendanaan tambahan. Konsultasikan dengan konsultan keuangan UMKM."
            )
        elif prob_tidak_sehat >= 0.70:
            return (
                "Kondisi keuangan UMKM Anda perlu perhatian. "
                "Fokuskan pada peningkatan net profit margin, kurangi pengeluaran tidak produktif, "
                "dan tingkatkan jumlah transaksi melalui strategi promosi digital."
            )
        else:
            return (
                "UMKM Anda berada di zona keuangan yang perlu diperbaiki. "
                "Evaluasi struktur biaya, tingkatkan repeat order rate, "
                "dan pertimbangkan diversifikasi produk untuk meningkatkan pendapatan."
            )


# ── Pipeline ML ──────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Buat sklearn Pipeline:
      Step 1: MinMaxScaler() — normalisasi ke [0, 1]
      Step 2: GaussianNB()   — classifier
    Pipeline menjamin scaler hanya di-fit sekali pada data latih.
    """
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("gnb",    GaussianNB()),
    ])


def train_pipeline(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Fit pipeline pada data latih.
    MinMaxScaler.fit() hanya menggunakan X_train.
    Return: pipeline yang sudah di-fit.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def save_pipeline(pipeline: Pipeline, path: str = "models/gnb_pipeline.pkl") -> None:
    """Simpan Pipeline ke .pkl menggunakan joblib.dump()."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: str = "models/gnb_pipeline.pkl") -> Pipeline:
    """
    Muat Pipeline dari .pkl menggunakan joblib.load().
    Raise FileNotFoundError jika file tidak ada.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {path}\n"
            "Jalankan `python train.py` untuk melatih model terlebih dahulu."
        )
    return joblib.load(path)


def predict_single(pipeline: Pipeline, input_dict: dict) -> dict:
    """
    Prediksi satu sampel dari dict input.
    Return: {label, label_teks, prob_sehat, prob_tidak_sehat, rekomendasi}
    """
    # Susun array input dengan urutan kolom yang benar
    X = np.array([[
        float(input_dict["Monthly_Revenue"]),
        float(input_dict["Net_Profit_Margin"]),
        float(input_dict["Burn_Rate_Ratio"]),
        int(float(input_dict["Transaction_Count"])),
        float(input_dict["Repeat_Order_Rate"]),
        int(float(input_dict["Business_Tenure_Months"])),
    ]])

    label  = int(pipeline.predict(X)[0])
    probas = pipeline.predict_proba(X)[0]  # [P(0), P(1)]

    return {
        "label":            label,
        "label_teks":       "Sehat" if label == 1 else "Tidak Sehat",
        "prob_sehat":       round(float(probas[1]), 4),
        "prob_tidak_sehat": round(float(probas[0]), 4),
        "rekomendasi":      generate_rekomendasi(label, probas),
    }


def predict_dataframe(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """
    Prediksi seluruh baris DataFrame.
    Menambahkan kolom: Prediksi_Label, Prediksi_Teks,
    Prob_Sehat, Prob_Tidak_Sehat.
    """
    result = df.copy()
    X = df[FITUR_TERPILIH].values.astype(float)

    labels = pipeline.predict(X)
    probas = pipeline.predict_proba(X)  # shape (n, 2)

    result["Prediksi_Label"]    = labels
    result["Prediksi_Teks"]     = ["Sehat" if l == 1 else "Tidak Sehat" for l in labels]
    result["Prob_Sehat"]        = [round(float(p[1]), 4) for p in probas]
    result["Prob_Tidak_Sehat"]  = [round(float(p[0]), 4) for p in probas]

    return result
