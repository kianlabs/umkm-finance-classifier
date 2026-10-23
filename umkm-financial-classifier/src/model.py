"""
model.py
Membangun, melatih, menyimpan, memuat, dan menjalankan inferensi
Gaussian Naïve Bayes untuk klasifikasi 4 kelas kondisi keuangan UMKM
(Critical, Elite, Growth, Struggling) dengan 5 fitur hasil SBS.

Catatan konfigurasi (terverifikasi eksperimen, lihat hidden_notes/eksperimen.md):
Pipeline TIDAK memakai scaler. GaussianNB mengestimasi mean & varians
per fitur per kelas, sehingga penskalaan per fitur tidak diperlukan;
konfigurasi tanpa scaler mereproduksi metrik naskah secara eksak
(Accuracy 0,888767), sedangkan MinMaxScaler menggesernya (0,888800).
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

from src.preprocessing import FITUR_TERPILIH, KELAS, LABEL_TEKS


# ── Rekomendasi Kontekstual per Kelas ────────────────────────────────────────

_REKOMENDASI = {
    "Elite": (
        "UMKM Anda berada di kelas Elite — kondisi keuangan sangat kuat dengan "
        "margin, rating pelanggan, dan loyalitas yang tinggi. Pertahankan strategi "
        "saat ini, eksplorasi ekspansi pasar, dan pertimbangkan reinvestasi "
        "keuntungan untuk pertumbuhan jangka panjang."
    ),
    "Growth": (
        "UMKM Anda berada di kelas Growth — kondisi keuangan sehat dan bertumbuh. "
        "Pantau rasio burn rate, pertahankan repeat order rate, dan tingkatkan "
        "rating pelanggan agar dapat naik ke kelas Elite."
    ),
    "Struggling": (
        "UMKM Anda berada di kelas Struggling — kondisi keuangan perlu perhatian. "
        "Fokuskan pada peningkatan net profit margin, tekan burn rate ratio di "
        "bawah 1, dan tingkatkan repeat order melalui promosi serta layanan "
        "pelanggan agar rating historis membaik."
    ),
    "Critical": (
        "UMKM Anda berada di kelas Critical — tekanan keuangan serius. Segera "
        "lakukan efisiensi biaya operasional, evaluasi burn rate ratio, perbaiki "
        "margin keuntungan, dan cari sumber pendanaan tambahan. Konsultasikan "
        "dengan konsultan keuangan UMKM."
    ),
}


def generate_rekomendasi(label, probas: np.ndarray = None) -> str:
    """
    Buat teks rekomendasi berdasarkan kelas prediksi.
    label: indeks kelas (int) atau nama kelas (str).
    probas: array probabilitas 4 kelas (opsional, untuk nuansa keyakinan).
    """
    if isinstance(label, (int, np.integer)):
        nama = LABEL_TEKS[int(label)]
    else:
        nama = str(label)
    teks = _REKOMENDASI.get(nama, "")
    if probas is not None:
        p = float(np.max(probas))
        if p < 0.55:
            teks += (
                " Catatan: keyakinan model pada prediksi ini moderat "
                f"({p * 100:.1f}%), lakukan evaluasi berkala terhadap arus kas."
            )
    return teks


# ── Pipeline ML ──────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Buat sklearn Pipeline berisi GaussianNB saja (tanpa scaler —
    lihat catatan konfigurasi di docstring modul).
    """
    return Pipeline([
        ("gnb", GaussianNB()),
    ])


def train_pipeline(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit pipeline pada data latih. Return: pipeline yang sudah di-fit."""
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


def _proba_dict(pipeline: Pipeline, probas: np.ndarray) -> dict:
    """Petakan array probabilitas ke dict {nama_kelas: probabilitas}."""
    classes = list(pipeline.classes_)
    out = {}
    for nama in KELAS:
        # posisi nama kelas di classes_ pipeline
        target = [k for k, v in LABEL_TEKS.items() if v == nama][0]
        pos = classes.index(target) if target in classes else KELAS.index(nama)
        out[nama] = round(float(probas[pos]), 4)
    return out


def predict_single(pipeline: Pipeline, input_dict: dict) -> dict:
    """
    Prediksi satu sampel dari dict input (5 fitur SBS).
    Return: {label, label_teks, probabilitas {kelas: p}, prob_max, rekomendasi}
    """
    X = pd.DataFrame(
        [[float(input_dict[f]) for f in FITUR_TERPILIH]],
        columns=FITUR_TERPILIH,
    )

    label = int(pipeline.predict(X)[0])
    probas = pipeline.predict_proba(X)[0]
    prob_dict = _proba_dict(pipeline, probas)

    return {
        "label": label,
        "label_teks": LABEL_TEKS[label],
        "probabilitas": prob_dict,
        "prob_max": round(float(np.max(probas)), 4),
        "rekomendasi": generate_rekomendasi(label, probas),
    }


def predict_dataframe(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """
    Prediksi seluruh baris DataFrame.
    Menambahkan kolom: Prediksi_Label, Prediksi_Teks,
    Prob_Critical, Prob_Elite, Prob_Growth, Prob_Struggling, Prob_Max.
    """
    result = df.copy()
    X = df[FITUR_TERPILIH]

    labels = pipeline.predict(X)
    probas = pipeline.predict_proba(X)
    classes = list(pipeline.classes_)

    result["Prediksi_Label"] = [int(l) for l in labels]
    result["Prediksi_Teks"] = [LABEL_TEKS[int(l)] for l in labels]
    for nama in KELAS:
        target = [k for k, v in LABEL_TEKS.items() if v == nama][0]
        pos = classes.index(target) if target in classes else KELAS.index(nama)
        result[f"Prob_{nama}"] = [round(float(p[pos]), 4) for p in probas]
    result["Prob_Max"] = [round(float(np.max(p)), 4) for p in probas]

    return result
