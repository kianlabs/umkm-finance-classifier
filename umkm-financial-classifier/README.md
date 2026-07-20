# Sistem Klasifikasi Kondisi Keuangan UMKM

Aplikasi web berbasis Flask yang menggunakan **Gaussian Naïve Bayes** untuk mengklasifikasikan kondisi keuangan UMKM menjadi **Sehat** atau **Tidak Sehat**.

## Persyaratan

- Python 3.10+
- pip

## Instalasi & Menjalankan

```bash
# 1. Masuk ke direktori proyek
cd umkm-financial-classifier

# 2. Install dependensi
pip install -r requirements.txt

# 3. Salin dataset
# Pastikan file dataset ada di:
#   data/raw/dataset_umkm.csv
# (Salin dari synthetic_umkm_data.csv di root skripsi)

# 4. Training pipeline (jalankan sekali)
python train.py

# 5. Jalankan aplikasi web
python app.py
# → Buka http://localhost:5000
```

## Struktur Direktori

```
umkm-financial-classifier/
├── app.py                    # Flask entry point
├── train.py                  # Script training
├── requirements.txt
├── data/
│   └── raw/
│       └── dataset_umkm.csv  # Dataset 150.000 baris
├── models/
│   ├── gnb_pipeline.pkl      # Model terlatih (dibuat setelah train.py)
│   └── metrics.json          # Metrik evaluasi
├── src/
│   ├── preprocessing.py      # Load, fitur, split, validasi
│   ├── model.py              # Build, train, save, load, predict
│   ├── evaluation.py         # Metrics, confusion matrix, ROC, CV
│   └── visualization.py      # Plot → base64 PNG
├── static/
│   ├── css/style.css
│   └── js/
│       ├── dashboard.js
│       ├── klasifikasi.js
│       ├── visualisasi.js
│       └── riwayat.js
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── klasifikasi.html
│   ├── visualisasi.html
│   └── riwayat.html
└── tests/
    ├── test_preprocessing.py
    ├── test_model.py
    ├── test_evaluation.py
    └── test_pbt.py
```

## Fitur Aplikasi

| Halaman | Deskripsi |
|---|---|
| **Dashboard** | Ringkasan metrik model (Accuracy, F1, ROC-AUC, CV) |
| **Klasifikasi** | Form input 6 fitur keuangan + prediksi real-time dengan gauge probabilitas |
| **Visualisasi** | EDA, Confusion Matrix, ROC Curve, Feature Importance, CV scores |
| **Riwayat** | Log prediksi tersimpan dengan filter dan export CSV |

## Fitur Input Model

| Fitur | Rentang | Satuan |
|---|---|---|
| Monthly Revenue | > 0 | Rupiah |
| Net Profit Margin | -35% s.d. 45% | % |
| Burn Rate Ratio | 0.44 s.d. 1.55 | Rasio |
| Transaction Count | 9 s.d. 285 | transaksi/bulan |
| Repeat Order Rate | 2% s.d. 54% | % |
| Business Tenure | 3 s.d. 179 | bulan |

## Menjalankan Tests

```bash
# Unit tests
pytest tests/ -v

# Property-based tests (butuh model sudah terlatih)
pytest tests/test_pbt.py -v
```

## Metrik Target

| Metrik | Target | Aktual |
|---|---|---|
| Accuracy | ≥ 94% | 94.70% |
| F1-Score | ≥ 94% | 94.72% |
| ROC-AUC | ≥ 98% | 98.59% |
| CV Accuracy | ≥ 94% | 95.02% ± 0.18% |
