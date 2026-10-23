# Sistem Klasifikasi Kondisi Keuangan UMKM

Aplikasi web berbasis Flask yang menggunakan **Gaussian Naïve Bayes** untuk
mengklasifikasikan kondisi keuangan UMKM ke dalam **4 kelas**:
**Critical, Struggling, Growth, Elite** — berdasarkan 5 fitur hasil
*Sequential Backward Selection* (SBS).

> Dokumentasi showcase lengkap ada di README root repo. File ini ringkasan teknis folder aplikasi.
>
> 🎓 **Project Skripsi** — aplikasi ini adalah program dari skripsi *"Penerapan Metode Naïve Bayes untuk Klasifikasi Kondisi Keuangan UMKM Berdasarkan Laporan Keuangan"* oleh Ridzkyan Buti Pratama, Teknik Informatika, Universitas Duta Bangsa Surakarta (2026).

## Persyaratan

- Python 3.10+
- pip

## Instalasi & Menjalankan

```bash
cd umkm-financial-classifier
pip install -r requirements.txt

# Dataset (150.000 baris, sintetis dari Kaggle: zkyfauzi/umkm-dataset)
# letakkan di: data/raw/dataset_umkm.csv

python train.py   # training pipeline (sekali)
python app.py     # → http://localhost:5000
```

## Struktur Direktori

```
umkm-financial-classifier/
├── app.py                    # Flask entry point
├── train.py                  # Script training (split 80/20 + CV 10-fold)
├── requirements.txt
├── data/raw/dataset_umkm.csv
├── models/
│   ├── gnb_pipeline.pkl      # Model terlatih (setelah train.py)
│   └── metrics.json          # Metrik evaluasi
├── src/
│   ├── preprocessing.py      # Load, 5 fitur SBS, label 4 kelas, split, validasi
│   ├── model.py              # Build, train, save, load, predict (4 kelas)
│   ├── evaluation.py         # Metrik macro/weighted, CM 4x4, ROC OvR, CV
│   └── visualization.py      # Plot multiclass → base64 PNG
├── static/css + js           # dashboard, klasifikasi, visualisasi, riwayat, tren
├── templates/                # base, dashboard, klasifikasi, visualisasi, riwayat
└── tests/                    # unit + property-based tests
```

## Fitur Aplikasi

| Halaman | Deskripsi |
|---|---|
| **Dashboard** | Ringkasan metrik model (Accuracy, Macro P/R/F1, ROC-AUC OvR, CV) |
| **Klasifikasi** | Form 5 fitur SBS + prediksi 4 kelas dengan probabilitas per kelas |
| **Visualisasi** | EDA, Confusion Matrix 4×4, ROC OvR, Feature Importance, CV scores |
| **Riwayat** | Log prediksi dengan filter per kelas, analisis tren tahunan |

## Fitur Input Model (hasil SBS dari 11 kandidat fitur)

| Fitur | Rentang Dataset | Satuan |
|---|---|---|
| Net Profit Margin | −35 s.d. 45 | % |
| Burn Rate Ratio | 0,437 s.d. 1,55 | Rasio |
| Avg Historical Rating | 1,5 s.d. 5,0 | Skala 1–5 |
| Business Tenure | 3 s.d. 179 | bulan |
| Repeat Order Rate | 2 s.d. 54,06 | % |

Fitur tereliminasi SBS (tidak dipakai model): Monthly Revenue,
Transaction Count, Review Volatility, Digital Adoption Score,
Location Competitiveness, Sentiment Score.

## Menjalankan Tests

```bash
pytest tests/ -v
```

## Metrik Model (test set 30.000 baris)

| Metrik | Nilai |
|---|---|
| Accuracy | 0,8888 |
| Macro Precision | 0,8352 |
| Macro Recall | 0,8738 |
| Macro F1 | 0,8521 |
| Weighted F1 | 0,8905 |
| Macro ROC-AUC (OvR) | 0,9791 |
| CV 10-Fold Accuracy | 88,84% ± 0,30% |
