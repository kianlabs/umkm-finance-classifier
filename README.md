# UMKM Financial Health Classifier

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)
![Tests](https://img.shields.io/badge/tests-50%20passed-brightgreen)
![CI](https://github.com/kianlabs/umkm-finance-classifier/actions/workflows/tests.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-blue)
![Skripsi](https://img.shields.io/badge/%F0%9F%8E%93%20Skripsi-Tugas%20Akhir%20S1-important)

Aplikasi web untuk mengklasifikasikan **kondisi keuangan UMKM ke dalam 4 kelas —
Critical, Struggling, Growth, dan Elite** — menggunakan **Gaussian Naïve Bayes**
dengan 5 fitur hasil seleksi *Sequential Backward Selection* (SBS).

Penjelasan di README ini disusun mengikuti urutan naskah skripsi: latar belakang dan
tujuan (Bab I), metode penelitian (Bab III), hasil dan pembahasan (Bab IV), lalu
implementasi prototipe web dan penutup (Bab V).

## 🎓 Project Skripsi

Repository ini berisi **program dari skripsi (tugas akhir) penulis**. Program
penelitian aslinya telah hilang dan yang tersisa hanya naskahnya, sehingga aplikasi
di repo ini adalah **hasil pembuatan ulang (rekonstruksi)** berdasarkan naskah
skripsi berikut:

| | |
|---|---|
| **Judul Skripsi** | Penerapan Metode Naïve Bayes untuk Klasifikasi Kondisi Keuangan UMKM Berdasarkan Laporan Keuangan |
| **Penulis** | Ridzkyan Buti Pratama |
| **Program Studi** | Teknik Informatika (S1) |
| **Universitas** | Universitas Duta Bangsa Surakarta |
| **Tahun** | 2026 |

## 1. Latar Belakang dan Tujuan Penelitian (Bab I)

UMKM merupakan tulang punggung perekonomian, tetapi kondisi keuangannya tidak
selalu mudah dinilai secara manual dari laporan keuangan. Penelitian ini menerapkan
metode **Naïve Bayes** untuk mengklasifikasikan kondisi keuangan UMKM ke dalam
empat kelas — **Critical, Struggling, Growth, dan Elite** — agar kondisi keuangan
sebuah usaha dapat diketahui secara cepat dan terukur dari indikator-indikator
keuangannya.

Tujuan penelitian:

- Membangun model klasifikasi kondisi keuangan UMKM menggunakan Gaussian Naïve Bayes.
- Menentukan fitur-fitur yang paling berpengaruh melalui seleksi fitur
  *Sequential Backward Selection* (SBS).
- Mengukur kinerja model dengan metrik evaluasi klasifikasi dan validasi silang.
- Mengimplementasikan model ke dalam prototipe sistem berbasis web.

## 2. Metode Penelitian (Bab III)

Penelitian dilakukan melalui tahapan berikut, sesuai urutan di Bab III naskah:

1. **Pengumpulan Data** — dataset UMKM sebanyak 150.000 baris (lihat seksi Dataset).
2. **Preprocessing Data** — pembersihan data dan penyiapan fitur numerik untuk pemodelan.
3. **Pembagian Data** — data dibagi menjadi data latih dan data uji dengan proporsi
   80:20 secara *stratified* (`random_state=42`), menghasilkan 30.000 data uji.
4. **Exploratory Data Analysis (EDA)** — eksplorasi distribusi kelas dan hubungan
   antar fitur sebelum pemodelan.
5. **Seleksi Fitur Menggunakan SBS** — *Sequential Backward Selection* mengeliminasi
   fitur satu per satu hingga tersisa fitur yang optimal (hasilnya di Bab IV).
6. **Pembangunan Model Gaussian Naïve Bayes** — model dilatih dari fitur terpilih
   tanpa penskalaan fitur.
7. **Evaluasi Model dan ROC-AUC** — kinerja diukur dengan accuracy, precision,
   recall, F1-score, confusion matrix, dan ROC-AUC *One-vs-Rest*.
8. **Validasi Silang 10-Fold** — kestabilan model diuji dengan *10-fold cross-validation*.
9. **Verifikasi Perhitungan** — hasil prediksi model diverifikasi dengan perhitungan
   manual probabilitas posterior pada satu sampel data.
10. **Implementasi Prototipe Sistem Berbasis Web** — model yang telah dievaluasi
    diterapkan ke aplikasi web Flask (lihat seksi 5).

## 3. Dataset

Dataset bersumber dari Kaggle: **[zkyfauzi/umkm-dataset](https://www.kaggle.com/datasets/zkyfauzi/umkm-dataset)**
dengan 150.000 baris data UMKM beserta label kondisi keuangannya.

Catatan penting: dataset ini bersifat **sintetis** (dibangkitkan secara artifisial,
bukan data UMKM riil). Konsekuensinya:

- Metrik tinggi (akurasi 88,9%, ROC-AUC 0,979) mencerminkan pola pada data sintetis;
  performa pada data UMKM riil kemungkinan lebih rendah dan perlu validasi lanjutan.
- File dataset (±20 MB) **disertakan di repo ini** di
  `umkm-financial-classifier/data/raw/dataset_umkm.csv` agar hasil penelitian
  dapat direproduksi persis; salinan aslinya juga tersedia di Kaggle pada tautan di atas.
  Format input klasifikasi mengikuti 5 fitur hasil SBS pada tabel di bawah.

## 4. Hasil dan Pembahasan (Bab IV)

### 4.1 Empat Kelas Kondisi Keuangan

| Kelas | Profil |
|---|---|
| 🚨 **Critical** | Tekanan keuangan serius: margin negatif, burn rate tinggi, rating & loyalitas rendah |
| ⚠️ **Struggling** | Keuangan perlu perhatian: margin tipis, pengeluaran mendekati/melebihi pendapatan |
| 📈 **Growth** | Sehat dan bertumbuh: margin positif, burn rate terkendali, pelanggan loyal |
| 👑 **Elite** | Sangat kuat: margin tinggi, rating pelanggan tinggi, repeat order tinggi |

### 4.2 Hasil Seleksi Fitur (SBS)

Dataset awal memiliki **11 kandidat fitur numerik**. SBS mengeliminasi fitur satu per satu
(yang kontribusinya paling kecil) hingga tersisa **5 fitur optimal**:

✅ **Terpilih:** Net Profit Margin (%), Burn Rate Ratio, Avg Historical Rating,
Business Tenure (bulan), Repeat Order Rate (%)

❌ **Tereliminasi:** Monthly Revenue, Transaction Count, Review Volatility,
Digital Adoption Score, Location Competitiveness, Sentiment Score

Temuan menariknya: besar omzet (*Monthly Revenue*) dan jumlah transaksi ternyata
**tidak** menentukan kondisi keuangan UMKM — efisiensi (margin & burn rate),
kepuasan pelanggan (rating), loyalitas (repeat order), dan pengalaman usaha (tenure) yang menentukan.

### 4.3 Hasil Evaluasi Model

Evaluasi pada test set 30.000 baris (split stratified 80/20, `random_state=42`).
Hasil evaluasi dari rekonstruksi ini sudah diverifikasi dan sama persis dengan
yang dilaporkan di naskah:

| Metrik | Nilai |
|---|---|
| Accuracy | **0,8888** |
| Macro Precision | 0,8352 |
| Macro Recall | 0,8738 |
| Macro F1-Score | 0,8521 |
| Weighted F1-Score | 0,8905 |
| Macro ROC-AUC (One-vs-Rest) | **0,9791** |
| CV 10-Fold Accuracy | 88,84% ± 0,30% |
| CV 10-Fold Macro F1 | 85,17% ± 0,31% |

Per kelas:

| Kelas | Precision | Recall | F1 | Support | ROC-AUC |
|---|---|---|---|---|---|
| Critical | 0,8752 | 0,8766 | 0,8759 | 2.512 | 0,9820 |
| Elite | 0,6699 | 0,8204 | 0,7375 | 2.038 | 0,9847 |
| Growth | 0,9471 | 0,8884 | 0,9168 | 17.136 | 0,9714 |
| Struggling | 0,8488 | 0,9099 | 0,8783 | 8.314 | 0,9784 |

Confusion matrix diagonal (prediksi benar): Critical 2.202 · Elite 1.672 · Growth 15.224 · Struggling 7.565.
Kelas Elite paling sulit dipisahkan dari Growth (precision terendah) — keduanya sama-sama profil keuangan positif.

### 4.4 Verifikasi Perhitungan

Verifikasi pada sampel perhitungan naskah (Net Profit Margin 4,09; Burn Rate Ratio
0,963; Rating 4,22; Tenure 85 bulan; Repeat Order Rate 19,15) menghasilkan kelas
**Growth** dengan probabilitas posterior **0,9976 (99,76%)** — sama dengan hasil
perhitungan yang dilaporkan di naskah.

## 5. Implementasi Prototipe Sistem Berbasis Web

Model diterapkan ke aplikasi web Flask dengan halaman-halaman sesuai rancangan
di naskah:

- **Beranda (Dashboard)** — metrik model (macro/weighted) & profil per kelas
- **Klasifikasi** — form 5 fitur SBS, hasil 4 kelas + probabilitas per kelas + rekomendasi per kelas
- **Riwayat** — log prediksi, filter per kelas, tren probabilitas positif per bulan
- **Visualisasi** — distribusi kelas, korelasi, KDE per kelas, confusion matrix 4×4, ROC One-vs-Rest, feature importance, skor CV

### Screenshot

| Dashboard | Klasifikasi |
|---|---|
| ![Dashboard](docs/screenshots/dashboard.png) | ![Klasifikasi](docs/screenshots/klasifikasi.png) |

| Visualisasi |
|---|
| ![Visualisasi](docs/screenshots/visualisasi.png) |

## 6. Penutup (Bab V)

Model Gaussian Naïve Bayes dengan 5 fitur hasil SBS mampu mengklasifikasikan
kondisi keuangan UMKM dengan akurasi 88,88% dan Macro ROC-AUC 0,9791 pada data uji.
Faktor yang menentukan kondisi keuangan UMKM bukanlah besar omzet, melainkan
efisiensi keuangan, kepuasan pelanggan, loyalitas, dan pengalaman usaha.
Penelitian lanjutan disarankan menggunakan data UMKM riil untuk memvalidasi model.

## Cara Menjalankan

```bash
# 1. Masuk folder aplikasi
cd umkm-financial-classifier

# 2. Install dependensi
pip install -r requirements.txt

# 3. Siapkan dataset di data/raw/dataset_umkm.csv
#    (unduh dari Kaggle: zkyfauzi/umkm-dataset — lihat catatan dataset di atas)

# 4. Training model (sekali)
python train.py

# 5. Jalankan aplikasi
python app.py
# → Buka http://localhost:5000
```

Menjalankan test:

```bash
cd umkm-financial-classifier
pytest tests/ -v     # 50 tests: unit + property-based (Hypothesis)
```

## Struktur Repo

```
.
├── README.md                  # ← Anda di sini
├── LICENSE
├── docs/screenshots/          # Screenshot aplikasi
└── umkm-financial-classifier/ # Aplikasi Flask
    ├── app.py                 # Entry point + REST API
    ├── train.py               # Training pipeline
    ├── requirements.txt
    ├── src/                   # preprocessing, model, evaluation, visualization
    ├── templates/ + static/   # UI: Dashboard, Klasifikasi, Visualisasi, Riwayat
    ├── models/                # gnb_pipeline.pkl + metrics.json (hasil training)
    ├── data/raw/              # dataset_umkm.csv (150.000 baris, disertakan di repo)
    └── tests/                 # Unit & property-based tests
```

## Penulis

**Ridzkyan Buti Pratama** — S.Kom, Teknik Informatika, Universitas Duta Bangsa Surakarta (2026)

- Portfolio: [kian.my.id](https://kian.my.id)
- LinkedIn: [linkedin.com/in/ridzkyanbutipratama](https://www.linkedin.com/in/ridzkyanbutipratama)
- GitHub: [@kianlabs](https://github.com/kianlabs)

## Lisensi

MIT — © 2026 Ridzkyan Buti Pratama
