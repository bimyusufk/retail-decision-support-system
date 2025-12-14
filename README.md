# 🛒 Sistem Pendukung Keputusan Retail (DSS)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-MLP%20Classifier-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

**Retail DSS** adalah dashboard analitik komprehensif yang dirancang untuk membantu manajer retail membuat keputusan berbasis data. Sistem ini menggabungkan **Market Basket Analysis (FP-Growth)**, **Segmentasi Pelanggan RFM**, **Analisis Product Affinity**, dan **Artificial Neural Networks (ANN)** untuk memprediksi probabilitas pembelian pelanggan dan mengekstrak insight marketing yang dapat ditindaklanjuti.

---

## 📋 Daftar Isi
- [Gambaran Umum](#-gambaran-umum)
- [Fitur Utama](#-fitur-utama)
- [Tech Stack](#-tech-stack)
- [Persiapan Dataset](#-persiapan-dataset)
- [Instalasi](#-instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Struktur Proyek](#-struktur-proyek)

---

## 🔭 Gambaran Umum

Aplikasi ini mengubah data transaksi retail mentah menjadi strategi marketing yang dapat ditindaklanjuti. Sistem ini menjawab pertanyaan bisnis kritis:

1. **Strategi Produk:** "Produk apa saja yang sering dibeli bersamaan?" (menggunakan Association Rules)
2. **Nilai Pelanggan:** "Siapa pelanggan paling bernilai saya?" (menggunakan Segmentasi RFM)
3. **Preferensi Demografis:** "Demografis mana yang lebih menyukai produk tertentu?" (menggunakan Analisis Product Affinity)
4. **Targeting Prediktif:** "Siapa yang paling mungkin membeli produk ini selanjutnya?" (menggunakan ANN)

Sistem ini dilengkapi modul **Business Insights** yang secara otomatis menghasilkan ringkasan eksekutif, segmen pelanggan dengan strategi yang dapat ditindaklanjuti, dan daftar target yang dapat diunduh untuk kampanye marketing.

---

## ✨ Fitur Utama

### 1. 📂 Manajemen Database
- Backend database SQLite untuk penyimpanan data yang efisien
- Import data otomatis dari file CSV
- Browser tabel dan antarmuka SQL query kustom
- Visualisasi Entity Relationship Diagram

### 2. 🔗 Market Basket Analysis (MBA)
- Implementasi **Algoritma FP-Growth** yang efisien (lebih cepat dari Apriori)
- Indikator progress untuk analisis yang memakan waktu lama
- Filter rules berdasarkan **Support**, **Confidence**, dan **Lift**
- Rekomendasi product bundling dan cross-selling

### 3. 👥 Segmentasi Pelanggan RFM
- Analisis nilai **Recency, Frequency, Monetary**
- 9 segmen pelanggan: Champions, Loyal, Potential Loyalist, New, Need Attention, About to Sleep, At Risk, Hibernating, Can't Lose Them
- Kartu strategi detail dengan insight dan rencana aksi per segmen
- Export pelanggan berbasis prioritas

### 4. 💝 Analisis Product Affinity
- Analisis preferensi produk berbasis demografis
- Perhitungan Affinity Index (deviasi dari rata-rata)
- Visualisasi heatmap hubungan produk-segmen
- Rekomendasi targeting strategis

### 5. 🧠 Pemodelan Prediktif (ANN)
- Classifier **Multi-Layer Perceptron (MLP)**
- **Pemilihan fitur demografis yang dapat dikonfigurasi** - pilih fitur mana yang akan dimasukkan dalam training
- Menangani data tidak seimbang menggunakan **SMOTE** atau Random Undersampling
- Analisis feature importance via permutation importance
- Skor **AUC-ROC** dan visualisasi Confusion Matrix

### 6. 📈 Hasil Prediksi & Buyer Persona
- Generasi buyer persona dengan profiling demografis
- Daftar pelanggan target dengan skor probabilitas
- Opsi export kampanye (Full List, Contact List, Summary Report)
- Rekomendasi targeting spesifik platform (Facebook, Google Ads, WhatsApp)

### 7. 💡 Business Insights Komprehensif
- **AI Smart Conclusion:** Ringkasan eksekutif yang di-generate otomatis
- **Business Health Score:** Indikator performa keseluruhan
- **Strategic Action Plan:** Matriks prioritas dengan timeline
- **Laporan yang Dapat Diunduh:** Export CSV untuk semua analisis (label kompatibel Excel)

---

## 🛠 Tech Stack

| Kategori | Teknologi |
|----------|-----------|
| **Frontend** | [Streamlit](https://streamlit.io/), Plotly |
| **Database** | SQLite |
| **Pemrosesan Data** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (MLPClassifier), Imbalanced-learn (SMOTE) |
| **Pattern Mining** | MLxtend (FP-Growth, Association Rules) |
| **Visualisasi** | Plotly Express, Matplotlib |

---

## 📥 Persiapan Dataset

Dataset **tidak disertakan** dalam repository ini karena keterbatasan ukuran file. Ikuti langkah-langkah berikut untuk mengunduh dan menyiapkan dataset:

### Langkah 1: Unduh dari Google Drive

Unduh file dataset dari Google Drive:

🔗 **[Unduh Dataset](https://drive.google.com/drive/folders/YOUR_FOLDER_ID_HERE)**

<!-- Ganti YOUR_FOLDER_ID_HERE dengan link Google Drive folder Anda yang sebenarnya -->

### Langkah 2: Tempatkan File di Folder `datasets/`

Setelah mengunduh, tempatkan `transaction_data.csv`, `hh_demographics.csv`, dan `product.csv`  di folder `datasets/`:

```
retail-decision-support-system/
├── datasets/
│   ├── transaction_data.csv    # Data transaksi (~145 MB)
│   ├── hh_demographics.csv     # Demografis pelanggan
│   └── product.csv             # Katalog produk
├── app.py
├── ...
```

### Langkah 3: Buat Database

1. Jalankan aplikasi: `streamlit run app.py`
2. Navigasi ke menu **Database**
3. Klik **"🚀 Buat Database dari CSV"** untuk membuat database SQLite
4. Tunggu proses import selesai (~1-2 menit tergantung spesifikasi hardware)

---

## ⚙️ Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/bimyusufk/retail-decision-support-system.git
cd retail-decision-support-system
```

### 2. Buat Virtual Environment (Disarankan)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Siapkan Dataset

Ikuti instruksi [Persiapan Dataset](#-persiapan-dataset) di atas.

### 5. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser Anda di `http://localhost:8501`

---

## 🚀 Cara Penggunaan

### 1. Setup Database
- Navigasi ke menu **Database**
- Buat database dari file CSV atau verifikasi database yang sudah ada
- Jelajahi tabel dan jalankan SQL query kustom

### 2. Analisis Association Rules
- Pergi ke menu **Association Rules**
- Konfigurasi pengelompokan basket (berdasarkan BASKET_ID atau DAY)
- Atur threshold minimum support dan confidence
- Klik **"Run Analysis"** untuk menemukan pola produk
- Export rules sebagai CSV untuk analisis lebih lanjut

### 3. Segmentasi Pelanggan RFM
- Pergi ke menu **RFM Analysis**
- Klik **"Hitung RFM"** untuk menghitung skor pelanggan
- Tinjau distribusi segmen dan detail pelanggan
- Unduh daftar pelanggan prioritas

### 4. Analisis Product Affinity
- Pergi ke menu **Product Affinity**
- Pilih dimensi demografis (Usia, Pendapatan, dll.)
- Jelajahi skor affinity dan perbandingan segmen
- Identifikasi peluang targeting

### 5. Training Model ANN
- Pergi ke menu **ANN Training**
- Pilih produk target dari pola populer atau masukkan secara manual
- **Pilih fitur demografis** yang akan dimasukkan dalam model
- Pilih metode resampling (SMOTE disarankan)
- Klik **"Mulai Training Model"**

### 6. Hasil Prediksi
- Pergi ke menu **Prediction Results**
- Jelajahi Buyer Persona, Feature Importance, dan Target Lists
- Filter berdasarkan threshold probabilitas
- Export target kampanye dan daftar kontak

### 7. Business Insights
- Pergi ke menu **Business Insights**
- Tinjau Executive Summary dan kesimpulan AI
- Cek Business Health Score
- Jelajahi insight detail per tipe analisis
- Unduh laporan komprehensif

---

## 📂 Struktur Proyek

```text
retail-decision-support-system/
├── .streamlit/
│   └── config.toml          # Konfigurasi tema Streamlit
├── datasets/
│   ├── transaction_data.csv # Data transaksi (unduh terpisah)
│   ├── hh_demographics.csv  # Demografis pelanggan (unduh terpisah)
│   ├── product.csv          # Katalog produk (unduh terpisah)
│   └── retail.db            # Database SQLite (auto-generated)
├── app.py                   # Aplikasi Streamlit utama
├── preprocessing.py         # Preprocessing data & analisis FP-Growth
├── model_utils.py           # Utilitas training & evaluasi ANN
├── database.py              # Operasi database SQLite & query
├── requirements.txt         # Dependencies Python
├── .gitignore              # Aturan git ignore
└── README.md               # Dokumentasi proyek
```

---

## 📊 Contoh Output

### Kartu Strategi Segmen RFM
Setiap segmen pelanggan mencakup:
- **Insight:** Mengapa segmen ini berperilaku demikian
- **Profil RFM:** Nilai rata-rata Recency, Frequency, Monetary
- **Strategi:** Pendekatan marketing yang direkomendasikan
- **Aksi:** Langkah selanjutnya yang spesifik

### Pemilihan Fitur ANN
Kustomisasi fitur demografis mana yang akan dimasukkan dalam training model:
- AGE_DESC (Usia)
- MARITAL_STATUS_CODE (Status Pernikahan)
- INCOME_DESC (Pendapatan)
- HOMEOWNER_DESC (Status Kepemilikan Rumah)
- HH_COMP_DESC (Komposisi Rumah Tangga)
- HOUSEHOLD_SIZE_DESC (Ukuran Rumah Tangga)
- KID_CATEGORY_DESC (Kategori Anak)

---


## 🙏 Ucapan Terima Kasih

- Sumber dataset: Dunnhumby - The Complete Journey
- Implementasi FP-Growth: Library MLxtend
- Framework UI: Streamlit
