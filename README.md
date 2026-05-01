# 🚗 OLX Car Price Predictor — LightGBM Model

Model prediksi harga mobil bekas dari data listing OLX Indonesia,
dibangun menggunakan LightGBM dengan hyperparameter tuning via Optuna.

---

## 📁 Struktur File

```
📦 olx-car-price-model/
├── model_artifacts/
│   ├── lgbm_car_price.joblib     ← Model utama (LightGBM)
│   ├── label_encoders.joblib     ← Encoder untuk fitur kategorikal
│   └── model_metadata.json       ← Metadata, hyperparameter, CV metrics
├── predict.py                    ← Script inferensi siap pakai
├── olx_pipeline.py               ← Pipeline lengkap (cleaning → modeling)
├── olx_cleaned.csv               ← Dataset bersih hasil cleaning
├── requirements.txt              ← Dependensi Python
└── README.md                     ← Dokumentasi ini
```

---

## ⚙️ Instalasi

```bash
pip install -r requirements.txt
```

---

## 🚀 Cara Penggunaan

### 1. Demo cepat (3 contoh prediksi)

```bash
python predict.py
```

### 2. Prediksi dari CSV

```bash
python predict.py --input data_baru.csv --output hasil.csv
```

### 3. Pakai dalam kode Python

```python
from predict import CarPriceModel

model = CarPriceModel()  # load model_artifacts/

# Prediksi 1 mobil
harga = model.predict_one({
    "year"        : 2022,
    "mileage"     : 15,          # dalam ribuan km (15 = 15.000 km)
    "merek"       : "Toyota",
    "transmisi"   : "Automatic",
    "bahan_bakar" : "Bensin",
    "tipe_bodi"   : "SUV",
    "kapasitas_cc": ">2.000 - 3.000 cc",
    "tipe_penjual": "Individu",
    "warna"       : "Hitam",
    "jumlah_foto" : 8,
    "favorit"     : 3,
    "has_video"   : 0,
    "has_promotion": 0,
    "is_hot"      : 0,
})
print(f"Prediksi: Rp {harga/1e6:.1f} juta")

# Cek kategori yang valid
print(model.valid_categories("merek"))
print(model.valid_categories("tipe_bodi"))
```

### 4. Prediksi batch (DataFrame)

```python
import pandas as pd
from predict import CarPriceModel

model    = CarPriceModel()
df_input = pd.read_csv("data_baru.csv")

df_input["predicted_price"] = model.predict_batch(df_input)
print(df_input[["merek","year","mileage","predicted_price"]].head())
```

---

## 📊 Performa Model

| Metric       | Nilai                |
|--------------|----------------------|
| R² (5-Fold)  | **0.7827 ± 0.0123**  |
| MAE          | **Rp 61.4 juta**     |
| MAPE         | **22.2%**            |
| Training set | 1.432 listing        |
| Algoritma    | LightGBM + Optuna    |

---

## 🔧 Fitur yang Digunakan (19 fitur)

| Fitur             | Tipe         | Keterangan                           |
|-------------------|--------------|--------------------------------------|
| `year`            | Numerik      | Tahun kendaraan                      |
| `mileage`         | Numerik      | Kilometer dalam ribuan               |
| `car_age`         | Engineered   | 2026 − year                          |
| `km_per_year`     | Engineered   | mileage / car_age                    |
| `listing_quality` | Engineered   | jumlah_foto×0.7 + favorit×0.3        |
| `is_luxury`       | Engineered   | 1 jika BMW/Mercedes/Porsche/dll      |
| `is_japanese`     | Engineered   | 1 jika Toyota/Honda/Daihatsu/dll     |
| `merek`           | Kategorikal  | Merek kendaraan                      |
| `transmisi`       | Kategorikal  | Automatic / Manual                   |
| `bahan_bakar`     | Kategorikal  | Bensin / Diesel / Hybrid / Electric  |
| `tipe_bodi`       | Kategorikal  | SUV / Sedan / Hatchback / dll        |
| `kapasitas_cc`    | Kategorikal  | Kapasitas mesin                      |
| `tipe_penjual`    | Kategorikal  | Individu / Diler                     |
| `warna`           | Kategorikal  | Warna kendaraan                      |
| `jumlah_foto`     | Numerik      | Jumlah foto listing                  |
| `favorit`         | Numerik      | Jumlah pengguna yang memfavorit      |
| `has_video`       | Boolean      | Ada video listing (0/1)              |
| `has_promotion`   | Boolean      | Ada promosi (0/1)                    |
| `is_hot`          | Boolean      | Ditandai hot listing (0/1)           |

---

## 🧹 Data Cleaning yang Dilakukan

| Step                        | Baris Dihapus | Alasan                                      |
|-----------------------------|:---:|---------------------------------------------|
| Missing price/year/mileage  | 0   | —                                           |
| Duplicate listing ID        | 56  | Scraping ganda, simpan yang terbaru         |
| Semantic duplicate          | 20  | Judul + harga + tahun + km identik          |
| Price floor < Rp 15 juta   | 1   | Bukan listing mobil wajar                   |
| Brand-level IQR outlier     | 16  | Harga anomali per merek (IQR × 2.5)         |
| **Total**                   | **93** | **1.525 → 1.432 baris (−6.1%)**         |

---

## ⚠️ Catatan Penting

- **Target transform**: model dilatih pada `log1p(price)`, prediksi dikembalikan ke IDR asli dengan `expm1()`
- **Nilai kategori tidak dikenal**: jika nilai tidak ada di training data, akan di-encode sebagai `0` (Unknown)
- **Mileage**: input dalam **ribuan km** (50 = 50.000 km)
- **Cek nilai valid**: gunakan `model.valid_categories("merek")` untuk melihat merek yang dikenali model

---

## 📦 Hyperparameter Terbaik (Optuna, 60 trials)

Tersimpan di `model_artifacts/model_metadata.json`.

---

*Dataset: OLX mobil bekas Indonesia — April 2026*
