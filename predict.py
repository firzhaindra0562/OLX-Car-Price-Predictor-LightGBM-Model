"""
predict.py — OLX Car Price Predictor
=====================================
Script untuk memuat model yang sudah disimpan dan melakukan prediksi harga.

Cara pakai:
    python predict.py
    python predict.py --input data_baru.csv --output hasil_prediksi.csv
"""

import argparse
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR    = Path("model_artifacts")
MODEL_FILE   = MODEL_DIR / "lgbm_car_price.joblib"
ENCODER_FILE = MODEL_DIR / "label_encoders.joblib"
META_FILE    = MODEL_DIR / "model_metadata.json"

LUXURY_BRANDS   = {"BMW","Mercedes-Benz","Porsche","Lexus","Audi","Jaguar","Volvo","Land Rover"}
JAPANESE_BRANDS = {"Toyota","Honda","Daihatsu","Suzuki","Mitsubishi","Nissan","Mazda","Subaru"}
CURRENT_YEAR    = 2026


# ══════════════════════════════════════════════════════════════════════════════
# ModelLoader — memuat semua artifact dari disk
# ══════════════════════════════════════════════════════════════════════════════
class CarPriceModel:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model     = joblib.load(model_dir / "lgbm_car_price.joblib")
        self.le_dict   = joblib.load(model_dir / "label_encoders.joblib")
        with open(model_dir / "model_metadata.json") as f:
            self.meta  = json.load(f)
        self.features  = self.meta["features"]
        self.cat_cols  = self.meta["cat_features"]
        print(f"✅ Model loaded: {self.meta['model_name']}")
        print(f"   Features    : {len(self.features)}")
        print(f"   Trained on  : {self.meta['training_rows']:,} baris")
        cv = self.meta["cv_metrics"].get("LightGBM (Tuned)", {})
        print(f"   CV R²       : {cv.get('R2','—')}  |  MAE: Rp {cv.get('MAE_juta','—')} juta")

    def _add_engineered(self, row: dict) -> dict:
        """Tambah fitur turunan dari raw input."""
        car_age               = max(CURRENT_YEAR - row["year"], 1)
        row["car_age"]        = car_age
        row["km_per_year"]    = row["mileage"] / car_age
        row["listing_quality"]= row.get("jumlah_foto", 0) * 0.7 + row.get("favorit", 0) * 0.3
        row["is_luxury"]      = int(row["merek"] in LUXURY_BRANDS)
        row["is_japanese"]    = int(row["merek"] in JAPANESE_BRANDS)
        return row

    def _encode_row(self, row: dict) -> dict:
        """Label-encode semua kolom kategori."""
        encoded = {}
        for feat in self.features:
            val = row.get(feat, 0)
            if feat in self.cat_cols:
                le  = self.le_dict[feat]
                val = le.transform([str(val)])[0] if str(val) in le.classes_ else 0
            encoded[feat] = val
        return encoded

    def predict_one(self, car: dict) -> float:
        """
        Prediksi harga 1 mobil. Kembalikan nilai dalam IDR (float).

        Parameter car (dict):
            year          : int   — tahun kendaraan, misal 2020
            mileage       : int   — kilometer dalam ribuan, misal 50 (= 50.000 km)
            merek         : str   — merek mobil, misal "Toyota"
            transmisi     : str   — "Automatic" / "Manual"
            bahan_bakar   : str   — "Bensin" / "Diesel" / "Hybrid" / "Electric"
            tipe_bodi     : str   — "SUV" / "Sedan" / "Hatchback" / dll
            kapasitas_cc  : str   — "1.000 - 1.500 cc" / ">2.000 - 3.000 cc" / dll
            tipe_penjual  : str   — "Individu" / "Diler"
            warna         : str   — "Hitam" / "Putih" / dll
            jumlah_foto   : int   — jumlah foto listing (opsional, default 5)
            favorit       : int   — jumlah favorit (opsional, default 0)
            has_video     : int   — 0 / 1
            has_promotion : int   — 0 / 1
            is_hot        : int   — 0 / 1
        """
        car    = {**car}  # copy
        car    = self._add_engineered(car)
        enc    = self._encode_row(car)
        X      = pd.DataFrame([enc])
        y_log  = self.model.predict(X)[0]
        return float(np.expm1(y_log))

    def predict_batch(self, df_input: pd.DataFrame) -> pd.Series:
        """
        Prediksi batch dari DataFrame.
        Kolom yang dibutuhkan sama seperti parameter predict_one().
        Kolom opsional yang tidak ada akan diisi 0 / Unknown.
        """
        results = []
        for _, row in df_input.iterrows():
            car = row.to_dict()
            # Isi default untuk kolom opsional
            car.setdefault("jumlah_foto", 5)
            car.setdefault("favorit", 0)
            car.setdefault("has_video", 0)
            car.setdefault("has_promotion", 0)
            car.setdefault("is_hot", 0)
            car.setdefault("tipe_bodi", "Unknown")
            car.setdefault("kapasitas_cc", "Unknown")
            car.setdefault("tipe_penjual", "Unknown")
            car.setdefault("warna", "Unknown")
            results.append(self.predict_one(car))
        return pd.Series(results, index=df_input.index, name="predicted_price")

    def valid_categories(self, column: str) -> list:
        """Tampilkan nilai kategori yang dikenali model untuk kolom tertentu."""
        if column in self.le_dict:
            return list(self.le_dict[column].classes_)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# CLI — Contoh penggunaan langsung
# ══════════════════════════════════════════════════════════════════════════════
def demo():
    """Contoh prediksi 3 mobil dengan profil berbeda."""
    model = CarPriceModel()
    print()

    test_cases = [
        {
            "desc"        : "Toyota Fortuner 2022, KM 15rb, SUV Automatic",
            "year"        : 2022, "mileage": 15,
            "merek"       : "Toyota", "transmisi": "Automatic",
            "bahan_bakar" : "Diesel", "tipe_bodi": "SUV",
            "kapasitas_cc": ">2.000 - 3.000 cc", "tipe_penjual": "Diler",
            "warna"       : "Hitam", "jumlah_foto": 10, "favorit": 5,
            "has_video": 0, "has_promotion": 0, "is_hot": 0,
        },
        {
            "desc"        : "Honda Brio 2019, KM 45rb, Hatchback Manual",
            "year"        : 2019, "mileage": 45,
            "merek"       : "Honda", "transmisi": "Manual",
            "bahan_bakar" : "Bensin", "tipe_bodi": "Hatchback",
            "kapasitas_cc": "1.000 - 1.500 cc", "tipe_penjual": "Individu",
            "warna"       : "Putih", "jumlah_foto": 6, "favorit": 2,
            "has_video": 0, "has_promotion": 0, "is_hot": 0,
        },
        {
            "desc"        : "BMW X5 2021, KM 20rb, SUV Automatic",
            "year"        : 2021, "mileage": 20,
            "merek"       : "BMW", "transmisi": "Automatic",
            "bahan_bakar" : "Bensin", "tipe_bodi": "SUV",
            "kapasitas_cc": ">2.000 - 3.000 cc", "tipe_penjual": "Diler",
            "warna"       : "Abu-abu", "jumlah_foto": 15, "favorit": 8,
            "has_video": 1, "has_promotion": 0, "is_hot": 0,
        },
    ]

    print("=" * 55)
    print("  DEMO PREDIKSI HARGA MOBIL BEKAS")
    print("=" * 55)
    for car in test_cases:
        desc  = car.pop("desc")
        price = model.predict_one(car)
        print(f"\n  {desc}")
        print(f"  → Prediksi: Rp {price/1e6:.1f} juta")
    print()


def predict_from_csv(input_path: str, output_path: str):
    """Baca CSV, prediksi semua baris, simpan hasilnya."""
    model    = CarPriceModel()
    df_input = pd.read_csv(input_path)
    print(f"\n  Input  : {input_path} ({len(df_input):,} baris)")
    df_input["predicted_price_idr"] = model.predict_batch(df_input)
    df_input["predicted_price_juta"] = (df_input["predicted_price_idr"] / 1e6).round(1)
    df_input.to_csv(output_path, index=False)
    print(f"  Output : {output_path}")
    print(f"\n  Preview prediksi:")
    print(df_input[["merek","year","mileage","predicted_price_juta"]].head(5).to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OLX Car Price Predictor")
    parser.add_argument("--input",  type=str, default=None, help="Path CSV input")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path CSV output")
    args = parser.parse_args()

    if args.input:
        predict_from_csv(args.input, args.output)
    else:
        demo()
