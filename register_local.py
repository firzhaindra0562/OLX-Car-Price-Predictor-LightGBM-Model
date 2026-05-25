"""
register_local.py
=================
Jalankan script ini SEKALI di komputermu untuk mendaftarkan model
ke MLflow registry lokal.

Struktur folder yang dibutuhkan:
    olx-project/
    ├── register_local.py       ← script ini
    ├── model_artifacts/
    │   ├── lgbm_car_price.joblib
    │   ├── label_encoders.joblib
    │   └── model_metadata.json

Cara pakai:
    python register_local.py
"""

import json
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Cek dependensi ────────────────────────────────────────────────────────
try:
    import mlflow
    import mlflow.pyfunc
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import Schema, ColSpec
except ImportError:
    print("❌ MLflow belum terinstall. Jalankan:")
    print("   pip install mlflow")
    sys.exit(1)

# ── Cek artifacts ada ─────────────────────────────────────────────────────
ARTIFACT_DIR = Path("model_artifacts")
required = ["lgbm_car_price.joblib", "label_encoders.joblib", "model_metadata.json"]
missing  = [f for f in required if not (ARTIFACT_DIR / f).exists()]
if missing:
    print(f"❌ File tidak ditemukan di {ARTIFACT_DIR}/:")
    for f in missing: print(f"   - {f}")
    print("\nPastikan folder model_artifacts/ ada di direktori yang sama.")
    sys.exit(1)

# ── Konstanta ─────────────────────────────────────────────────────────────
MLFLOW_URI      = "sqlite:///mlflow.db"
MODEL_NAME      = "OLXCarPrice"
CURRENT_YEAR    = 2026
LUXURY_BRANDS   = {"BMW","Mercedes-Benz","Porsche","Lexus","Audi","Jaguar","Volvo","Land Rover"}
JAPANESE_BRANDS = {"Toyota","Honda","Daihatsu","Suzuki","Mitsubishi","Nissan","Mazda","Subaru"}

FEATURES = [
    "year","mileage","merek","transmisi","bahan_bakar","tipe_bodi",
    "kapasitas_cc","tipe_penjual","warna","jumlah_foto","favorit",
    "has_video","has_promotion","is_hot",
    "car_age","km_per_year","listing_quality","is_luxury","is_japanese",
]
CAT_COLS = ["merek","transmisi","bahan_bakar","tipe_bodi",
            "kapasitas_cc","tipe_penjual","warna"]


# ══════════════════════════════════════════════════════════════════════════════
# Custom PythonModel — preprocessing + LightGBM dalam satu unit
# ══════════════════════════════════════════════════════════════════════════════
class OLXCarPriceModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model   = joblib.load(context.artifacts["lgbm_model"])
        self.le_dict = joblib.load(context.artifacts["label_encoders"])
        with open(context.artifacts["metadata"]) as f:
            self.meta = json.load(f)

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["car_age"]         = (CURRENT_YEAR - df["year"]).clip(lower=1)
        df["km_per_year"]     = df["mileage"] / df["car_age"]
        df["listing_quality"] = df.get("jumlah_foto", 5) * 0.7 + df.get("favorit", 0) * 0.3
        df["is_luxury"]       = df["merek"].isin(LUXURY_BRANDS).astype(int)
        df["is_japanese"]     = df["merek"].isin(JAPANESE_BRANDS).astype(int)
        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in CAT_COLS:
            if col not in df.columns:
                df[col] = "Unknown"
            le     = self.le_dict[col]
            df[col] = df[col].astype(str).apply(
                lambda v: le.transform([v])[0] if v in le.classes_ else 0
            )
        return df

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        defaults = {
            "jumlah_foto": 5, "favorit": 0,
            "has_video": 0, "has_promotion": 0, "is_hot": 0,
            "tipe_bodi": "Unknown", "kapasitas_cc": "Unknown",
            "tipe_penjual": "Unknown", "warna": "Unknown",
        }
        for col, val in defaults.items():
            if col not in model_input.columns:
                model_input[col] = val

        df   = self._encode(self._engineer(model_input))
        pred = np.expm1(self.model.predict(df[FEATURES].fillna(0)))
        return pd.DataFrame({
            "predicted_price_idr":  pred.round(0).astype(int),
            "predicted_price_juta": (pred / 1e6).round(1),
        })


# ══════════════════════════════════════════════════════════════════════════════
# Main — registrasi ke MLflow lokal
# ══════════════════════════════════════════════════════════════════════════════
def main():
    with open(ARTIFACT_DIR / "model_metadata.json") as f:
        meta = json.load(f)
    cv = meta["cv_metrics"]["LightGBM (Tuned)"]

    print("━" * 50)
    print("  Mendaftarkan Model ke MLflow Registry Lokal")
    print("━" * 50)
    print(f"  Tracking URI : {MLFLOW_URI}")
    print(f"  Model name   : {MODEL_NAME}\n")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("OLX-Car-Price-Prediction")

    # Input/output schema
    INPUT_SCHEMA = Schema([
        ColSpec("long",   "year"),
        ColSpec("long",   "mileage"),
        ColSpec("string", "merek"),
        ColSpec("string", "transmisi"),
        ColSpec("string", "bahan_bakar"),
        ColSpec("string", "tipe_bodi"),
        ColSpec("string", "kapasitas_cc"),
        ColSpec("string", "tipe_penjual"),
        ColSpec("string", "warna"),
        ColSpec("long",   "jumlah_foto"),
        ColSpec("long",   "favorit"),
        ColSpec("long",   "has_video"),
        ColSpec("long",   "has_promotion"),
        ColSpec("long",   "is_hot"),
    ])
    OUTPUT_SCHEMA = Schema([
        ColSpec("long",   "predicted_price_idr"),
        ColSpec("double", "predicted_price_juta"),
    ])

    artifacts = {
        "lgbm_model":     str(ARTIFACT_DIR / "lgbm_car_price.joblib"),
        "label_encoders": str(ARTIFACT_DIR / "label_encoders.joblib"),
        "metadata":       str(ARTIFACT_DIR / "model_metadata.json"),
    }

    with mlflow.start_run(run_name="lgbm-optuna-v1") as run:
        # Log params
        for k, v in meta["hyperparameters"].items():
            mlflow.log_param(f"lgb_{k}", v)
        mlflow.log_param("training_rows", meta["training_rows"])
        mlflow.log_param("algorithm",     "LightGBM")
        mlflow.log_param("tuner",         "Optuna 60 trials")

        # Log metrics
        mlflow.log_metric("cv_r2",        cv["R2"])
        mlflow.log_metric("cv_r2_std",    cv["R2_std"])
        mlflow.log_metric("cv_mae_juta",  cv["MAE_juta"])
        mlflow.log_metric("cv_mape_pct",  cv["MAPE_pct"])

        # Log + register model
        print("  [1/3] Logging model ke MLflow ...")
        mlflow.pyfunc.log_model(
            artifact_path         = "model",
            python_model          = OLXCarPriceModel(),
            artifacts             = artifacts,
            signature             = ModelSignature(
                inputs=INPUT_SCHEMA, outputs=OUTPUT_SCHEMA
            ),
            registered_model_name = MODEL_NAME,
            pip_requirements      = [
                f"lightgbm=={__import__('lightgbm').__version__}",
                f"scikit-learn=={__import__('sklearn').__version__}",
                f"pandas=={__import__('pandas').__version__}",
                f"numpy=={__import__('numpy').__version__}",
                f"joblib=={__import__('joblib').__version__}",
            ],
        )

        mlflow.set_tag("dataset",  "OLX Mobil Bekas Indonesia")
        mlflow.set_tag("status",   "production")
        run_id = run.info.run_id

    # Set alias "production"
    print("  [2/3] Mengatur alias @production ...")
    client  = mlflow.tracking.MlflowClient(MLFLOW_URI)
    version = sorted(
        client.search_model_versions(f"name='{MODEL_NAME}'"),
        key=lambda v: int(v.version)
    )[-1].version
    client.set_registered_model_alias(MODEL_NAME, "production", version)

    # Verifikasi
    print("  [3/3] Verifikasi model bisa di-load ...")
    test_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@production")
    test_input = pd.DataFrame([{
        "year": 2021, "mileage": 30, "merek": "Toyota",
        "transmisi": "Automatic", "bahan_bakar": "Bensin",
    }])
    result = test_model.predict(test_input)
    price  = result["predicted_price_juta"].iloc[0]

    print()
    print("━" * 50)
    print(f"  ✅ Registrasi berhasil!")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Version : {version}  (@production)")
    print(f"  Run ID  : {run_id}")
    print(f"  Test    : Toyota 2021 KM 30k → Rp {price} juta")
    print("━" * 50)
    print()
    print("  Langkah selanjutnya:")
    print("  1. Jalankan web app:")
    print("     python run.py --token YOUR_NGROK_TOKEN")
    print()
    print("  2. (Opsional) Lihat MLflow UI:")
    print("     mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    print()


if __name__ == "__main__":
    main()
