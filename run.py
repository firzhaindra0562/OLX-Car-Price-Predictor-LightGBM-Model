"""
run.py — OLX Car Price Predictor Launcher
==========================================
Menjalankan FastAPI server + membuka tunnel ngrok ke publik.

Cara pakai:
    python run.py                          # default port 8000
    python run.py --port 8080              # port custom
    python run.py --token YOUR_TOKEN       # pakai ngrok auth token
    python run.py --no-ngrok               # hanya local (tanpa tunnel)

Setup ngrok (sekali saja):
    1. Daftar gratis di https://ngrok.com
    2. Copy auth token dari https://dashboard.ngrok.com/get-started/your-authtoken
    3. python run.py --token YOUR_TOKEN
       (atau set env: NGROK_AUTHTOKEN=your_token)
"""

import argparse
import os
import sys
import time
import threading
import uvicorn
from pyngrok import ngrok, conf

# ── Parse args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="OLX Car Price Predictor Web App")
parser.add_argument("--port",        type=int, default=8000,  help="Local port (default: 8000)")
parser.add_argument("--token",       type=str, default=None,  help="Ngrok auth token")
parser.add_argument("--no-ngrok",    action="store_true",     help="Run locally only (no tunnel)")
parser.add_argument("--mlflow-uri",  type=str, default=None,  help="MLflow tracking URI (default: sqlite:///mlflow.db)")
parser.add_argument("--model-uri",   type=str, default=None,  help="MLflow model URI (default: models:/OLXCarPrice@production)")
args = parser.parse_args()

PORT      = args.port
USE_NGROK = not args.no_ngrok

# Set MLflow env vars sebelum app.py di-import
if args.mlflow_uri:
    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
if args.model_uri:
    os.environ["MLFLOW_MODEL_URI"] = args.model_uri

def start_server():
    """Jalankan FastAPI dengan uvicorn."""
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, log_level="warning")

def open_ngrok(port, token=None):
    """Buka tunnel ngrok dan print URL publik."""
    time.sleep(1.5)  # tunggu server siap

    # Set auth token jika disediakan atau dari env
    ngrok_token = token or os.environ.get("NGROK_AUTHTOKEN")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        print(f"  ✅ Ngrok auth token terkonfigurasi")
    else:
        print("  ⚠️  Tidak ada auth token — menggunakan sesi anonim ngrok")
        print("      (Tunnel aktif ±2 jam, 1 tunnel sekaligus)")
        print("      Daftar gratis: https://ngrok.com\n")

    tunnel = ngrok.connect(port, "http")
    public_url = tunnel.public_url

    print("━" * 55)
    print("  🌐 APLIKASI BERHASIL DIPUBLIKASIKAN!")
    print("━" * 55)
    print(f"\n  🔗 URL Publik  : {public_url}")
    print(f"  🏠 URL Lokal   : http://localhost:{port}")
    print(f"  📖 API Docs    : {public_url}/docs")
    print(f"  📡 API Predict : {public_url}/api/predict")
    print(f"\n  Bagikan URL ini ke publik ↑")
    print(f"\n  Tekan Ctrl+C untuk menghentikan server")
    print("━" * 55)

    return public_url

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  OLX Car Price Predictor — Web App")
    print("═" * 55)
    mlflow_uri   = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    model_uri    = os.environ.get("MLFLOW_MODEL_URI",    "models:/OLXCarPrice@production")
    print(f"\n  Port         : {PORT}")
    print(f"  Ngrok        : {'Ya' if USE_NGROK else 'Tidak (lokal saja)'}")
    print(f"  MLflow URI   : {mlflow_uri}")
    print(f"  Model URI    : {model_uri}\n")

    # Verifikasi model ada
    from pathlib import Path
    if not Path("model_artifacts/lgbm_car_price.joblib").exists():
        print("❌ model_artifacts/ tidak ditemukan!")
        print("   Pastikan folder model_artifacts/ ada di direktori yang sama dengan run.py")
        sys.exit(1)

    if USE_NGROK:
        # Buka ngrok di thread terpisah
        ngrok_thread = threading.Thread(
            target=open_ngrok, args=(PORT, args.token), daemon=True
        )
        ngrok_thread.start()
    else:
        print(f"  ✅ Buka browser: http://localhost:{PORT}")
        print("  Tekan Ctrl+C untuk stop\n")

    # Jalankan server (blocking)
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n\n  👋 Server dihentikan.")
        if USE_NGROK:
            ngrok.kill()
