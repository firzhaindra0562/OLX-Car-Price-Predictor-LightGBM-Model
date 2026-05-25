"""
app.py — OLX Car Price Predictor Web App
FastAPI + HTML UI yang dapat diakses publik via ngrok
Model di-load dari MLflow Model Registry (@production alias)
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import mlflow
import mlflow.pyfunc

# ── MLflow Config ─────────────────────────────────────────────────────────
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI    = os.getenv("MLFLOW_MODEL_URI",    "models:/OLXCarPrice@production")
METADATA_DIR = Path(os.getenv("METADATA_DIR",  "model_artifacts"))

# ── Load model dari MLflow Registry ───────────────────────────────────────
print(f"  Loading model dari MLflow: {MODEL_URI}")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow_model = mlflow.pyfunc.load_model(MODEL_URI)
print(f"  ✅ Model loaded dari MLflow Registry")

# Metadata (kategori dropdown) tetap dibaca dari file lokal
with open(METADATA_DIR / "model_metadata.json") as f:
    meta = json.load(f)

CAT_COLS = ["merek","transmisi","bahan_bakar","tipe_bodi","kapasitas_cc","tipe_penjual","warna"]
OPTIONS  = {
    col: [c for c in meta["classes_per_encoder"][col] if c != "Unknown"]
    for col in CAT_COLS
}

# ── FastAPI App ───────────────────────────────────────────────────────────
app = FastAPI(title="OLX Car Price Predictor", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class CarInput(BaseModel):
    year:          int
    mileage:       int               # ribuan km
    merek:         str
    transmisi:     str
    bahan_bakar:   str
    tipe_bodi:     Optional[str] = "Unknown"
    kapasitas_cc:  Optional[str] = "Unknown"
    tipe_penjual:  Optional[str] = "Individu"
    warna:         Optional[str] = "Unknown"
    jumlah_foto:   Optional[int] = 5
    favorit:       Optional[int] = 0
    has_video:     Optional[int] = 0
    has_promotion: Optional[int] = 0
    is_hot:        Optional[int] = 0


def predict_price(car: dict) -> dict:
    # Kirim raw input langsung ke MLflow model.
    # Preprocessing (feature engineering + encoding) sudah dibungkus
    # di dalam OLXCarPriceModel.predict() di MLflow Registry.
    input_df = pd.DataFrame([{
        "year":          car["year"],
        "mileage":       car["mileage"],
        "merek":         car["merek"],
        "transmisi":     car.get("transmisi",     "Automatic"),
        "bahan_bakar":   car.get("bahan_bakar",   "Bensin"),
        "tipe_bodi":     car.get("tipe_bodi",     "Unknown"),
        "kapasitas_cc":  car.get("kapasitas_cc",  "Unknown"),
        "tipe_penjual":  car.get("tipe_penjual",  "Individu"),
        "warna":         car.get("warna",         "Unknown"),
        "jumlah_foto":   int(car.get("jumlah_foto",   5)),
        "favorit":       int(car.get("favorit",        0)),
        "has_video":     int(car.get("has_video",       0)),
        "has_promotion": int(car.get("has_promotion",   0)),
        "is_hot":        int(car.get("is_hot",          0)),
    }])

    result = mlflow_model.predict(input_df)
    price  = float(result["predicted_price_idr"].iloc[0])

    return {
        "price_idr":  int(price),
        "price_juta": round(price / 1e6, 1),
        "price_fmt":  f"Rp {price/1e6:,.1f} juta",
        "confidence": f"MLflow Registry · {MODEL_URI} · R²=0.78 · MAE≈Rp61jt",
        "model_uri":  MODEL_URI,
    }


# ── API endpoint ──────────────────────────────────────────────────────────
@app.post("/api/predict")
async def api_predict(car: CarInput):
    result = predict_price(car.model_dump())
    return JSONResponse(result)


@app.get("/api/options")
async def api_options():
    return JSONResponse(OPTIONS)


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_source": "MLflow Registry",
        "model_uri":    MODEL_URI,
        "tracking_uri": MLFLOW_URI,
        "version":      "1.0",
    }


# ── Web UI ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def ui(request: Request):
    opts = OPTIONS
    def opt_html(col, default=""):
        return "\n".join(
            f'<option value="{v}" {"selected" if v==default else ""}>{v}</option>'
            for v in opts[col]
        )

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Prediksi Harga Mobil Bekas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:      #f5f4ef;
    --surface: #ffffff;
    --border:  #e5e3dc;
    --text:    #1a1915;
    --muted:   #6b6860;
    --accent:  #c96a2a;
    --r:       8px;
    --font:    'Geist', ui-sans-serif, system-ui, sans-serif;
  }}
  body {{
    font-family: var(--font); background: var(--bg); color: var(--text);
    min-height: 100vh; font-size: 14px; line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }}
  .shell {{
    display: grid; grid-template-columns: 260px 1fr; min-height: 100vh;
  }}
  .sidebar {{
    background: var(--surface); border-right: 1px solid var(--border);
    padding: 28px 20px; display: flex; flex-direction: column; gap: 24px;
  }}
  .logo {{ padding-bottom: 20px; border-bottom: 1px solid var(--border); }}
  .logo-name {{ font-size: 15px; font-weight: 600; letter-spacing: -0.2px; }}
  .logo-sub  {{ font-size: 12px; color: var(--muted); margin-top: 3px; }}
  .section-label {{
    font-size: 11px; font-weight: 500; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px;
  }}
  .stat-list {{ display: flex; flex-direction: column; gap: 4px; }}
  .stat {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 10px; background: var(--bg); border-radius: var(--r); font-size: 12px;
  }}
  .stat-k {{ color: var(--muted); }}
  .stat-v {{ font-weight: 500; }}
  .main {{
    display: flex; flex-direction: column; align-items: center; padding: 48px 24px;
  }}
  .content {{ width: 100%; max-width: 540px; display: flex; flex-direction: column; gap: 16px; }}
  .page-title {{ font-size: 20px; font-weight: 600; letter-spacing: -0.3px; margin-bottom: 4px; }}
  .page-desc  {{ font-size: 13px; color: var(--muted); }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden;
  }}
  .card-head {{
    padding: 12px 18px; border-bottom: 1px solid var(--border);
    font-size: 11px; font-weight: 500; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.5px;
  }}
  .card-body {{ padding: 18px; display: flex; flex-direction: column; gap: 14px; }}
  .row-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  .field {{ display: flex; flex-direction: column; gap: 4px; }}
  .field label {{ font-size: 12px; font-weight: 500; color: var(--muted); }}
  input, select {{
    width: 100%; padding: 8px 10px; border: 1px solid var(--border);
    border-radius: var(--r); font-family: var(--font); font-size: 13.5px;
    color: var(--text); background: var(--surface); outline: none;
    transition: border-color .15s; -webkit-appearance: none; appearance: none;
  }}
  select {{
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11' height='11' viewBox='0 0 24 24' fill='none' stroke='%236b6860' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 10px center; padding-right: 28px;
  }}
  input:focus, select:focus {{
    border-color: var(--accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 10%, transparent);
  }}
  .seg {{
    display: flex; border: 1px solid var(--border); border-radius: var(--r);
    overflow: hidden; background: var(--bg);
  }}
  .seg-btn {{
    flex: 1; padding: 7px 6px; border: none; background: transparent;
    font-family: var(--font); font-size: 13px; color: var(--muted);
    cursor: pointer; transition: background .1s, color .1s;
  }}
  .seg-btn:not(:last-child) {{ border-right: 1px solid var(--border); }}
  .seg-btn.on {{ background: var(--surface); color: var(--text); font-weight: 500; box-shadow: 0 1px 2px rgba(0,0,0,.05); }}
  .btn {{
    width: 100%; padding: 9px; border: none; border-radius: var(--r);
    background: var(--text); color: white; font-family: var(--font);
    font-size: 13.5px; font-weight: 500; cursor: pointer; letter-spacing: -0.1px;
    transition: opacity .15s;
  }}
  .btn:hover {{ opacity: 0.87; }}
  .btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  .result-card {{ display: none; }}
  .result-top {{
    padding: 24px 18px 18px; border-bottom: 1px solid var(--border);
  }}
  .result-lbl {{ font-size: 11px; font-weight: 500; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
  .result-price {{
    font-size: 34px; font-weight: 300; letter-spacing: -1.5px; color: var(--text); line-height: 1;
    margin: 6px 0 4px;
  }}
  .result-price b {{ font-weight: 600; color: var(--accent); }}
  .result-vehicle {{ font-size: 12px; color: var(--muted); }}
  .result-foot {{ padding: 14px 18px; display: flex; flex-direction: column; gap: 8px; }}
  .bar-labels {{ display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); }}
  .bar-track {{ height: 2px; background: var(--border); border-radius: 2px; }}
  .bar-fill {{ height: 100%; border-radius: 2px; background: var(--accent); transition: width .6s cubic-bezier(.4,0,.2,1); }}
  .result-meta {{ font-size: 11px; color: var(--muted); }}
  .err {{
    font-size: 13px; color: #b91c1c; padding: 8px 10px;
    background: #fff1f0; border: 1px solid #fecaca; border-radius: var(--r); display: none;
  }}
  .spinner {{
    width: 12px; height: 12px; border: 2px solid rgba(255,255,255,.3);
    border-top-color: white; border-radius: 50%;
    animation: spin .6s linear infinite; display: inline-block; vertical-align: middle; margin-right: 5px;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  @media (max-width: 680px) {{
    .shell {{ grid-template-columns: 1fr; }}
    .sidebar {{ border-right: none; border-bottom: 1px solid var(--border); flex-direction: row; flex-wrap: wrap; gap: 16px; padding: 16px; }}
    .logo {{ padding-bottom: 0; border-bottom: none; flex: 1 1 100%; }}
    .main {{ padding: 20px 14px; }}
  }}
</style>
</head>
<body>
<div class="shell">

  <aside class="sidebar">
    <div class="logo">
      <div class="logo-name">Prediksi Harga Mobil</div>
      <div class="logo-sub">OLX Indonesia &mdash; LightGBM</div>
    </div>
    <div>
      <div class="section-label">Performa Model</div>
      <div class="stat-list">
        <div class="stat"><span class="stat-k">R&sup2; Score</span><span class="stat-v">0.78</span></div>
        <div class="stat"><span class="stat-k">MAE</span><span class="stat-v">Rp 61 juta</span></div>
        <div class="stat"><span class="stat-k">MAPE</span><span class="stat-v">22%</span></div>
        <div class="stat"><span class="stat-k">Validasi</span><span class="stat-v">5-Fold CV</span></div>
        <div class="stat"><span class="stat-k">Data</span><span class="stat-v">1.432 listing</span></div>
      </div>
    </div>
    <div>
      <div class="section-label">Pipeline</div>
      <div class="stat-list">
        <div class="stat"><span class="stat-k">Algoritma</span><span class="stat-v">LightGBM</span></div>
        <div class="stat"><span class="stat-k">Tuning</span><span class="stat-v">Optuna</span></div>
        <div class="stat"><span class="stat-k">Registry</span><span class="stat-v">MLflow</span></div>
        <div class="stat"><span class="stat-k">Fitur</span><span class="stat-v">19 fitur</span></div>
      </div>
    </div>
  </aside>

  <main class="main">
    <div class="content">
      <div>
        <div class="page-title">Estimasi harga kendaraan</div>
        <div class="page-desc">Masukkan detail kendaraan untuk mendapatkan estimasi harga pasar.</div>
      </div>

      <div class="card">
        <div class="card-head">Detail Kendaraan</div>
        <div class="card-body">

          <div class="row-2">
            <div class="field">
              <label>Merek</label>
              <select id="merek">
                <option value="">Pilih merek</option>
                {opt_html("merek","Toyota")}
              </select>
            </div>
            <div class="field">
              <label>Tipe Bodi</label>
              <select id="tipe_bodi">
                <option value="">Pilih tipe</option>
                {opt_html("tipe_bodi","SUV")}
              </select>
            </div>
          </div>

          <div class="row-2">
            <div class="field">
              <label>Tahun</label>
              <input type="number" id="year" value="2020" min="1980" max="2026">
            </div>
            <div class="field">
              <label>Kilometer (ribuan)</label>
              <input type="number" id="mileage" value="50" min="0" max="500">
            </div>
          </div>

          <div class="row-2">
            <div class="field">
              <label>Kapasitas Mesin</label>
              <select id="kapasitas_cc">
                {opt_html("kapasitas_cc",">1.500 - 2.000 cc")}
              </select>
            </div>
            <div class="field">
              <label>Warna</label>
              <select id="warna">
                <option value="">Pilih warna</option>
                {opt_html("warna","Hitam")}
              </select>
            </div>
          </div>

          <div class="field">
            <label>Transmisi</label>
            <div class="seg" id="seg-tr">
              <button class="seg-btn on" data-v="Automatic" onclick="pick(this,'seg-tr')">Automatic</button>
              <button class="seg-btn"    data-v="Manual"    onclick="pick(this,'seg-tr')">Manual</button>
            </div>
          </div>

          <div class="field">
            <label>Bahan Bakar</label>
            <div class="seg" id="seg-bb">
              <button class="seg-btn on" data-v="Bensin"  onclick="pick(this,'seg-bb')">Bensin</button>
              <button class="seg-btn"    data-v="Diesel"  onclick="pick(this,'seg-bb')">Diesel</button>
              <button class="seg-btn"    data-v="Hybrid"  onclick="pick(this,'seg-bb')">Hybrid</button>
              <button class="seg-btn"    data-v="Listrik" onclick="pick(this,'seg-bb')">Listrik</button>
            </div>
          </div>

          <div class="field">
            <label>Tipe Penjual</label>
            <div class="seg" id="seg-pj">
              <button class="seg-btn on" data-v="Individu" onclick="pick(this,'seg-pj')">Individu</button>
              <button class="seg-btn"    data-v="Diler"    onclick="pick(this,'seg-pj')">Diler</button>
            </div>
          </div>

          <div class="err" id="err"></div>

          <button class="btn" id="btn" onclick="go()">Hitung Estimasi Harga</button>
        </div>
      </div>

      <div class="card result-card" id="res">
        <div class="result-top">
          <div class="result-lbl">Estimasi Harga Pasar</div>
          <div class="result-price" id="res-price">—</div>
          <div class="result-vehicle" id="res-vehicle">—</div>
        </div>
        <div class="result-foot">
          <div>
            <div class="bar-labels">
              <span>Rp 15 juta</span>
              <span id="bar-mid" style="font-weight:500;color:var(--text)">—</span>
              <span>Rp 3 miliar</span>
            </div>
            <div class="bar-track" style="margin-top:6px">
              <div class="bar-fill" id="bar-fill" style="width:0%"></div>
            </div>
          </div>
          <div class="result-meta">LightGBM · MLflow Registry · R&sup2;=0.78 · MAE Rp 61 juta · Data OLX Indonesia</div>
        </div>
      </div>

    </div>
  </main>
</div>

<script>
function pick(el, gid) {{
  document.querySelectorAll('#'+gid+' .seg-btn').forEach(b=>b.classList.remove('on'));
  el.classList.add('on');
}}
function val(gid) {{
  const a = document.querySelector('#'+gid+' .seg-btn.on');
  return a ? a.dataset.v : null;
}}
async function go() {{
  const btn = document.getElementById('btn');
  const err = document.getElementById('err');
  err.style.display = 'none';

  const merek    = document.getElementById('merek').value;
  const year     = parseInt(document.getElementById('year').value);
  const mileage  = parseInt(document.getElementById('mileage').value);
  const tipe_bodi= document.getElementById('tipe_bodi').value;
  const kap      = document.getElementById('kapasitas_cc').value;
  const warna    = document.getElementById('warna').value;
  const tr       = val('seg-tr');
  const bb       = val('seg-bb');
  const pj       = val('seg-pj');

  if (!merek)                              return showErr('Pilih merek kendaraan.');
  if (!year || year<1980 || year>2026)     return showErr('Tahun harus antara 1980–2026.');
  if (isNaN(mileage))                      return showErr('Masukkan kilometer kendaraan.');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Menghitung...';

  try {{
    const r = await fetch('/api/predict', {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        year, mileage, merek,
        transmisi:    tr  || 'Automatic',
        bahan_bakar:  bb  || 'Bensin',
        tipe_bodi:    tipe_bodi || 'Unknown',
        kapasitas_cc: kap || 'Unknown',
        tipe_penjual: pj  || 'Individu',
        warna:        warna || 'Unknown',
        jumlah_foto:5, favorit:0, has_video:0, has_promotion:0, is_hot:0
      }})
    }});
    if (!r.ok) throw new Error();
    const d = await r.json();
    const juta = d.price_juta;
    const fmt  = juta>=1000
      ? 'Rp ' + (juta/1000).toLocaleString('id-ID',{{minimumFractionDigits:2,maximumFractionDigits:2}}) + ' miliar'
      : 'Rp ' + juta.toLocaleString('id-ID') + ' juta';

    document.getElementById('res-price').innerHTML = fmt.replace(/([0-9][0-9.,]*)/, '<b>$1</b>');
    document.getElementById('res-vehicle').textContent =
      merek+' '+year+' · KM '+mileage.toLocaleString('id-ID')+'.000 · '+(tipe_bodi||'')+(tipe_bodi&&tr?' · ':'')+tr+' · '+bb;

    const pct = Math.min(100,Math.max(1,(Math.log(Math.max(15,juta))-Math.log(15))/(Math.log(3000)-Math.log(15))*100));
    document.getElementById('bar-fill').style.width = pct+'%';
    document.getElementById('bar-mid').textContent  = fmt;

    const card = document.getElementById('res');
    card.style.display = 'block';
    card.scrollIntoView({{behavior:'smooth',block:'nearest'}});
  }} catch(e) {{
    showErr('Gagal menghubungi server. Pastikan server aktif.');
  }} finally {{
    btn.disabled = false;
    btn.innerHTML = 'Hitung Estimasi Harga';
  }}
}}
function showErr(m) {{
  const e = document.getElementById('err');
  e.textContent = m; e.style.display = 'block';
}}
</script>
</body>
</html>"""
    return HTMLResponse(html)
