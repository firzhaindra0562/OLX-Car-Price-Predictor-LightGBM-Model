"""
╔══════════════════════════════════════════════════════════════════════╗
║        OLX MOBIL BEKAS — FULL ML PIPELINE                           ║
║        Cleaning → EDA → Feature Engineering → Modeling              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, json
from pathlib import Path

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ─── Style & Palette ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.dpi":        130,
})

BLUE    = "#2563EB"
GREEN   = "#16A34A"
ORANGE  = "#EA580C"
RED     = "#DC2626"
PURPLE  = "#7C3AED"
TEAL    = "#0891B2"
GRAY    = "#64748B"
PALETTE = [BLUE, GREEN, ORANGE, RED, PURPLE, TEAL, "#DB2777", "#CA8A04"]

DATA_PATH  = r"C:\Users\firzh\OneDrive\Dokumen\Kuliah\Scraping\best model training\olx_final.csv"
OUT_DIR    = Path(".")
N_TRIALS   = 60
N_FOLDS    = 5
SEED       = 42

FEATURES = ["year", "mileage", "merek", "transmisi", "bahan_bakar",
            "tipe_bodi", "kapasitas_cc", "tipe_penjual", "warna",
            "jumlah_foto", "favorit", "has_video", "has_promotion", "is_hot"]
TARGET   = "price"
CAT_COLS = ["merek","transmisi","bahan_bakar","tipe_bodi",
            "kapasitas_cc","tipe_penjual","warna"]
BOOL_COLS = ["has_video","has_promotion","is_hot"]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 ─ DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def clean_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df   = df_raw.copy()
    log  = {}
    n0   = len(df)
    
    print("━"*60)
    print("  STEP 1 — DATA CLEANING")
    print("━"*60)
    print(f"  Raw rows  : {n0:,}")

    # ── 1a. Drop fully empty or corrupted rows ─────────────────────────────
    df = df.dropna(subset=["price","year","mileage"])
    log["drop_missing_core"] = n0 - len(df)
    print(f"  [1a] Drop rows missing price/year/mileage : -{log['drop_missing_core']}")

    # ── 1b. Duplicate by listing ID (keep first scrape) ───────────────────
    before = len(df)
    df = df.sort_values("created_at", ascending=False)
    df = df.drop_duplicates(subset=["id"], keep="first")
    log["dup_id"] = before - len(df)
    print(f"  [1b] Duplicate listing ID removed         : -{log['dup_id']}")

    # ── 1c. Semantic duplicates (same title + price + year + km) ──────────
    before = len(df)
    df = df.drop_duplicates(subset=["judul","harga_raw","tahun","km_raw"], keep="first")
    log["dup_semantic"] = before - len(df)
    print(f"  [1c] Semantic duplicates removed          : -{log['dup_semantic']}")

    # ── 1d. Price floor: < 15 juta → likely data error / non-car ──────────
    before = len(df)
    df = df[df["price"] >= 15_000_000]
    log["price_floor"] = before - len(df)
    print(f"  [1d] Price < Rp 15 juta removed           : -{log['price_floor']}")

    # ── 1e. Anomalous price-per-brand outliers (IQR on log-price per merek)
    before = len(df)
    df["log_price"] = np.log1p(df["price"])

    def iqr_mask(group):
        Q1, Q3 = group["log_price"].quantile([0.25, 0.75])
        IQR    = Q3 - Q1
        return (group["log_price"] >= Q1 - 2.5*IQR) & \
               (group["log_price"] <= Q3 + 2.5*IQR)

    valid_mask = df.groupby("merek", group_keys=False).apply(iqr_mask)
    df         = df[valid_mask]
    log["price_outlier_brand"] = before - len(df)
    print(f"  [1e] Brand-level price outliers removed   : -{log['price_outlier_brand']}")

    # ── 1f. Year sanity: only 1980–2026 ───────────────────────────────────
    before = len(df)
    df     = df[(df["year"] >= 1980) & (df["year"] <= 2026)]
    log["year_invalid"] = before - len(df)
    print(f"  [1f] Invalid year removed                 : -{log['year_invalid']}")

    # ── 1g. Mileage cap: > 500 rb km is almost certainly a unit error ─────
    #        (mileage column is km_raw which is in thousands → max real = 500)
    before = len(df)
    df     = df[df["mileage"] <= 500]
    log["km_cap"] = before - len(df)
    print(f"  [1g] Mileage > 500k km removed            : -{log['km_cap']}")

    # ── 1h. Fill missing categoricals ────────────────────────────────────
    for c in CAT_COLS:
        df[c] = df[c].fillna("Unknown")
    for c in BOOL_COLS:
        df[c] = df[c].fillna(False).astype(int)

    df = df.drop(columns=["log_price"], errors="ignore")
    
    n_final = len(df)
    log["total_removed"] = n0 - n_final
    log["final_rows"]    = n_final
    
    print(f"\n  {'─'*40}")
    print(f"  Total removed  : {log['total_removed']:,} rows  ({log['total_removed']/n0*100:.1f}%)")
    print(f"  Clean dataset  : {n_final:,} rows\n")
    return df, log


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 ─ EDA VISUALIZATIONS (3 figure files)
# ══════════════════════════════════════════════════════════════════════════════
def plot_eda(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    print("━"*60)
    print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
    print("━"*60)

    # ── Figure 1: Overview & Cleaning Impact ──────────────────────────────
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle("OLX Mobil Bekas — Dataset Overview & Cleaning Impact",
                  fontsize=15, fontweight="bold", y=1.01)

    # [0,0] Price distribution — raw vs clean
    ax = axes[0, 0]
    ax.hist(df_raw["price"]/1e6,  bins=60, alpha=0.5, color=RED,   label=f"Raw (n={len(df_raw):,})", density=True)
    ax.hist(df_clean["price"]/1e6, bins=60, alpha=0.7, color=BLUE, label=f"Clean (n={len(df_clean):,})", density=True)
    ax.set_xlabel("Harga (Rp Juta)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Distribusi Harga: Raw vs Clean", fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # [0,1] Price log-scale distribution
    ax = axes[0, 1]
    vals = np.log10(df_clean["price"])
    ax.hist(vals, bins=50, color=BLUE, edgecolor="white", alpha=0.85)
    ax.axvline(vals.mean(), color=RED, linestyle="--", linewidth=2, label=f"Mean = 10^{vals.mean():.2f}")
    ax.axvline(vals.median(), color=ORANGE, linestyle="--", linewidth=2, label=f"Median = 10^{vals.median():.2f}")
    ax.set_xlabel("log₁₀(Harga)", fontsize=10)
    ax.set_title("Distribusi Harga (log-scale)", fontweight="bold")
    ax.legend(fontsize=9)
    tick_vals = [7, 7.5, 8, 8.5, 9]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f"Rp{10**v/1e6:.0f}jt" if v < 8.5 else f"Rp{10**v/1e9:.1f}M" for v in tick_vals], fontsize=8)

    # [0,2] Year distribution
    ax = axes[0, 2]
    yr_counts = df_clean["year"].value_counts().sort_index()
    ax.bar(yr_counts.index, yr_counts.values, color=GREEN, alpha=0.85, width=0.85, edgecolor="white")
    ax.set_xlabel("Tahun", fontsize=10)
    ax.set_ylabel("Jumlah Listing", fontsize=10)
    ax.set_title("Distribusi Tahun Kendaraan", fontweight="bold")
    ax.set_xlim(1979, 2027)

    # [1,0] Mileage distribution
    ax = axes[1, 0]
    ax.hist(df_clean["mileage"], bins=50, color=ORANGE, edgecolor="white", alpha=0.85)
    ax.axvline(df_clean["mileage"].mean(), color=RED, linestyle="--", linewidth=2,
               label=f"Mean = {df_clean['mileage'].mean():.0f}k km")
    ax.set_xlabel("Kilometer (ribuan)", fontsize=10)
    ax.set_title("Distribusi Kilometer", fontweight="bold")
    ax.legend(fontsize=9)

    # [1,1] Missing values heatmap
    ax = axes[1, 1]
    miss_cols = ["harga_diskon","diskon_amount","varian","tipe_bodi",
                 "kapasitas_cc","tipe_penjual","bursa_mobil"]
    miss_pct = [(df_clean[c].isna().mean()*100) for c in miss_cols]
    colors_miss = [RED if p > 50 else ORANGE if p > 20 else GREEN for p in miss_pct]
    bars = ax.barh(miss_cols, miss_pct, color=colors_miss, edgecolor="white")
    for bar, pct in zip(bars, miss_pct):
        ax.text(pct+0.5, bar.get_y()+bar.get_height()/2, f"{pct:.1f}%",
                va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Missing (%)", fontsize=10)
    ax.set_title("Missing Values per Kolom", fontweight="bold")
    ax.set_xlim(0, 115)

    # [1,2] Cleaning funnel
    ax = axes[1, 2]
    steps  = ["Raw\nData", "Drop\nMissing Core", "Dup\nID", "Dup\nSemantik",
              "Price\nFloor", "Brand\nOutlier", "Final\nClean"]
    counts = [len(df_raw), len(df_raw)-0, len(df_raw)-0-0, len(df_raw)-0-0-0,
              len(df_raw)-0-0-0-0, len(df_raw)-0-0-0-0-0, len(df_clean)]
    # Recalculate from actual cleaning
    n = len(df_raw)
    funnel = [n]
    for step_key in ["drop_missing_core","dup_id","dup_semantic",
                     "price_floor","price_outlier_brand"]:
        n -= cleaning_log.get(step_key, 0)
        funnel.append(n)
    funnel.append(len(df_clean))
    steps2 = ["Raw","−Missing Core","−Dup ID","−Dup Semantik",
              "−Price Floor","−Brand Outlier","Final"]
    clrs = [GRAY] + [RED]*5 + [GREEN]
    ax.barh(range(len(steps2)), funnel, color=clrs, edgecolor="white", alpha=0.85)
    for i, (v, lbl) in enumerate(zip(funnel, steps2)):
        ax.text(v+5, i, f"{v:,}", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(steps2)))
    ax.set_yticklabels(steps2, fontsize=9)
    ax.set_xlabel("Jumlah Baris", fontsize=10)
    ax.set_title("Data Cleaning Funnel", fontweight="bold")
    ax.set_xlim(0, max(funnel)*1.15)

    fig1.tight_layout()
    fig1.savefig(OUT_DIR/"eda_1_overview.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print("  ✅ eda_1_overview.png")

    # ── Figure 2: Price Analysis ───────────────────────────────────────────
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle("OLX Mobil Bekas — Analisis Harga", fontsize=15, fontweight="bold", y=1.01)

    # [0,0] Price by brand (top 10)
    ax = axes[0, 0]
    top_brands = df_clean["merek"].value_counts().head(10).index
    brand_med  = df_clean[df_clean["merek"].isin(top_brands)]\
                     .groupby("merek")["price"].median().sort_values(ascending=True)/1e6
    bars = ax.barh(brand_med.index, brand_med.values, color=PALETTE[:len(brand_med)], edgecolor="white")
    for bar, val in zip(bars, brand_med.values):
        ax.text(val+2, bar.get_y()+bar.get_height()/2, f"Rp{val:.0f}jt",
                va="center", fontsize=8)
    ax.set_xlabel("Median Harga (Rp Juta)", fontsize=10)
    ax.set_title("Median Harga per Merek (Top 10)", fontweight="bold")

    # [0,1] Price by body type
    ax = axes[0, 1]
    body_med = df_clean.groupby("tipe_bodi")["price"].median().dropna()\
                   .sort_values(ascending=True)/1e6
    body_med = body_med[body_med.index != "Unknown"]
    clrs = [BLUE if v < 300 else ORANGE if v < 600 else RED for v in body_med.values]
    bars = ax.barh(body_med.index, body_med.values, color=clrs, edgecolor="white")
    for bar, val in zip(bars, body_med.values):
        ax.text(val+2, bar.get_y()+bar.get_height()/2, f"{val:.0f}jt",
                va="center", fontsize=8)
    ax.set_xlabel("Median Harga (Rp Juta)", fontsize=10)
    ax.set_title("Median Harga per Tipe Bodi", fontweight="bold")

    # [0,2] Price by transmission
    ax = axes[0, 2]
    tr_data = [df_clean[df_clean["transmisi"]==t]["price"].values/1e6
               for t in df_clean["transmisi"].dropna().unique()]
    tr_lbls = df_clean["transmisi"].dropna().unique().tolist()
    bp = ax.boxplot(tr_data, patch_artist=True, notch=True,
                    medianprops={"color":"white","linewidth":2})
    for patch, clr in zip(bp["boxes"], [BLUE, GREEN]):
        patch.set_facecolor(clr)
        patch.set_alpha(0.7)
    ax.set_xticklabels(tr_lbls, fontsize=10)
    ax.set_ylabel("Harga (Rp Juta)", fontsize=10)
    ax.set_title("Distribusi Harga per Transmisi", fontweight="bold")

    # [1,0] Price vs Year scatter
    ax = axes[1, 0]
    sc = ax.scatter(df_clean["year"], df_clean["price"]/1e6,
                    alpha=0.35, s=15, c=df_clean["mileage"],
                    cmap="RdYlGn_r", edgecolors="none")
    cbar = fig2.colorbar(sc, ax=ax)
    cbar.set_label("Kilometer (ribuan)", fontsize=8)
    # Trend line
    z = np.polyfit(df_clean["year"], df_clean["price"]/1e6, 2)
    p = np.poly1d(z)
    xr = np.linspace(df_clean["year"].min(), df_clean["year"].max(), 200)
    ax.plot(xr, p(xr), color=RED, linewidth=2, label="Trend (poly-2)")
    ax.set_xlabel("Tahun", fontsize=10)
    ax.set_ylabel("Harga (Rp Juta)", fontsize=10)
    ax.set_title("Harga vs Tahun (warna = KM)", fontweight="bold")
    ax.legend(fontsize=9)

    # [1,1] Price vs Mileage scatter
    ax = axes[1, 1]
    ax.scatter(df_clean["mileage"], df_clean["price"]/1e6,
               alpha=0.35, s=15, color=PURPLE, edgecolors="none")
    z2 = np.polyfit(df_clean["mileage"], df_clean["price"]/1e6, 1)
    p2 = np.poly1d(z2)
    xr2 = np.linspace(0, df_clean["mileage"].max(), 200)
    ax.plot(xr2, p2(xr2), color=RED, linewidth=2, label=f"Trend (r={df_clean['mileage'].corr(df_clean['price']):.2f})")
    ax.set_xlabel("Kilometer (ribuan)", fontsize=10)
    ax.set_ylabel("Harga (Rp Juta)", fontsize=10)
    ax.set_title("Harga vs Kilometer", fontweight="bold")
    ax.legend(fontsize=9)

    # [1,2] Price by fuel type
    ax = axes[1, 2]
    fuel_med = df_clean.groupby("bahan_bakar")["price"].median().sort_values()/1e6
    clrs = {"Bensin": BLUE, "Diesel": GREEN, "Hybrid": ORANGE, "Electric": RED}
    bar_clrs = [clrs.get(f, GRAY) for f in fuel_med.index]
    bars = ax.bar(fuel_med.index, fuel_med.values, color=bar_clrs, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, fuel_med.values):
        ax.text(bar.get_x()+bar.get_width()/2, val+3, f"{val:.0f}jt",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Median Harga (Rp Juta)", fontsize=10)
    ax.set_title("Median Harga per Bahan Bakar", fontweight="bold")

    fig2.tight_layout()
    fig2.savefig(OUT_DIR/"eda_2_price_analysis.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print("  ✅ eda_2_price_analysis.png")

    # ── Figure 3: Feature Correlations & Market Insights ─────────────────
    fig3, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig3.suptitle("OLX Mobil Bekas — Korelasi Fitur & Insight Pasar",
                  fontsize=15, fontweight="bold", y=1.01)

    # [0,0] Correlation heatmap (numeric features)
    ax = axes[0, 0]
    num_df = df_clean[["price","year","mileage","jumlah_foto","favorit"]].copy()
    num_df.columns = ["Harga","Tahun","KM","Jml Foto","Favorit"]
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, mask=mask, vmin=-1, vmax=1,
                annot_kws={"size":10, "weight":"bold"},
                linewidths=0.5, cbar_kws={"shrink":0.8})
    ax.set_title("Korelasi Antar Fitur Numerik", fontweight="bold")

    # [0,1] Volume listing per merek (top 12)
    ax = axes[0, 1]
    top12 = df_clean["merek"].value_counts().head(12)
    ax.barh(top12.index[::-1], top12.values[::-1], color=PALETTE[:12][::-1], edgecolor="white")
    for i, (val, lbl) in enumerate(zip(top12.values[::-1], top12.index[::-1])):
        ax.text(val+1, i, str(val), va="center", fontsize=9)
    ax.set_xlabel("Jumlah Listing", fontsize=10)
    ax.set_title("Volume Listing per Merek (Top 12)", fontweight="bold")

    # [0,2] Penjual Individu vs Dealer
    ax = axes[0, 2]
    sell_types = df_clean["tipe_penjual"].value_counts()
    pie_clrs   = [BLUE, GREEN, ORANGE]
    wedges, texts, autotexts = ax.pie(
        sell_types.values, labels=sell_types.index,
        autopct="%1.1f%%", colors=pie_clrs, startangle=90,
        pctdistance=0.78, wedgeprops={"edgecolor":"white","linewidth":2})
    for at in autotexts:
        at.set_fontsize(10); at.set_fontweight("bold")
    ax.set_title("Proporsi Tipe Penjual", fontweight="bold")

    # [1,0] Price trend by year (median + IQR band)
    ax = axes[1, 0]
    yr_grp = df_clean[df_clean["year"] >= 2010].groupby("year")["price"]
    yr_med = yr_grp.median()/1e6
    yr_q25 = yr_grp.quantile(0.25)/1e6
    yr_q75 = yr_grp.quantile(0.75)/1e6
    ax.fill_between(yr_med.index, yr_q25, yr_q75, alpha=0.25, color=BLUE, label="IQR (25-75%)")
    ax.plot(yr_med.index, yr_med.values, color=BLUE, linewidth=2.5, marker="o", markersize=5, label="Median")
    ax.set_xlabel("Tahun", fontsize=10)
    ax.set_ylabel("Harga (Rp Juta)", fontsize=10)
    ax.set_title("Tren Median Harga per Tahun (2010+)", fontweight="bold")
    ax.legend(fontsize=9)

    # [1,1] KM distribution per brand top 6
    ax = axes[1, 1]
    top6_brands = df_clean["merek"].value_counts().head(6).index.tolist()
    data_km = [df_clean[df_clean["merek"]==b]["mileage"].values for b in top6_brands]
    bp = ax.boxplot(data_km, patch_artist=True, notch=False,
                    medianprops={"color":"white","linewidth":2})
    for patch, clr in zip(bp["boxes"], PALETTE[:6]):
        patch.set_facecolor(clr); patch.set_alpha(0.75)
    ax.set_xticklabels(top6_brands, rotation=20, fontsize=8)
    ax.set_ylabel("Kilometer (ribuan)", fontsize=10)
    ax.set_title("Distribusi KM per Merek (Top 6)", fontweight="bold")

    # [1,2] Listing count per province (top 8)
    ax = axes[1, 2]
    prov_count = df_clean["provinsi"].value_counts().head(8)
    ax.pie(prov_count.values, labels=prov_count.index,
           autopct="%1.1f%%", colors=PALETTE, startangle=140,
           pctdistance=0.80, wedgeprops={"edgecolor":"white","linewidth":1.5})
    ax.set_title("Distribusi Listing per Provinsi (Top 8)", fontweight="bold")

    fig3.tight_layout()
    fig3.savefig(OUT_DIR/"eda_3_insights.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    print("  ✅ eda_3_insights.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 ─ FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame):
    print("━"*60)
    print("  STEP 3 — FEATURE ENGINEERING")
    print("━"*60)

    df = df.copy()

    # Car age (from 2026)
    df["car_age"]     = 2026 - df["year"]

    # KM per year (proxy for usage intensity)
    df["km_per_year"] = df["mileage"] / (df["car_age"].clip(lower=1))

    # Price-proxy score: high photo count + favorited = seller invests more
    df["listing_quality"] = df["jumlah_foto"] * 0.7 + df["favorit"] * 0.3

    # Is luxury brand?
    luxury = {"BMW","Mercedes-Benz","Porsche","Lexus","Audi","Jaguar","Volvo","Land Rover"}
    df["is_luxury"] = df["merek"].isin(luxury).astype(int)

    # Is Japanese brand?
    japanese = {"Toyota","Honda","Daihatsu","Suzuki","Mitsubishi","Nissan","Mazda","Subaru"}
    df["is_japanese"] = df["merek"].isin(japanese).astype(int)

    engineered = ["car_age","km_per_year","listing_quality","is_luxury","is_japanese"]
    print(f"  New features added: {engineered}\n")

    global FEATURES
    FEATURES = FEATURES + engineered
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 ─ MODELING: XGBoost vs LightGBM + Optuna
# ══════════════════════════════════════════════════════════════════════════════
def prepare_ml(df: pd.DataFrame):
    data = df[FEATURES + [TARGET]].copy()
    for c in CAT_COLS:
        data[c] = data[c].fillna("Unknown")
    for c in BOOL_COLS:
        data[c] = data[c].fillna(0).astype(int)

    le_dict = {}
    for c in CAT_COLS:
        le = LabelEncoder()
        data[c] = le.fit_transform(data[c].astype(str))
        le_dict[c] = le

    data = data.dropna()
    X     = data[FEATURES]
    y_log = np.log1p(data[TARGET].values)
    y_raw = data[TARGET].values
    return X, y_log, y_raw, le_dict


def cv_metrics(model, X, y_log, y_raw, kf, label):
    maes, mapes, r2s = [], [], []
    for tr, val in kf.split(X):
        model.fit(X.iloc[tr], y_log[tr])
        pred_log = model.predict(X.iloc[val])
        pred_idr = np.expm1(pred_log)
        true_idr = y_raw[val]
        maes.append(np.mean(np.abs(pred_idr - true_idr)))
        mapes.append(np.mean(np.abs((pred_idr - true_idr)/true_idr)) * 100)
        ss_r = np.sum((y_log[val] - pred_log)**2)
        ss_t = np.sum((y_log[val] - y_log[val].mean())**2)
        r2s.append(1 - ss_r/ss_t)
    return {"label": label,
            "r2":    np.mean(r2s),   "r2_std":  np.std(r2s),
            "mae":   np.mean(maes),  "mape":    np.mean(mapes)}


def tune_xgboost(X, y_log, kf):
    def obj(trial):
        p = {"n_estimators":    trial.suggest_int("n_estimators", 200, 1000),
             "max_depth":       trial.suggest_int("max_depth", 3, 10),
             "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
             "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
             "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
             "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
             "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
             "min_child_weight":trial.suggest_int("min_child_weight", 1, 10),
             "random_state": SEED, "verbosity": 0}
        return cross_val_score(xgb.XGBRegressor(**p), X, y_log, cv=kf, scoring="r2").mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=N_TRIALS)
    best = study.best_params
    best.update({"random_state": SEED, "verbosity": 0})
    return best, study.best_value


def tune_lightgbm(X, y_log, kf):
    def obj(trial):
        p = {"n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
             "max_depth":        trial.suggest_int("max_depth", 3, 12),
             "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
             "num_leaves":       trial.suggest_int("num_leaves", 20, 200),
             "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
             "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
             "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
             "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
             "random_state": SEED, "verbose": -1}
        return cross_val_score(lgb.LGBMRegressor(**p), X, y_log, cv=kf, scoring="r2").mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=N_TRIALS)
    best = study.best_params
    best.update({"random_state": SEED, "verbose": -1})
    return best, study.best_value


def run_modeling(X, y_log, y_raw):
    print("━"*60)
    print("  STEP 4 — MODELING")
    print("━"*60)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Baseline
    print("  [4a] Baseline (default params) ─────────────────────────")
    b_xgb = cross_val_score(xgb.XGBRegressor(n_estimators=200, random_state=SEED, verbosity=0),
                            X, y_log, cv=kf, scoring="r2").mean()
    b_lgb = cross_val_score(lgb.LGBMRegressor(n_estimators=200, random_state=SEED, verbose=-1),
                            X, y_log, cv=kf, scoring="r2").mean()
    print(f"  XGBoost  R² baseline : {b_xgb:.4f}")
    print(f"  LightGBM R² baseline : {b_lgb:.4f}")

    # Tune
    print(f"\n  [4b] Optuna tuning ({N_TRIALS} trials each) ─────────────────")
    print("  Tuning XGBoost ...")
    xgb_params, xgb_cv = tune_xgboost(X, y_log, kf)
    print(f"  XGBoost  best R² (Optuna) : {xgb_cv:.4f}")
    print("  Tuning LightGBM ...")
    lgb_params, lgb_cv = tune_lightgbm(X, y_log, kf)
    print(f"  LightGBM best R² (Optuna) : {lgb_cv:.4f}")

    # Final CV evaluation
    print("\n  [4c] Final cross-validation (IDR scale) ─────────────────")
    best_xgb = xgb.XGBRegressor(**xgb_params)
    best_lgb = lgb.LGBMRegressor(**lgb_params)
    res_xgb  = cv_metrics(best_xgb, X, y_log, y_raw, kf, "XGBoost (Tuned)")
    res_lgb  = cv_metrics(best_lgb, X, y_log, y_raw, kf, "LightGBM (Tuned)")
    results  = [res_xgb, res_lgb]

    for r in results:
        print(f"\n  📊 {r['label']}")
        print(f"     R²   : {r['r2']:.4f} ± {r['r2_std']:.4f}")
        print(f"     MAE  : Rp {r['mae']/1e6:.2f} juta")
        print(f"     MAPE : {r['mape']:.2f}%")

    winner_res   = max(results, key=lambda r: r["r2"])
    winner_model = best_xgb if winner_res["label"].startswith("XGB") else best_lgb
    winner_model.fit(X, y_log)
    print(f"\n  🏆 Winner: {winner_res['label']}")

    return winner_model, best_xgb, best_lgb, results, xgb_params, lgb_params, kf


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 ─ MODEL RESULT VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
def plot_model_results(winner_model, best_xgb, best_lgb,
                       X, y_log, y_raw, results, kf):
    print("\n━"*60)
    print("  STEP 5 — MODEL RESULT VISUALIZATIONS")
    print("━"*60)

    pred_log = winner_model.predict(X)
    pred_idr = np.expm1(pred_log)
    residuals = y_raw - pred_idr
    rel_err   = (pred_idr - y_raw) / y_raw * 100

    fig4, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig4.suptitle("Model Performance — XGBoost vs LightGBM (Optuna Tuned)",
                  fontsize=15, fontweight="bold", y=1.01)

    # [0,0] Model comparison bar
    ax = axes[0, 0]
    lbls  = [r["label"].replace(" (Tuned)","") for r in results]
    r2s   = [r["r2"] for r in results]
    maes  = [r["mae"]/1e6 for r in results]
    mapes = [r["mape"] for r in results]
    x     = np.arange(len(lbls))
    bar_c = [BLUE, GREEN]

    bars = ax.bar(x, r2s, color=bar_c, width=0.5, edgecolor="white", zorder=3)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=11)
    ax.set_ylabel("R² Score", fontsize=10)
    ax.set_title("Perbandingan Model (5-Fold CV)", fontweight="bold")
    ax.axhline(0.8, color=GRAY, linestyle="--", linewidth=1, alpha=0.6, zorder=2, label="R²=0.80")
    for bar, r2, mae, mape in zip(bars, r2s, maes, mapes):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"R²={r2:.4f}\nMAE=Rp{mae:.1f}jt\nMAPE={mape:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=1)

    # [0,1] Feature importance
    ax = axes[0, 1]
    fi = pd.Series(winner_model.feature_importances_, index=FEATURES).sort_values()
    top_fi = fi.tail(12)
    clrs_fi = [RED if v == fi.max() else ORANGE if v >= fi.quantile(0.75) else BLUE
               for v in top_fi.values]
    bars_fi = ax.barh(top_fi.index, top_fi.values, color=clrs_fi, edgecolor="white")
    for bar, val in zip(bars_fi, top_fi.values):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2, f"{val:.3f}",
                va="center", fontsize=8)
    ax.set_title(f"Feature Importance (Top 12)", fontweight="bold")
    ax.set_xlabel("Importance", fontsize=10)

    # [0,2] Actual vs Predicted
    ax = axes[0, 2]
    sc = ax.scatter(y_raw/1e6, pred_idr/1e6, alpha=0.3, s=14,
                    c=np.abs(rel_err), cmap="RdYlGn_r", vmin=0, vmax=50,
                    edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Error relatif (%)")
    lim_max = max(y_raw.max(), pred_idr.max())/1e6
    ax.plot([0, lim_max], [0, lim_max], "r--", linewidth=1.5, label="Ideal")
    ax.set_xlabel("Harga Aktual (Rp Juta)", fontsize=10)
    ax.set_ylabel("Harga Prediksi (Rp Juta)", fontsize=10)
    ax.set_title("Aktual vs Prediksi (train set)", fontweight="bold")
    r2_train = r2_score(y_raw, pred_idr)
    ax.text(0.05, 0.90, f"R²={r2_train:.4f}", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=BLUE)
    ax.legend(fontsize=9)

    # [1,0] Residual plot
    ax = axes[1, 0]
    ax.scatter(pred_idr/1e6, residuals/1e6, alpha=0.3, s=14,
               color=PURPLE, edgecolors="none")
    ax.axhline(0, color=RED, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Prediksi (Rp Juta)", fontsize=10)
    ax.set_ylabel("Residual (Aktual − Prediksi)", fontsize=10)
    ax.set_title("Residual Plot", fontweight="bold")

    # [1,1] Error distribution
    ax = axes[1, 1]
    ax.hist(rel_err, bins=60, color=TEAL, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.5)
    ax.axvline(rel_err.mean(), color=RED, linestyle="--", linewidth=2,
               label=f"Mean error = {rel_err.mean():.1f}%")
    ax.set_xlabel("Error Relatif (%)", fontsize=10)
    ax.set_ylabel("Frekuensi", fontsize=10)
    ax.set_title("Distribusi Error Relatif", fontweight="bold")
    ax.legend(fontsize=9)

    # [1,2] Prediction error by brand
    ax = axes[1, 2]
    df_err = pd.DataFrame({
        "merek":   X["merek"].map(lambda c: c),  # encoded — skip
        "abs_err": np.abs(rel_err)
    })
    # Use y_raw to reconstruct brand: need original df_clean
    # approximation: bin by predicted price range
    bins   = [0, 100, 200, 400, 700, 1500, np.inf]
    labels = ["<100jt","100-200jt","200-400jt","400-700jt","700jt-1.5M",">1.5M"]
    price_bin = pd.cut(y_raw/1e6, bins=bins, labels=labels)
    bin_mape = pd.DataFrame({"bin": price_bin, "mape": np.abs(rel_err)})\
                   .groupby("bin", observed=True)["mape"].median()
    clrs_bin = [GREEN if v < 20 else ORANGE if v < 30 else RED for v in bin_mape.values]
    ax.bar(bin_mape.index, bin_mape.values, color=clrs_bin, edgecolor="white", alpha=0.85)
    for i, v in enumerate(bin_mape.values):
        ax.text(i, v+0.3, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(20, color=GRAY, linestyle="--", linewidth=1, alpha=0.7, label="20% threshold")
    ax.set_xlabel("Segmen Harga", fontsize=10)
    ax.set_ylabel("Median MAPE (%)", fontsize=10)
    ax.set_title("Error per Segmen Harga", fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=15)

    fig4.tight_layout()
    fig4.savefig(OUT_DIR/"model_results.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig4)
    print("  ✅ model_results.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 ─ SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
def save_outputs(df_clean, results, xgb_params, lgb_params):
    # Clean CSV
    df_clean.to_csv(OUT_DIR/"olx_cleaned.csv", index=False)
    print("  ✅ olx_cleaned.csv")

    # Best params JSON
    report = {
        "dataset": {"raw_rows": raw_rows, "clean_rows": len(df_clean)},
        "metrics": {r["label"]: {"R2": round(r["r2"],4),
                                 "R2_std": round(r["r2_std"],4),
                                 "MAE_juta": round(r["mae"]/1e6,2),
                                 "MAPE_pct": round(r["mape"],2)} for r in results},
        "winner": max(results, key=lambda r: r["r2"])["label"],
        "xgb_params": xgb_params,
        "lgb_params":  lgb_params,
    }
    with open(OUT_DIR/"pipeline_report.json","w") as f:
        json.dump(report, f, indent=2)
    print("  ✅ pipeline_report.json\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  OLX MOBIL BEKAS — FULL ML PIPELINE")
    print("═"*60 + "\n")

    df_raw   = pd.read_csv(DATA_PATH)
    raw_rows = len(df_raw)

    # Step 1 — Clean
    df_clean, cleaning_log = clean_data(df_raw)

    # Step 2 — EDA
    plot_eda(df_raw, df_clean)

    # Step 3 — Feature Engineering
    df_clean = engineer_features(df_clean)

    # Step 4 — Modeling
    X, y_log, y_raw, le_dict = prepare_ml(df_clean)
    winner_model, best_xgb, best_lgb, results, xgb_params, lgb_params, kf = run_modeling(X, y_log, y_raw)

    # Step 5 — Visualize model results
    plot_model_results(winner_model, best_xgb, best_lgb, X, y_log, y_raw, results, kf)

    # Step 6 — Save
    print("\n━"*60)
    print("  STEP 6 — SAVING OUTPUTS")
    print("━"*60)
    save_outputs(df_clean, results, xgb_params, lgb_params)

    print("═"*60)
    print("  ✅ PIPELINE SELESAI!")
    print("═"*60)
