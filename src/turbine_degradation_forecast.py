"""
turbine_degradation_forecast.py
-------------------------------
Turbine Blade degradation forecast module.
Predicts a Soiling/Degradation Risk Index for wind turbine blades using
NASA POWER meteorological data and MERRA-2 aerosol reanalysis.

Key features:
  - Wind-Dust Flux: WS * AOD (proxy for particle impingement rate)
  - Impact Kinetic Energy: WS^3 * AOD (proxy for erosion potential)
  - Wash-off Factor: Rolling precipitation (natural cleaning)
  - Relative Humidity Effects: High RH can increase particle adhesion
"""

import os
import glob
import argparse
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MERRA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "merra2")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "models_turbine")

# ── 1. Feature engineering ────────────────────────────────────────────────────

def engineer_turbine_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Specific feature engineering for Turbine Blade degradation.
    """
    f = df.copy()

    # A. Wind-Dust Flux (Impingement Rate)
    # If MERRA-2 AOD is not available, we use a proxy or just WS
    aod_col = "AOD_total" if "AOD_total" in f.columns else None
    ws_col = "WindSpeed_50m_ms" if "WindSpeed_50m_ms" in f.columns else "WindSpeed_10m_ms"

    if aod_col and ws_col in f.columns:
        f["wind_dust_flux"] = f[ws_col] * f[aod_col]
        # Impact Kinetic Energy (proportional to v^2 or v^3)
        f["impact_energy_proxy"] = (f[ws_col]**3) * f[aod_col]

    # B. Wash-off factors (Precipitation cleaning)
    if "Precip_mm" in f.columns:
        f["precip_7d_accum"] = f["Precip_mm"].rolling(window=7, min_periods=1).sum()
        f["days_since_rain"] = (f["Precip_mm"] == 0).astype(int).groupby((f["Precip_mm"] > 0.1).cumsum()).cumcount()

    # C. Adhesion conditions (RH and Temp)
    if "RelHumidity_pct" in f.columns:
        f["high_rh_flag"] = (f["RelHumidity_pct"] > 80).astype(int)

    # D. Degradation Risk Index (Proxy Target)
    # In the absence of ground truth, we define a Risk Index
    # based on cumulative flux minus wash-off

    # Fallback if no AOD: use WindSpeed as a proxy for flux
    if aod_col and aod_col in f.columns:
        flux_proxy = f["wind_dust_flux"]
    else:
        # If no AOD, we assume a constant background aerosol for the proxy
        flux_proxy = f[ws_col] * 0.2

    # Simplified linear accumulation model
    flux = flux_proxy.fillna(flux_proxy.median())
    cleaning = f["Precip_mm"].fillna(0) * 0.5 # 1mm rain removes 0.5 units of flux (arbitrary for proxy)

    risk_accum = [0]
    for i in range(1, len(f)):
        new_val = max(0, risk_accum[-1] + flux.iloc[i] - cleaning.iloc[i])
        risk_accum.append(new_val)
    f["soiling_risk_index"] = risk_accum

    # E. Circular seasonality
    f["doy"] = f.index.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * f["doy"] / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * f["doy"] / 365.25)

    return f

def prepare_xy(df: pd.DataFrame, target: str = "soiling_risk_index"):
    # Drop columns that are not features
    drop_cols = ["site_id", "site_name", "latitude", "longitude", "doy"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if target in X.columns:
        y = X[target]
        X = X.drop(columns=[target])
    else:
        # Fallback if we don't have the generated target
        return None, None

    # Fill NaNs
    X = X.fillna(X.median(numeric_only=True))
    return X, y

# ── 2. Model ──────────────────────────────────────────────────────────────────

def train_turbine_model(site_id: str, df: pd.DataFrame):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\n--- Training Turbine Blade Degradation Forecast for {site_id} ---")

    df_feat = engineer_turbine_features(df)
    X, y = prepare_xy(df_feat)

    if X is None:
        print(f"Skipping {site_id}: missing features/target")
        return

    tscv = TimeSeriesSplit(n_splits=5)
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)

    # Simple walk-forward validation
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        print(f"  Fold {fold} MAE: {mae:.4f}")

    # Final fit
    model.fit(X, y)

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{site_id}_turbine_degradation_xgb.json")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Feature Importance
    importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    print("\nTop Features:")
    print(importance.head(5))

# ── 3. Main ───────────────────────────────────────────────────────────────────

def main():
    # Load sites and merge data
    # For now, let's look for combined data or individual CSVs
    raw_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

    for pf in raw_files:
        site_id = os.path.basename(pf).split("_")[0] + "_" + os.path.basename(pf).split("_")[1]
        pdf = pd.read_csv(pf, index_col="date", parse_dates=True)

        # Look for matching MERRA-2
        m_file = os.path.join(MERRA_DIR, f"{site_id}_merra2_aerosol.csv")
        if os.path.exists(m_file):
            mdf = pd.read_csv(m_file, index_col="date", parse_dates=True)
            df = pdf.join(mdf, how="left")
            print(f"Integrated MERRA-2 for {site_id}")
        else:
            df = pdf
            print(f"Warning: MERRA-2 data missing for {site_id}, using POWER-only proxies.")

        train_turbine_model(site_id, df)

if __name__ == "__main__":
    main()
