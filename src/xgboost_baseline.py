"""
xgboost_baseline.py
--------------------
Train an XGBoost regression model to predict GHI (Global Horizontal Irradiance)
using lagged and engineered features from NASA POWER data.

This is the Week 2 baseline. It runs entirely on data you already have from
batch_fetcher.py — no MERRA-2 registration required.

Usage:
    python src/xgboost_baseline.py
    python src/xgboost_baseline.py --site site_001 --target GHI_Wm2

Output:
    outputs/models/{site_id}_xgb_baseline.json   — trained model
    outputs/models/{site_id}_feature_importance.csv
    outputs/models/{site_id}_cv_results.csv       — cross-val metrics
"""

import os
import glob
import argparse
import json
import warnings
import pandas as pd
import numpy as np

try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit, cross_validate
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise SystemExit(
        "Missing dependencies. Run:\n"
        "  pip install xgboost scikit-learn pandas numpy"
    )

warnings.filterwarnings("ignore", category=UserWarning)

RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")

TARGET_COL = "GHI_Wm2"   # What we're predicting (can override via CLI)


# ── 1. Feature engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, target: str = TARGET_COL) -> pd.DataFrame:
    """
    Build the feature matrix from a raw POWER DataFrame.

    Feature groups:
      A) Calendar / astronomical  — day-of-year encodes sun path without leaking future
      B) Aerosol proxy            — GHI ÷ ClearSkyGHI captures aerosol + cloud attenuation
      C) Meteorological           — temperature, humidity, wind, precipitation, cloud
      D) Lagged target            — yesterday's and last-week's GHI (autoregressive signal)
      E) Rolling statistics       — 7-day & 30-day mean/std of target and key meteo vars
    """
    f = df.copy()

    # ── A. Calendar ──────────────────────────────────────────────────────────
    f["doy"]       = f.index.dayofyear
    f["doy_sin"]   = np.sin(2 * np.pi * f["doy"] / 365.25)   # circular encoding
    f["doy_cos"]   = np.cos(2 * np.pi * f["doy"] / 365.25)
    f["month"]     = f.index.month
    f["month_sin"] = np.sin(2 * np.pi * f["month"] / 12)
    f["month_cos"] = np.cos(2 * np.pi * f["month"] / 12)
    f["year"]      = f.index.year

    # ── B. Aerosol proxy ─────────────────────────────────────────────────────
    if "ClearSkyGHI_Wm2" in f.columns and target in f.columns:
        # Clearness index: 0 (fully cloudy/hazy) to ~1 (perfect sky)
        f["clearness_idx"] = f[target] / f["ClearSkyGHI_Wm2"].replace(0, np.nan)
        f["clearness_idx"] = f["clearness_idx"].clip(0, 1.2)   # physical bounds

        # Cloud+aerosol attenuation (complement)
        f["attenuation"] = 1.0 - f["clearness_idx"].fillna(0)

    # ── C. Meteorological ────────────────────────────────────────────────────
    meteo_cols = [
        "Temp_2m_C", "TempMax_2m_C", "TempMin_2m_C",
        "RelHumidity_pct", "Precip_mm",
        "WindSpeed_2m_ms", "WindSpeed_10m_ms", "WindDir_10m_deg",
        "CloudAmount_pct", "DNI_Wm2", "DiffuseIrr_Wm2",
    ]
    for col in meteo_cols:
        if col in f.columns:
            # Diurnal temperature range
            if col == "TempMax_2m_C" and "TempMin_2m_C" in f.columns:
                f["temp_range_C"] = f["TempMax_2m_C"] - f["TempMin_2m_C"]

            # Wind direction: decompose into u/v components
            if col == "WindDir_10m_deg" and "WindSpeed_10m_ms" in f.columns:
                rad = np.deg2rad(f["WindDir_10m_deg"])
                f["wind_u"] = f["WindSpeed_10m_ms"] * np.sin(rad)
                f["wind_v"] = f["WindSpeed_10m_ms"] * np.cos(rad)

    # ── D. Lagged target (autoregressive) ────────────────────────────────────
    if target in f.columns:
        f[f"{target}_lag1"]  = f[target].shift(1)    # yesterday
        f[f"{target}_lag2"]  = f[target].shift(2)
        f[f"{target}_lag7"]  = f[target].shift(7)    # same day last week
        f[f"{target}_lag14"] = f[target].shift(14)

    # ── E. Rolling statistics ────────────────────────────────────────────────
    roll_cols = [target, "clearness_idx", "CloudAmount_pct", "Temp_2m_C"]
    for col in roll_cols:
        if col not in f.columns:
            continue
        for window in [7, 30]:
            f[f"{col}_roll{window}mean"] = (
                f[col].shift(1).rolling(window, min_periods=3).mean()
            )
            f[f"{col}_roll{window}std"] = (
                f[col].shift(1).rolling(window, min_periods=3).std()
            )

    return f


def prepare_xy(df: pd.DataFrame, target: str = TARGET_COL):
    """
    Split the engineered DataFrame into X (features) and y (target).
    Drops metadata columns and rows with NaN in target.
    """
    drop_always = ["site_id", "site_name", "latitude", "longitude", "doy", "month"]
    feature_df  = df.drop(columns=[c for c in drop_always if c in df.columns])

    # Remove leaky columns (current-day irradiance components if predicting GHI)
    leaky = []
    if target == "GHI_Wm2":
        leaky = [c for c in feature_df.columns
                 if c in ("DNI_Wm2", "DiffuseIrr_Wm2", "ClearSkyGHI_Wm2")
                 and c != target]
    feature_df = feature_df.drop(columns=leaky)

    # Drop rows where target is missing
    mask   = feature_df[target].notna()
    y      = feature_df.loc[mask, target]
    X      = feature_df.loc[mask].drop(columns=[target])

    # Fill remaining NaNs with column medians (safe for tree models)
    X = X.fillna(X.median(numeric_only=True))

    return X, y


# ── 2. Model definition ───────────────────────────────────────────────────────

def build_model() -> xgb.XGBRegressor:
    """
    Conservative XGBoost hyperparameters for a tabular daily time-series baseline.
    These are defensible starting values — tune via Optuna in Week 3.
    """
    return xgb.XGBRegressor(
        n_estimators     = 600,
        learning_rate    = 0.05,
        max_depth        = 5,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        reg_alpha        = 0.1,    # L1 regularisation
        reg_lambda       = 1.0,    # L2 regularisation
        objective        = "reg:squarederror",
        eval_metric      = "mae",
        early_stopping_rounds = 40,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
    )


# ── 3. Cross-validation ───────────────────────────────────────────────────────

def time_series_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """
    Walk-forward cross-validation respecting temporal order.
    Uses sklearn's TimeSeriesSplit so no future data leaks into training folds.
    """
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    model  = build_model()

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit with early stopping on a small held-out eval set within the fold
        eval_frac   = max(0.05, min(0.15, 200 / len(X_tr)))
        split_point = int(len(X_tr) * (1 - eval_frac))
        X_fit, X_eval_f = X_tr.iloc[:split_point], X_tr.iloc[split_point:]
        y_fit, y_eval_f = y_tr.iloc[:split_point], y_tr.iloc[split_point:]

        model.fit(
            X_fit, y_fit,
            eval_set=[(X_eval_f, y_eval_f)],
            verbose=False,
        )

        preds = model.predict(X_val)
        mae   = mean_absolute_error(y_val, preds)
        rmse  = np.sqrt(mean_squared_error(y_val, preds))
        r2    = r2_score(y_val, preds)
        mape  = np.mean(np.abs((y_val - preds) / (y_val + 1e-6))) * 100

        fold_results.append({
            "fold": fold,
            "n_train": len(y_tr),
            "n_val":   len(y_val),
            "MAE":     round(mae,  2),
            "RMSE":    round(rmse, 2),
            "R2":      round(r2,   4),
            "MAPE_pct": round(mape, 2),
        })
        print(f"    Fold {fold}: MAE={mae:.1f} W/m²  RMSE={rmse:.1f}  R²={r2:.3f}  MAPE={mape:.1f}%")

    cv_df = pd.DataFrame(fold_results)
    summary = {
        "mean_MAE":      round(cv_df["MAE"].mean(), 2),
        "mean_RMSE":     round(cv_df["RMSE"].mean(), 2),
        "mean_R2":       round(cv_df["R2"].mean(), 4),
        "mean_MAPE_pct": round(cv_df["MAPE_pct"].mean(), 2),
    }
    return {"folds": cv_df, "summary": summary}


# ── 4. Final fit & save ───────────────────────────────────────────────────────

def train_and_save(
    df: pd.DataFrame,
    site_id: str,
    target: str = TARGET_COL,
    output_dir: str = MODEL_DIR,
) -> dict:
    """Full pipeline: engineer → split → CV → final fit → save."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'═'*56}")
    print(f"  Site: {site_id}  |  Target: {target}")
    print(f"{'═'*56}")

    # ── Feature engineering ───────────────────────────────────────────────
    df_feat = engineer_features(df, target=target)
    X, y    = prepare_xy(df_feat, target=target)

    print(f"  Samples     : {len(y):,}")
    print(f"  Features    : {X.shape[1]}")
    print(f"  Date range  : {y.index[0].date()} → {y.index[-1].date()}")
    print(f"  Target stats: mean={y.mean():.1f}  std={y.std():.1f}  max={y.max():.1f} W/m²")

    # ── Walk-forward CV ───────────────────────────────────────────────────
    print(f"\n  Walk-forward cross-validation (5 folds):")
    cv = time_series_cv(X, y, n_splits=5)
    s  = cv["summary"]
    print(f"\n  CV Summary → MAE={s['mean_MAE']} W/m²  RMSE={s['mean_RMSE']}  "
          f"R²={s['mean_R2']}  MAPE={s['mean_MAPE_pct']}%")

    # ── Save CV results ───────────────────────────────────────────────────
    cv_path = os.path.join(output_dir, f"{site_id}_cv_results.csv")
    cv["folds"].to_csv(cv_path, index=False)

    # ── Final model on all data ───────────────────────────────────────────
    print(f"\n  Fitting final model on full dataset …")
    final_model = build_model()

    # Use last 10% as eval set for early stopping
    split_point = int(len(X) * 0.9)
    final_model.fit(
        X.iloc[:split_point], y.iloc[:split_point],
        eval_set=[(X.iloc[split_point:], y.iloc[split_point:])],
        verbose=False,
    )

    # ── Feature importance ────────────────────────────────────────────────
    importance = pd.DataFrame({
        "feature":    X.columns,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1

    imp_path = os.path.join(output_dir, f"{site_id}_feature_importance.csv")
    importance.to_csv(imp_path, index=False)

    print(f"\n  Top 10 features:")
    for _, row in importance.head(10).iterrows():
        bar = "█" * int(row["importance"] * 300)
        print(f"    {int(row['rank']):2d}. {row['feature']:<35s} {bar}")

    # ── Save model ────────────────────────────────────────────────────────
    model_path = os.path.join(output_dir, f"{site_id}_xgb_baseline.json")
    final_model.save_model(model_path)

    # ── Save metadata ─────────────────────────────────────────────────────
    meta = {
        "site_id":        site_id,
        "target":         target,
        "n_samples":      int(len(y)),
        "n_features":     int(X.shape[1]),
        "features":       list(X.columns),
        "date_range":     [str(y.index[0].date()), str(y.index[-1].date())],
        "cv_summary":     s,
        "model_file":     model_path,
        "xgboost_version": xgb.__version__,
    }
    meta_path = os.path.join(output_dir, f"{site_id}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Files saved:")
    print(f"    Model      → {model_path}")
    print(f"    Features   → {imp_path}")
    print(f"    CV results → {cv_path}")
    print(f"    Metadata   → {meta_path}")

    return meta


# ── 5. Entry point ────────────────────────────────────────────────────────────

def run(site_filter: str = None, target: str = TARGET_COL):
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}. Run batch_fetcher.py first."
        )

    all_meta = []
    for filepath in sorted(csv_files):
        fname   = os.path.basename(filepath)
        site_id = "_".join(fname.replace(".csv", "").split("_")[:2])

        if site_filter and site_filter not in site_id:
            continue

        df   = pd.read_csv(filepath, index_col="date", parse_dates=True)
        meta = train_and_save(df, site_id=site_id, target=target)
        all_meta.append(meta)

    # Summary table
    if len(all_meta) > 1:
        print(f"\n{'═'*56}")
        print("  ALL SITES — CV SUMMARY")
        print(f"{'═'*56}")
        for m in all_meta:
            s = m["cv_summary"]
            print(f"  {m['site_id']:<20s}  MAE={s['mean_MAE']} W/m²  R²={s['mean_R2']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost GHI baseline")
    parser.add_argument("--site",   default=None,       help="Filter to one site_id prefix")
    parser.add_argument("--target", default=TARGET_COL, help="Target column to predict")
    args = parser.parse_args()
    run(site_filter=args.site, target=args.target)
