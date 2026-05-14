"""
src/visualizer.py
-----------------
Generates a 3-panel dashboard for XGBoost model evaluation:
1. Feature Importance (Horizontal Bar)
2. SHAP Summary Plot
3. Training/Validation Loss (Log Loss/MAE)

Usage:
    python src/visualizer.py --site site_001
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports")
RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def load_data_for_site(site_id):
    """Load raw data and re-run engineering to get X for SHAP."""
    from xgboost_baseline import engineer_features, prepare_xy

    csv_path = os.path.join(RAW_DIR, f"{site_id}_2000_2023.csv")
    if not os.path.exists(csv_path):
        # Try generic name if exact years are unknown
        import glob
        matches = glob.glob(os.path.join(RAW_DIR, f"{site_id}*.csv"))
        if matches:
            csv_path = matches[0]
        else:
            raise FileNotFoundError(f"No data found for {site_id} in {RAW_DIR}")

    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    df_feat = engineer_features(df)
    X, y = prepare_xy(df_feat)
    return X, y

def plot_dashboard(site_id):
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. Load data
    print(f"Generating dashboard for {site_id}...")
    X, y = load_data_for_site(site_id)

    # 2. Load model
    model_path = os.path.join(MODEL_DIR, f"{site_id}_xgb_baseline.json")
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # 3. Load Importance
    imp_path = os.path.join(MODEL_DIR, f"{site_id}_feature_importance.csv")
    importance_df = pd.read_csv(imp_path).head(15)

    # 4. Load History
    hist_path = os.path.join(MODEL_DIR, f"{site_id}_eval_history.json")
    with open(hist_path, "r") as f:
        history = json.load(f)

    # Create Figure
    fig = plt.figure(figsize=(18, 6))

    # Panel 1: Feature Importance
    ax1 = fig.add_subplot(1, 3, 1)
    sns.barplot(data=importance_df, x="importance", y="feature", hue="feature", palette="viridis", legend=False, ax=ax1)
    ax1.set_title(f"XGBoost Feature Importance: {site_id}")
    ax1.set_xlabel("F-Score / Weight")

    # Panel 2: SHAP Summary (We use shap's own plotting but into our axis)
    ax2 = fig.add_subplot(1, 3, 2)
    explainer = shap.TreeExplainer(model)
    # Sampling for speed if needed, but here we use the full set if small
    X_sample = X.sample(min(1000, len(X)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    plt.sca(ax2) # set current axis
    shap.summary_plot(shap_values, X_sample, show=False, plot_size=None)
    ax2.set_title("SHAP Feature Impact (Relative Contribution)")

    # Panel 3: Training History
    ax3 = fig.add_subplot(1, 3, 3)

    # Determine which keys are present (usually validation_0 and validation_1)
    # validation_0: train, validation_1: val
    if 'validation_0' in history:
        m0 = list(history['validation_0'].keys())[0]
        ax3.plot(history['validation_0'][m0], label=f'Train {m0.upper()}')

    if 'validation_1' in history:
        m1 = list(history['validation_1'].keys())[0]
        ax3.plot(history['validation_1'][m1], label=f'Val {m1.upper()}', linestyle='--')

    ax3.set_title("Training vs Validation Loss")
    ax3.set_xlabel("Boosting Iterations")
    ax3.set_ylabel("Error")
    ax3.legend()

    plt.tight_layout()
    output_path = os.path.join(REPORT_DIR, f"{site_id}_dashboard.png")
    plt.savefig(output_path, dpi=150)
    print(f"Dashboard saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True, help="Site ID (e.g. esoit_001)")
    args = parser.parse_args()
    plot_dashboard(args.site)
