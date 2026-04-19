"""
data_processor.py
-----------------
Loads raw CSVs and produces clean aggregated summaries.
"""
import os
import glob
import argparse
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports")

def load_site_csv(filepath):
    return pd.read_csv(filepath, index_col="date", parse_dates=True)

def monthly_summary(df):
    cols = ["GHI_Wm2", "DNI_Wm2", "ClearSkyGHI_Wm2", "WindSpeed_10m_ms", "Temp_2m_C", "Precip_mm", "CloudAmount_pct"]
    available = [c for c in cols if c in df.columns]
    monthly = df[available].resample("M").mean()   # changed ME to M
    monthly.index = monthly.index.strftime("%Y-%m")
    return monthly

def annual_summary(df):
    result = {}
    if "GHI_Wm2" in df.columns:
        df["PSH"] = df["GHI_Wm2"] / 1000
        result["PSH_annual_mean"] = df["PSH"].resample("Y").mean()   # changed YE to Y
        result["GHI_annual_total_kWh_m2"] = df["GHI_Wm2"].resample("Y").sum() / 1000
    if "WindSpeed_10m_ms" in df.columns:
        result["Wind10m_annual_mean_ms"] = df["WindSpeed_10m_ms"].resample("Y").mean()
    if "Precip_mm" in df.columns:
        result["Precip_annual_total_mm"] = df["Precip_mm"].resample("Y").sum()
    if "Temp_2m_C" in df.columns:
        result["Temp_annual_mean_C"] = df["Temp_2m_C"].resample("Y").mean()
    out = pd.DataFrame(result)
    out.index = out.index.year
    out.index.name = "year"
    return out

def offgrid_score(annual):
    scores = {}
    notes = []
    if "PSH_annual_mean" in annual.columns:
        psh = annual["PSH_annual_mean"].mean()
        solar_score = min(40, round((psh / 7.0) * 40, 1))
        scores["solar"] = solar_score
        notes.append(f"Solar — avg PSH: {psh:.2f} h/day → {solar_score}/40 pts")
    if "Wind10m_annual_mean_ms" in annual.columns:
        ws = annual["Wind10m_annual_mean_ms"].mean()
        wind_score = min(30, round((ws / 8.0) * 30, 1))
        scores["wind"] = wind_score
        notes.append(f"Wind  — avg 10m speed: {ws:.2f} m/s → {wind_score}/30 pts")
    if "Precip_annual_total_mm" in annual.columns:
        rain = annual["Precip_annual_total_mm"].mean()
        if rain < 150:
            rain_score = 5
        elif rain < 600:
            rain_score = 15
        elif rain < 1200:
            rain_score = 10
        else:
            rain_score = 6
        scores["rain"] = rain_score
        notes.append(f"Rain  — avg annual: {rain:.0f} mm/yr → {rain_score}/15 pts")
    if "Temp_annual_mean_C" in annual.columns:
        temp = annual["Temp_annual_mean_C"].mean()
        if 15 <= temp <= 30:
            temp_score = 15
        elif 10 <= temp <= 35:
            temp_score = 10
        else:
            temp_score = 5
        scores["temp"] = temp_score
        notes.append(f"Temp  — avg annual: {temp:.1f}°C → {temp_score}/15 pts")
    total = sum(scores.values())
    return {
        "total_score": round(total, 1),
        "max_score": 100,
        "breakdown": scores,
        "notes": notes,
        "rating": "🟢 Excellent" if total >= 75 else "🟡 Good" if total >= 55 else "🟠 Moderate" if total >= 35 else "🔴 Poor"
    }

def process_all(raw_dir=RAW_DIR, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {raw_dir}. Run batch_fetcher.py first.")
        return
    print(f"\n Processing {len(csv_files)} site file(s) …\n")
    all_scores = []
    for filepath in csv_files:
        site_id = os.path.basename(filepath).split("_")[0] + "_" + os.path.basename(filepath).split("_")[1]
        print(f"{'─'*50}\n  {os.path.basename(filepath)}")
        df = load_site_csv(filepath)
        monthly = monthly_summary(df)
        annual = annual_summary(df)
        score = offgrid_score(annual)
        monthly.to_csv(os.path.join(output_dir, f"{site_id}_monthly.csv"))
        annual.to_csv(os.path.join(output_dir, f"{site_id}_annual.csv"))
        print(f"\n  Off-Grid Score: {score['total_score']}/100  {score['rating']}")
        for note in score["notes"]:
            print(f"    {note}")
        all_scores.append({"site_id": site_id, "total_score": score["total_score"], "rating": score["rating"], **score["breakdown"]})
    scores_df = pd.DataFrame(all_scores).set_index("site_id")
    scores_df.sort_values("total_score", ascending=False, inplace=True)
    scores_path = os.path.join(output_dir, "site_scores.csv")
    scores_df.to_csv(scores_path)
    print(f"\n{'═'*50}\n  SITE RANKING\n{'═'*50}\n{scores_df[['total_score','rating']].to_string()}\n\n  Scores saved → {scores_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=RAW_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args()
    process_all(raw_dir=args.input, output_dir=args.output)
