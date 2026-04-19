"""
nasa_fetcher.py
---------------
Core client for the NASA POWER API.
Pulls historical meteorological & solar data for a single GPS coordinate.

NASA POWER API Docs: https://power.larc.nasa.gov/docs/
No API key required. Data available from 1981 to ~2 months ago.
"""

import requests
import pandas as pd
import json
import time
import os
from datetime import datetime
from typing import Optional


# ── NASA POWER parameters we care about for off-grid site assessment ──────────
PARAMETERS = {
    # Solar
    "ALLSKY_SFC_SW_DWN":  "GHI_Wm2",          # Global Horizontal Irradiance
    "ALLSKY_SFC_SW_DNI":  "DNI_Wm2",          # Direct Normal Irradiance
    "ALLSKY_SFC_SW_DIFF": "DiffuseIrr_Wm2",   # Diffuse irradiance
    "CLRSKY_SFC_SW_DWN":  "ClearSkyGHI_Wm2",  # Clear-sky GHI
    # Wind
    "WS2M":               "WindSpeed_2m_ms",   # Wind speed at 2 m
    "WS10M":              "WindSpeed_10m_ms",  # Wind speed at 10 m
    "WD10M":              "WindDir_10m_deg",   # Wind direction at 10 m
    # Temperature & humidity
    "T2M":                "Temp_2m_C",         # Temperature at 2 m
    "T2M_MAX":            "TempMax_2m_C",
    "T2M_MIN":            "TempMin_2m_C",
    "RH2M":               "RelHumidity_pct",   # Relative humidity
    # Precipitation & cloud
    "PRECTOTCORR":        "Precip_mm",         # Precipitation (corrected)
    "CLOUD_AMT":          "CloudAmount_pct",   # Cloud amount
}

NASA_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_site_data(
    latitude: float,
    longitude: float,
    start_year: int = 2000,
    end_year: int = 2023,
    community: str = "RE",
    output_dir: str = "data/raw",
    site_id: str = "site",
    retries: int = 3,
    retry_delay: float = 5.0,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical daily data from NASA POWER for one GPS coordinate.

    Args:
        latitude:    Decimal degrees, e.g. -1.2921
        longitude:   Decimal degrees, e.g. 36.8219
        start_year:  First year of data (min 1981)
        end_year:    Last year of data
        community:   NASA POWER community code:
                       'RE' = Renewable Energy
                       'AG' = Agroclimatology
                       'SB' = Sustainable Buildings
        output_dir:  Where to save the raw CSV
        site_id:     Used for the output filename
        retries:     Number of retry attempts on failure
        retry_delay: Seconds between retries

    Returns:
        pandas DataFrame with one row per day, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    params_str = ",".join(PARAMETERS.keys())
    start_str  = f"{start_year}0101"
    end_str    = f"{end_year}1231"

    url_params = {
        "parameters": params_str,
        "community":  community,
        "longitude":  longitude,
        "latitude":   latitude,
        "start":      start_str,
        "end":        end_str,
        "format":     "JSON",
    }

    print(f"\n[{site_id}] Fetching NASA POWER data …")
    print(f"  Coordinates : ({latitude}, {longitude})")
    print(f"  Period      : {start_year}–{end_year}")
    print(f"  URL         : {NASA_BASE_URL}")

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(NASA_BASE_URL, params=url_params, timeout=120)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            print(f"  Attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                print(f"  Retrying in {retry_delay}s …")
                time.sleep(retry_delay)
            else:
                print("  All retries exhausted. Returning None.")
                return None

    data_json = resp.json()

    # ── Parse response ────────────────────────────────────────────────────────
    try:
        param_data = data_json["properties"]["parameter"]
    except KeyError:
        print("  ERROR: Unexpected API response structure.")
        print(json.dumps(data_json, indent=2)[:500])
        return None

    # Build a dict {date_str: {col: value}}
    rows = {}
    for nasa_key, col_name in PARAMETERS.items():
        if nasa_key not in param_data:
            print(f"  WARNING: Parameter {nasa_key} missing from response.")
            continue
        for date_str, value in param_data[nasa_key].items():
            if date_str not in rows:
                rows[date_str] = {}
            # NASA uses -999 for missing values
            rows[date_str][col_name] = None if value == -999.0 else value

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df.sort_index(inplace=True)

    # Add metadata columns
    df.insert(0, "site_id",   site_id)
    df.insert(1, "latitude",  latitude)
    df.insert(2, "longitude", longitude)

    # ── Save raw CSV ──────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, f"{site_id}_{start_year}_{end_year}.csv")
    df.to_csv(out_path)
    print(f"  ✓ Saved {len(df):,} rows → {out_path}")

    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly average summary of key metrics for a site DataFrame.
    Useful for quick sanity-checks and reporting.
    """
    numeric_cols = df.select_dtypes("number").columns.tolist()
    # Drop metadata columns from aggregation
    agg_cols = [c for c in numeric_cols if c not in ("latitude", "longitude")]

    monthly = (
        df[agg_cols]
        .resample("ME")   # Month-end frequency
        .agg(["mean", "min", "max", "std"])
    )
    monthly.columns = ["_".join(c) for c in monthly.columns]
    return monthly


# ── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with Nairobi coordinates
    df = fetch_site_data(
        latitude=-1.2921,
        longitude=36.8219,
        start_year=2015,
        end_year=2020,
        site_id="nairobi_test",
    )
    if df is not None:
        print("\n── Monthly summary (first 6 months) ──")
        print(summarise(df).head(6).to_string())
