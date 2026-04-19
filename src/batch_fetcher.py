"""
batch_fetcher.py
----------------
Reads all sites from config/sites.json and fetches NASA POWER data for each.
Saves one CSV per site in data/raw/, and a combined master CSV in data/.

Usage:
    python src/batch_fetcher.py
    python src/batch_fetcher.py --config config/sites.json --output data/raw
"""

import json
import os
import argparse
import time
import pandas as pd
from tqdm import tqdm

from nasa_fetcher import fetch_site_data, summarise

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "sites.json")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MASTER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "all_sites_combined.csv")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def run_batch(config_path: str = CONFIG_PATH, output_dir: str = OUTPUT_DIR):
    config = load_config(config_path)
    sites  = config["sites"]
    fs     = config.get("fetch_settings", {})

    start_year = fs.get("start_year", 2000)
    end_year   = fs.get("end_year",   2023)
    community  = fs.get("community",  "RE")

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"  NASA POWER Batch Fetcher — {len(sites)} site(s)")
    print(f"  Period: {start_year}–{end_year}  |  Community: {community}")
    print(f"╚══════════════════════════════════════════════════╝")

    all_dfs    = []
    failed     = []
    log_lines  = []

    for site in tqdm(sites, desc="Sites", unit="site"):
        site_id = site["id"]
        name    = site.get("name", site_id)
        lat     = site["latitude"]
        lon     = site["longitude"]

        print(f"\n{'─'*50}")
        print(f"  Site : {name}  ({site_id})")
        print(f"  Notes: {site.get('notes', '—')}")

        df = fetch_site_data(
            latitude   = lat,
            longitude  = lon,
            start_year = start_year,
            end_year   = end_year,
            community  = community,
            output_dir = output_dir,
            site_id    = site_id,
        )

        if df is not None:
            df["site_name"] = name
            all_dfs.append(df)
            log_lines.append(f"✓  {site_id} ({name}) — {len(df):,} rows")
        else:
            failed.append(site_id)
            log_lines.append(f"✗  {site_id} ({name}) — FAILED")

        # Be polite to the API between requests
        time.sleep(2)

    # ── Combine all sites ─────────────────────────────────────────────────────
    if all_dfs:
        master = pd.concat(all_dfs)
        os.makedirs(os.path.dirname(MASTER_PATH), exist_ok=True)
        master.to_csv(MASTER_PATH)
        print(f"\n✓ Master file saved → {MASTER_PATH}  ({len(master):,} total rows)")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n╔══ BATCH SUMMARY ═══════════════════════════════╗")
    for line in log_lines:
        print(f"  {line}")
    if failed:
        print(f"\n  ⚠ Failed sites: {', '.join(failed)}")
    print("╚════════════════════════════════════════════════╝")

    return all_dfs, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch NASA POWER fetcher")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to sites.json")
    parser.add_argument("--output", default=OUTPUT_DIR,  help="Output directory for CSVs")
    args = parser.parse_args()

    run_batch(config_path=args.config, output_dir=args.output)
