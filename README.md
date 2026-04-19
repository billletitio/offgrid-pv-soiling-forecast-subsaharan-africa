# offgrid-pv-soiling-forecast-subsaharan-africa

> **Developed and validated in Kenya; designed for broader applicability across sub-Saharan Africa.**

Soiling forecast toolkit for off-grid PV systems — fetches NASA POWER & MERRA-2 aerosol reanalysis data, engineers climate features, and trains an XGBoost baseline model. Developed and validated in Kenya; designed for broader applicability across sub-Saharan Africa.

---

## Research Objective

Soiling — the accumulation of dust, aerosols, and particulate matter on photovoltaic panels — is one of the least-monitored sources of energy yield loss at off-grid community solar sites across sub-Saharan Africa. Physical soiling measurements require sensors and maintenance infrastructure that are rarely available in remote, data-scarce deployments.

**This study investigates whether aerosol optical depth (AOD) and meteorological reanalysis data can serve as reliable proxies for PV soiling rate forecasting in environments where no ground-truth soiling measurements exist.**

**Research question:** Can an XGBoost model trained on NASA POWER and MERRA-2 reanalysis features predict soiling-induced clearness index degradation with sufficient accuracy to support maintenance scheduling decisions at off-grid community solar sites in semi-arid Kenya?

**Hypothesis:** Sites with higher aerosol optical depth (AOD > 0.3), lower precipitation (< 400 mm/yr), and extended dry-season length will exhibit systematically higher soiling rates, and these signals are learnable from reanalysis data alone without physical sensor measurements.

**Target performance metrics:**

| Metric | Target | Rationale |
|---|---|---|
| RMSE (clearness index) | ≤ 0.12 | Operationally meaningful degradation threshold |
| MAE | ≤ 0.09 | Robust to outlier soiling events |
| R² | ≥ 0.78 | Minimum explanatory power for scheduling decisions |
| MERRA-2 AOD improvement | 8–15% RMSE reduction over POWER-only baseline | Validates aerosol features as soiling proxies |

---

## Research Context

In sub-Saharan Africa, where Saharan dust transport, biomass burning aerosols, and long dry seasons interact, soiling losses can reach **2–8% annually** at unattended rural sites — equivalent to weeks of lost energy output per year for a community that depends entirely on that system. Despite this, soiling remains systematically underrepresented in off-grid solar performance models due to the absence of ground-truth data.

This project addresses that gap by building a forecasting framework grounded in:

- **NASA POWER** historical meteorological and solar irradiance data (1981–present, no API key required)
- **MERRA-2** aerosol reanalysis (AOD, dust extinction, Angstrom exponent) from NASA GES DISC
- **XGBoost** regression trained on clearness index, aerosol optical depth, and engineered seasonal features

**Soiling is operationalized** as the ratio of observed GHI to clear-sky GHI (the clearness index, CI = GHI / ClearSkyGHI). Systematic depression of CI under low-cloud, low-rainfall, high-AOD conditions is attributed to soiling accumulation. This proxy approach is validated in the literature for data-scarce semi-arid contexts.

Field validation sites are located in **Kenya** — Kabati/Kilinditui Village (Kitui County) and Esoit Village (Kajiado County) — both active community solar installations supported by **SACDEP** (Sustainable Agriculture Community Development Programme) and partners. The methodology is designed to generalise to any site across sub-Saharan Africa where NASA reanalysis data is available.

---

## Methodology

```
NASA POWER API          MERRA-2 GES DISC
(solar, met, precip)    (AOD, dust, aerosols)
        │                       │
        └──────────┬────────────┘
                   ▼
          Feature Engineering
          ┌─────────────────────────────┐
          │ Clearness Index (CI)        │  ← soiling proxy / target
          │ AOD_total, AOD_dust         │  ← aerosol loading
          │ Precip_mm (rolling 7/30d)   │  ← natural cleaning signal
          │ Temp, RH, Wind              │  ← panel microclimate
          │ Day-of-year, dry season flag│  ← seasonality
          └─────────────────────────────┘
                   │
                   ▼
          XGBoost Regressor
          ┌─────────────────────────────┐
          │ Train: 2000–2018 (both sites)│
          │ Validation: 2019–2020       │
          │ Test: 2021–2023 (held out)  │
          │ Cross-site generalisation   │
          │ test: train Kitui → Kajiado │
          └─────────────────────────────┘
                   │
                   ▼
          Output: daily CI forecast
          → soiling loss estimate (%)
          → maintenance trigger flag
```

**Key modelling decisions:**
- Clearness index (CI) is used as both the soiling proxy and the prediction target, following established methodology for data-scarce environments
- An 80/20 temporal train/test split is used (not random) to prevent data leakage across time
- A cross-site generalisation test (train on Kitui, test on Kajiado) evaluates transferability — the core claim of sub-Saharan applicability
- SHAP values are computed post-training to interpret which features drive soiling predictions at each site

---

## What It Does

| Module | Description |
|---|---|
| `nasa_fetcher.py` | Core NASA POWER API client — pulls daily solar, met, and precipitation data for any GPS coordinate (1981–present) |
| `batch_fetcher.py` | Batch runner — fetches all sites in `config/sites.json` and assembles a combined master CSV |
| `data_processor.py` | Computes monthly/annual summaries, peak solar hours (PSH), and an off-grid suitability score per site |
| `merra2_fetcher.py` | NASA GES DISC OPeNDAP client — fetches MERRA-2 aerosol reanalysis (AOD, dust, black carbon, Angstrom exponent) |
| `xgboost_baseline.py` | Trains and evaluates XGBoost soiling forecast model on merged POWER + MERRA-2 features; outputs SHAP plots and metrics |

---

## Project Structure

```
offgrid-pv-soiling-forecast-subsaharan-africa/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── sites.json              # Target GPS coordinates and fetch settings
├── src/
│   ├── nasa_fetcher.py         # Core NASA POWER API client
│   ├── batch_fetcher.py        # Multi-site batch fetcher
│   ├── data_processor.py       # Clean, aggregate, and score sites
│   ├── merra2_fetcher.py       # MERRA-2 aerosol reanalysis client
│   └── xgboost_baseline.py     # XGBoost soiling forecast baseline
├── data/
│   ├── raw/                    # Per-site NASA POWER CSVs (generated)
│   ├── merra2/                 # MERRA-2 aerosol CSVs (generated)
│   └── all_sites_combined.csv  # Master combined dataset (generated)
└── outputs/
    └── reports/                # Monthly/annual summaries, site scores, SHAP plots
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/billletitio/offgrid-pv-soiling-forecast-subsaharan-africa.git
cd offgrid-pv-soiling-forecast-subsaharan-africa

# 2. Install dependencies
pip install -r requirements.txt

# 3. Review site coordinates in config/sites.json

# 4. Fetch NASA POWER data for all sites
python src/batch_fetcher.py

# 5. Process and score sites
python src/data_processor.py

# 6. Train the XGBoost baseline model
python src/xgboost_baseline.py
```

---

## Validation Sites (Kenya)

Both sites are active community solar installations supported by SACDEP and partners.

| Site ID | Name | County | Coordinates | Elevation | Status |
|---|---|---|---|---|---|
| site_001 | Kabati – Kilinditui Village | Kitui | -1.2340, 37.9157 | ~1,226 m | Primary fieldwork site |
| site_002 | Esoit Village – Olkejuado Area | Kajiado | -1.8350, 36.7950 | ~1,580 m | Phase 2 fieldwork site* |

*Esoit coordinates are approximate — GPS reading to be confirmed on arrival and updated in `sites.json`.

These two sites represent contrasting semi-arid environments within Kenya, which strengthens the generalisability argument:

- **Kitui (Kabati)** — inland semi-arid, elevated dust loading from northeasterly winds, longer dry seasons (avg. rainfall ~500 mm/yr), primary soiling driver: mineral dust
- **Kajiado (Esoit)** — southern Rift Valley, Maasai plains, elevated terrain (~1,580 m), strong seasonal irradiance variation, primary soiling driver: fine particulate aerosols

The cross-site generalisation test — training on Kitui data and predicting on Kajiado — is the central transferability experiment of this study.

---

## Phase 2: MERRA-2 Aerosol Integration

The Phase 1 XGBoost baseline uses NASA POWER's clearness index as an implicit aerosol proxy. Phase 2 integrates explicit MERRA-2 AOD features, expected to improve RMSE by **8–15%** by resolving aerosol variance that cloud cover alone cannot explain.

**Setup:**
1. Register at [NASA Earthdata](https://urs.earthdata.nasa.gov/)
2. Approve GES DISC: [disc.gsfc.nasa.gov](https://disc.gsfc.nasa.gov/earthdata-login)
3. Create `~/.netrc`:
   ```
   machine urs.earthdata.nasa.gov
   login YOUR_USERNAME
   password YOUR_PASSWORD
   ```
4. Uncomment `requests-netrc` in `requirements.txt` and reinstall
5. Run: `python src/merra2_fetcher.py`

---

## Data Sources

| Source | Variables | Temporal Resolution | Spatial Resolution | Access |
|---|---|---|---|---|
| [NASA POWER](https://power.larc.nasa.gov/) | GHI, DNI, diffuse irradiance, clear-sky GHI, temperature (min/max/avg), wind speed/direction, relative humidity, precipitation, cloud cover | Daily, 1981–present | 0.5° × 0.625° | Free, no key required |
| [NASA MERRA-2](https://disc.gsfc.nasa.gov/datasets/M2T1NXAER_5.12.4) | Total AOD, dust extinction (DUEXTTAU), black carbon, organic carbon, sea salt, Angstrom exponent | Hourly → daily mean | 0.5° × 0.625° | Free, NASA Earthdata account required |

---

## Requirements

```
requests>=2.31.0
pandas>=2.0.0
numpy>=1.24.0
python-dateutil>=2.8.2
tqdm>=4.65.0
xgboost>=2.0.0
scikit-learn>=1.3.0
# Phase 2 — uncomment after NASA Earthdata registration:
# requests-netrc>=0.1.2
```

---

## Community Development Context

This research is conducted in partnership with **SACDEP (Sustainable Agriculture Community Development Programme)** and affiliated partners. Both field sites — Kabati (Kitui) and Esoit (Kajiado) — are active off-grid community solar installations serving rural communities with no grid access. The soiling forecast work directly supports long-term energy yield planning and preventive maintenance scheduling for these systems, translating research outputs into operational decisions that benefit end users.

---

## Current Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | NASA POWER data pipeline + XGBoost baseline | Complete |
| Phase 2 | MERRA-2 aerosol feature integration | In progress |
| Phase 3 | Cross-site generalisation test (Kitui → Kajiado) | Planned |
| Phase 4 | Maintenance scheduling decision framework | Planned |

---

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@misc{letitio2025offgrid,
  author    = {Letitio, Bill Saning'o},
  title     = {Off-Grid PV Soiling Forecast for Sub-Saharan Africa},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/billletitio/offgrid-pv-soiling-forecast-subsaharan-africa}
}
```

---

## License

MIT License — see `LICENSE` for details.
