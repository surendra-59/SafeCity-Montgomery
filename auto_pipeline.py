"""
auto_pipeline.py
================
Automated data-fetch + full retraining pipeline for SafeCity Montgomery.

What it does (in order):
  Step 0 — Fetch fresh 311 + Code Violations data from ArcGIS REST APIs
  Step 1 — Clean 311 requests          (mirrors Cell 1 of development.ipynb)
  Step 2 — Clean Code Violations        (mirrors Cell 2 of development.ipynb)
  Step 3 — Clean Weather Sirens         (mirrors Cell 3 of development.ipynb)
  Step 4 — Build feature matrix         (mirrors Cell 4 of development.ipynb)
  Step 5 — Train model + score + save   (mirrors Cell 5 of development.ipynb)

Run:
    uv run python auto_pipeline.py

Output files (identical to what development.ipynb produces):
    Dataset/311_requests_full.csv              ← refreshed raw data
    Dataset/montgomery_code_violations_full.csv ← refreshed raw data
    Dataset/311_requests_cleaned.csv
    Dataset/violations_cleaned.csv
    Dataset/sirens_cleaned.csv
    Dataset/feature_matrix.csv
    Dataset/risk_scores.csv
    Dataset/feature_importance.csv
    nuisance_predictor.pkl
    model_evaluation.png
"""

import os
import sys
import time
import warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PATHS & API ENDPOINTS
# ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

API_311 = (
    "https://gis.montgomeryal.gov/server/rest/services/"
    "HostedDatasets/Received_311_Service_Request/MapServer/0/query"
)
API_VIO = (
    "https://gis.montgomeryal.gov/server/rest/services/"
    "HostedDatasets/Code_Violations/MapServer/0/query"
)

SEP = "=" * 60


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════
# STEP 0 — FETCH DATA FROM ARCGIS REST APIS
# ═══════════════════════════════════════════════════════════════

def _paginate_api(url: str, where_clause: str, label: str) -> list:
    """
    Core paginator: fetches all pages from an ArcGIS FeatureService
    matching the given where clause. Returns a list of attribute dicts.
    """
    params = {
        "where":             where_clause,
        "outFields":         "*",
        "f":                 "json",
        "resultRecordCount": 2000,
        "resultOffset":      0,
        "orderByFields":     "",   # no ordering needed; we filter by date
    }

    records = []
    page    = 0

    while page < 500:
        params["resultOffset"] = page * 2000
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log(f"  ⚠  API error (page {page}): {e}")
            break

        features = data.get("features", [])
        if not features:
            break

        for feat in features:
            row  = feat.get("attributes", {})
            geom = feat.get("geometry")
            if geom:
                row["_api_x"] = geom.get("x")
                row["_api_y"] = geom.get("y")
            records.append(row)

        log(f"  Page {page + 1}: +{len(features)} rows  (running total: {len(records)})")
        if len(features) < 2000:
            break
        page += 1
        time.sleep(0.05)

    return records


def _get_max_date_311(out_path: str) -> str:
    """
    Reads existing 311 CSV and returns an ArcGIS WHERE clause that
    filters for records AFTER the latest Create_Date already stored.
    Create_Date is stored as Unix milliseconds in the raw CSV.
    Returns '1=1' (fetch all) if no existing file is found.
    """
    if not os.path.exists(out_path):
        return "1=1"
    try:
        df_exist = pd.read_csv(out_path, usecols=["Create_Date"], low_memory=False)
        max_ts   = pd.to_numeric(df_exist["Create_Date"], errors="coerce").max()
        if pd.isna(max_ts):
            return "1=1"
        # ArcGIS where clause for Unix-ms timestamp field
        where = f"Create_Date > {int(max_ts)}"
        max_dt = pd.to_datetime(int(max_ts), unit="ms")
        log(f"  Existing data up to: {max_dt.strftime('%Y-%m-%d')} — fetching newer records only")
        return where
    except Exception as e:
        log(f"  ⚠  Could not read existing file ({e}) — full download")
        return "1=1"


def _get_max_date_violations(out_path: str) -> str:
    """
    Reads existing violations CSV and returns an ArcGIS WHERE clause
    filtering for CaseDate AFTER the latest date already stored.
    CaseDate is a string date field in the raw CSV.
    Returns '1=1' if no existing file.
    """
    if not os.path.exists(out_path):
        return "1=1"
    try:
        df_exist = pd.read_csv(out_path, usecols=["CaseDate"], low_memory=False)
        max_dt   = pd.to_datetime(df_exist["CaseDate"], errors="coerce").max()
        if pd.isna(max_dt):
            return "1=1"
        # ArcGIS date filter syntax
        where = f"CaseDate > DATE '{max_dt.strftime('%Y-%m-%d')}'"
        log(f"  Existing data up to: {max_dt.strftime('%Y-%m-%d')} — fetching newer records only")
        return where
    except Exception as e:
        log(f"  ⚠  Could not read existing file ({e}) — full download")
        return "1=1"


def fetch_incremental(url: str, label: str, out_filename: str,
                       where_clause: str) -> tuple:
    """
    Fetch only new records (delta) and APPEND them to the existing CSV.
    Falls back to the existing file if the API returns nothing new.
    Returns (out_path, stats_dict) where stats_dict contains:
      label, new_rows, total_rows, is_full, fetched (bool)
    """
    out_path    = os.path.join(DATASET_DIR, out_filename)
    is_full     = (where_clause == "1=1")

    log(f"{'Full download' if is_full else 'Incremental fetch'}: {label}")

    new_records = _paginate_api(url, where_clause, label)

    if not new_records:
        if os.path.exists(out_path):
            existing_rows = len(pd.read_csv(out_path, low_memory=False))
            log(f"  ✅ No new records — existing {out_filename} is up to date")
            return out_path, {"label": label, "new_rows": 0, "total_rows": existing_rows, "is_full": is_full, "fetched": False}
        else:
            log(f"  ✗  No data and no existing file — cannot continue")
            sys.exit(1)

    df_new = pd.DataFrame(new_records)
    log(f"  New records fetched: {len(df_new):,}")

    if is_full or not os.path.exists(out_path):
        # First run: just save
        df_new.to_csv(out_path, index=False)
        log(f"  💾 Saved {len(df_new):,} rows → {out_filename}")
        return out_path, {"label": label, "new_rows": len(df_new), "total_rows": len(df_new), "is_full": True, "fetched": True}
    else:
        # Subsequent runs: append new rows to existing
        df_exist = pd.read_csv(out_path, low_memory=False)
        before   = len(df_exist)
        df_combined = pd.concat([df_exist, df_new], ignore_index=True)
        df_combined.drop_duplicates(inplace=True)
        added = len(df_combined) - before
        df_combined.to_csv(out_path, index=False)
        log(f"  💾 Appended {added:,} new rows "
            f"(total: {len(df_combined):,}) → {out_filename}")
        return out_path, {"label": label, "new_rows": added, "total_rows": len(df_combined), "is_full": False, "fetched": True}


def step0_fetch_api():
    log(SEP)
    log("STEP 0 — INCREMENTAL DATA FETCH FROM APIs")
    log(SEP)

    path_311 = os.path.join(DATASET_DIR, "311_requests_full.csv")
    path_vio = os.path.join(DATASET_DIR, "montgomery_code_violations_full.csv")

    # Determine what's already on disk and build WHERE clauses
    where_311 = _get_max_date_311(path_311)
    where_vio = _get_max_date_violations(path_vio)

    # Fetch only the delta (or full data on first run)
    _, stats_311 = fetch_incremental(API_311, "311 Service Requests",  "311_requests_full.csv",              where_311)
    _, stats_vio = fetch_incremental(API_VIO, "Code Violations",       "montgomery_code_violations_full.csv", where_vio)

    return path_311, path_vio, [stats_311, stats_vio]


# ═══════════════════════════════════════════════════════════════
# STEP 1 — CLEAN 311 REQUESTS  (Cell 1 of development.ipynb)
# ═══════════════════════════════════════════════════════════════

def parse_int_date(series):
    """Try Unix-ms first, then YYYYMMDD integer, then generic parse."""
    s = series.dropna()
    if s.empty:
        return pd.to_datetime(series, errors="coerce")
    try:
        sample = float(s.iloc[0])
        if sample > 1e10:
            return pd.to_datetime(series, unit="ms", errors="coerce")
        elif sample > 1e7:
            return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")
    except (TypeError, ValueError):
        pass
    return pd.to_datetime(series, errors="coerce")


def step1_clean_311():
    log(SEP)
    log("STEP 1 — CLEANING 311 DATA")
    log(SEP)

    log("Loading data...")
    df = pd.read_csv(os.path.join(DATASET_DIR, "311_requests_full.csv"), low_memory=False)
    log(f"  Original shape: {df.shape}")

    # ── 2. Drop audit columns ─────────────────────────────────
    drop_cols = ["created_user", "created_date", "last_edited_user",
                 "last_edited_date", "GlobalID"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    log(f"  After dropping audit columns: {df.shape}")

    # ── 3. Consolidate lat/lon  ───────────────────────────────
    # Capital columns = no nulls → keep; drop lowercase duplicates
    # API geometry may have arrived as _api_x/_api_y → promote to Lat/Lon
    if "Latitude" not in df.columns and "_api_y" in df.columns:
        df.rename(columns={"_api_y": "Latitude", "_api_x": "Longitude"}, inplace=True)
    elif "Latitude" not in df.columns and "latitude" in df.columns:
        df.rename(columns={"latitude": "Latitude", "longitude": "Longitude"}, inplace=True)

    for drop in ["latitude", "longitude", "_api_x", "_api_y"]:
        if drop in df.columns and "Latitude" in df.columns:
            df.drop(columns=[drop], inplace=True)

    log(f"  After dropping duplicate lat/lon: {df.shape}")

    # ── 4. Drop rows with missing Address or bad coords ──────
    before = len(df)
    df.dropna(subset=["Address"], inplace=True)
    if "Latitude" in df.columns and "Longitude" in df.columns:
        df = df[(df["Latitude"] != 0.0) | (df["Longitude"] != 0.0)]
    log(f"  Dropped rows with missing Address or 0,0 coords → {df.shape}")

    # ── 5. Drop rows with missing Department ─────────────────
    df.dropna(subset=["Department"], inplace=True)

    # ── 6. Drop rows with missing District ───────────────────
    before = len(df)
    df.dropna(subset=["District"], inplace=True)
    log(f"  Dropped {before - len(df)} rows with missing District → {df.shape}")

    # ── 7. Parse dates ────────────────────────────────────────
    df["Create_Date"] = parse_int_date(df["Create_Date"])
    df["Close_Date"]  = pd.to_datetime(df["Close_Date"], unit="ms", errors="coerce")
    log(f"  Create_Date sample: {df['Create_Date'].dropna().iloc[0]}")
    log(f"  Close_Date  sample: {df['Close_Date'].dropna().iloc[0]}")

    # ── 8. Time features ──────────────────────────────────────
    df["Year"]           = df["Create_Date"].dt.year
    df["create_month"]   = df["Create_Date"].dt.month
    df["create_dow"]     = df["Create_Date"].dt.dayofweek
    df["create_quarter"] = df["Create_Date"].dt.quarter
    season_map = {12:"Winter", 1:"Winter",  2:"Winter",
                   3:"Spring",  4:"Spring",  5:"Spring",
                   6:"Summer",  7:"Summer",  8:"Summer",
                   9:"Fall",   10:"Fall",   11:"Fall"}
    df["season"] = df["create_month"].map(season_map)

    # ── 9. Resolution days ────────────────────────────────────
    df["resolution_days"] = (df["Close_Date"] - df["Create_Date"]).dt.days
    df.loc[df["resolution_days"] < 0, "resolution_days"] = np.nan
    log(f"  resolution_days: mean={df['resolution_days'].mean():.1f}, "
        f"median={df['resolution_days'].median():.1f}")

    # ── 10. Nuisance flag ─────────────────────────────────────
    nuisance_keywords = [
        "nuisance", "stagnant", "mosquito", "ditch", "drainage",
        "overgrown", "debris", "vacant", "illegal dump", "standing water",
        "leaves", "inlet", "sewage", "waste", "weed", "junk", "rodent"
    ]
    pattern = "|".join(nuisance_keywords)
    df["is_nuisance"] = (df["Request_Type"].str.lower()
                           .str.contains(pattern, na=False).astype(int))
    log(f"  Nuisance-flagged rows: {df['is_nuisance'].sum()} / {len(df)}")

    # ── 11. Chronic locations ─────────────────────────────────
    addr_counts   = df["Address"].str.strip().str.upper().value_counts()
    chronic_addrs = set(addr_counts[addr_counts >= 3].index)
    df["Address_upper"]       = df["Address"].str.strip().str.upper()
    df["is_chronic_location"] = df["Address_upper"].isin(chronic_addrs).astype(int)
    df.drop(columns=["Address_upper"], inplace=True)
    log(f"  Chronic locations flagged: {df['is_chronic_location'].sum()} rows")

    # ── 12. Encode categoricals ───────────────────────────────
    status_map = {"Closed": 0, "In Progress": 1, "Open": 2, "On Hold": 3}
    df["status_encoded"] = df["Status"].map(status_map)
    df["District"]       = df["District"].astype(float).fillna(0).astype(int)

    if "Origin" in df.columns:
        df = pd.get_dummies(df, columns=["Origin"], drop_first=False)
    df = pd.get_dummies(df, columns=["season"], drop_first=False)
    df["department_encoded"] = df["Department"].astype("category").cat.codes
    log(f"  Shape after encoding: {df.shape}")

    # ── 13. Cleanup raw date / id columns ────────────────────
    df.drop(columns=[c for c in ["OBJECTID"] 
                     if c in df.columns], inplace=True)

    log(f"\n✅ Final cleaned shape: {df.shape}")
    out = os.path.join(DATASET_DIR, "311_requests_cleaned.csv")
    df.to_csv(out, index=False)
    log(f"✅ Saved → Dataset/311_requests_cleaned.csv")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 2 — CLEAN CODE VIOLATIONS  (Cell 2 of development.ipynb)
# ═══════════════════════════════════════════════════════════════

def step2_clean_violations():
    log(SEP)
    log("STEP 2 — CLEANING CODE VIOLATIONS DATA")
    log(SEP)

    log("Loading data...")
    df = pd.read_csv(os.path.join(DATASET_DIR, "montgomery_code_violations_full.csv"),
                     low_memory=False)
    log(f"  Original shape: {df.shape}")

    # ── 2. Drop audit columns ─────────────────────────────────
    drop_cols = ["created_user", "created_date", "last_edited_user",
                 "last_edited_date", "GlobalID", "OBJECTID"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    log(f"  After dropping audit columns: {df.shape}")

    # ── 3. Drop DEBUG_TEST rows ───────────────────────────────
    if "CaseStatus" in df.columns:
        before = len(df)
        df = df[df["CaseStatus"] != "DEBUG_TEST"]
        log(f"  Dropped {before - len(df)} DEBUG_TEST rows → {df.shape}")

    # ── 4. Parse CaseDate ─────────────────────────────────────
    df["CaseDate"] = pd.to_datetime(df["CaseDate"], errors="coerce")
    log(f"  CaseDate sample: {df['CaseDate'].dropna().iloc[0]}")
    log(f"  CaseDate nulls after parse: {df['CaseDate'].isnull().sum()}")

    # ── 5. Time features ──────────────────────────────────────
    df["case_month"]   = df["CaseDate"].dt.month
    df["case_dow"]     = df["CaseDate"].dt.dayofweek
    df["case_quarter"] = df["CaseDate"].dt.quarter
    df["case_year"]    = df["CaseDate"].dt.year
    df["Year"]         = df["case_year"]
    df["Month"]        = df["case_month"]
    season_map = {12:"Winter", 1:"Winter",  2:"Winter",
                   3:"Spring",  4:"Spring",  5:"Spring",
                   6:"Summer",  7:"Summer",  8:"Summer",
                   9:"Fall",   10:"Fall",   11:"Fall"}
    df["season"] = df["case_month"].map(season_map)

    # ── 6. Handle missing CaseType ────────────────────────────
    df["CaseType"] = df["CaseType"].fillna("UNKNOWN")

    # ── 7. Drop missing CouncilDistrict ───────────────────────
    before = len(df)
    df.dropna(subset=["CouncilDistrict"], inplace=True)
    log(f"  Dropped {before - len(df)} rows with missing CouncilDistrict → {df.shape}")

    # ── 8. Drop missing Address ───────────────────────────────
    before = len(df)
    df.dropna(subset=["Address1"], inplace=True)
    log(f"  Dropped {before - len(df)} rows with missing Address → {df.shape}")

    # ── 9. Handle missing lat/lon ─────────────────────────────
    # API geometry may arrive as _api_x/_api_y → promote
    if "latitude" not in df.columns and "_api_y" in df.columns:
        df.rename(columns={"_api_y": "latitude", "_api_x": "longitude"}, inplace=True)
    elif "latitude" not in df.columns and "Latitude" in df.columns:
        df.rename(columns={"Latitude": "latitude", "Longitude": "longitude"}, inplace=True)
    for drop in ["_api_x", "_api_y"]:
        if drop in df.columns:
            df.drop(columns=[drop], inplace=True)

    before = len(df)
    df.dropna(subset=["latitude", "longitude"], inplace=True)
    log(f"  Dropped {before - len(df)} rows with missing lat/lon → {df.shape}")

    # ── 10. Fill missing Zip ──────────────────────────────────
    if "Zip" in df.columns:
        df["Zip"] = df.groupby("CouncilDistrict")["Zip"].transform(
            lambda x: x.fillna(x.median())
        )
        df["Zip"] = df["Zip"].fillna(df["Zip"].median())
        df["Zip"] = df["Zip"].astype(int)

    # ── 11. Clean STATE / City ────────────────────────────────
    if "STATE" in df.columns:
        df["STATE"] = df["STATE"].fillna("AL").str.upper().str.strip()
    if "City" in df.columns:
        df["City"] = df["City"].fillna("UNKNOWN").str.upper().str.strip()

    # ── 12. Lien status ───────────────────────────────────────
    df["LienStatus"] = df["LienStatus"].fillna("No Lien")

    # ── 13. Drop ComplaintRem (87% null free-text) ────────────
    if "ComplaintRem" in df.columns:
        df.drop(columns=["ComplaintRem"], inplace=True)

    # ── 14. Drop projected coordinate columns ─────────────────
    for col in ["ParcelNo_X", "ParcelNo_Y"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # ── 15. Chronic parcels ───────────────────────────────────
    parcel_counts   = df["ParcelNo"].value_counts()
    chronic_parcels = set(parcel_counts[parcel_counts >= 3].index)
    df["is_chronic_parcel"] = df["ParcelNo"].isin(chronic_parcels).astype(int)
    log(f"  Chronic parcels flagged: {df['is_chronic_parcel'].sum()} rows")

    # ── 16. Environmental violation flag ─────────────────────
    env_types = ["NUISANCE", "OPEN VACANT", "PARKING ON FRONT LAWN"]
    df["is_env_violation"] = df["CaseType"].isin(env_types).astype(int)
    log(f"  Environmental violations flagged: {df['is_env_violation'].sum()} rows")

    # ── 17. Open case flag ────────────────────────────────────
    df["is_open_case"] = (df["CaseStatus"] == "OPEN").astype(int)
    log(f"  Open cases flagged: {df['is_open_case'].sum()} rows")

    # ── 18. Encode categoricals ───────────────────────────────
    status_map = {"CLOSED": 0, "OPEN": 1}
    df["case_status_encoded"] = df["CaseStatus"].map(status_map)

    lien_map = {"No Lien": 0, "Lien Released": 1, "Lien Filed": 2}
    df["lien_status_encoded"] = df["LienStatus"].map(lien_map)

    df["CouncilDistrict"] = (df["CouncilDistrict"].astype(str)
                               .str.extract(r"(\d+)")[0]
                               .astype(float).astype("Int64"))

    df = pd.get_dummies(df, columns=["CaseType", "season"], drop_first=False)
    df["city_encoded"] = df["City"].astype("category").cat.codes

    log(f"  Shape after encoding: {df.shape}")

    # ── 19. Drop raw date + redundant columns ─────────────────
    df.drop(columns=[c for c in ["STATE"] if c in df.columns], inplace=True)

    log(f"\n✅ Final cleaned shape: {df.shape}")
    out = os.path.join(DATASET_DIR, "violations_cleaned.csv")
    df.to_csv(out, index=False)
    log(f"✅ Saved → Dataset/violations_cleaned.csv")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 3 — CLEAN WEATHER SIRENS  (Cell 3 of development.ipynb)
# ═══════════════════════════════════════════════════════════════

def step3_clean_sirens():
    log(SEP)
    log("STEP 3 — CLEANING WEATHER SIRENS")
    log(SEP)

    # If already cleaned, use it
    cleaned_path = os.path.join(DATASET_DIR, "sirens_cleaned.csv")
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
        log(f"  ✅ Loaded existing sirens_cleaned.csv: {df.shape}")
        return df

    raw_path = os.path.join(DATASET_DIR, "Weather_Sirens.csv")
    if not os.path.exists(raw_path):
        log("  ⚠  Weather_Sirens.csv not found — siren features will be unavailable")
        return pd.DataFrame(columns=["latitude", "longitude"])

    log("Loading data...")
    df = pd.read_csv(raw_path)
    log(f"  Original shape: {df.shape}")

    # ── 2. Drop unmatched geocodes (Status = 'T') ─────────────
    before = len(df)
    df = df[df["Status"] == "M"]
    log(f"  Dropped {before - len(df)} unmatched (T) sirens → {df.shape}")

    # ── 4. Select core columns ────────────────────────────────
    coord_cols = [c for c in df.columns
                  if c in ["X", "Y", "Longitude", "Latitude", "POINT_X", "POINT_Y", "LON", "LAT"]]
    log(f"  Detected coordinate columns: {coord_cols}")
    user_cols  = [c for c in df.columns if c.startswith("USER_")]
    keep_cols  = [c for c in ["ObjectID", "Status", "Score"] + coord_cols + user_cols
                  if c in df.columns]
    df = df[keep_cols]
    log(f"  After column reduction: {df.shape}")
    log(f"  Kept columns: {df.columns.tolist()}")

    # ── 5. Rename for clarity ─────────────────────────────────
    rename_map = {}
    for col in df.columns:
        clean = col.replace("USER_", "").strip().lower().replace(" ", "_").rstrip("_")
        if clean != col:
            rename_map[col] = clean
    for col in df.columns:
        if col in ["X", "LON", "Longitude", "POINT_X"]:
            rename_map[col] = "longitude"
        elif col in ["Y", "LAT", "Latitude", "POINT_Y"]:
            rename_map[col] = "latitude"
    df.rename(columns=rename_map, inplace=True)
    df.rename(columns={"ObjectID": "objectid", "Status": "status", "Score": "score"},
              inplace=True)
    log(f"  Renamed columns: {df.columns.tolist()}")

    # ── 6. Validate coordinates ───────────────────────────────
    if "latitude" in df.columns and "longitude" in df.columns:
        valid = (df["latitude"].between(32.0, 32.7) &
                 df["longitude"].between(-86.6, -85.9))
        n_invalid = (~valid).sum()
        if n_invalid:
            log(f"  ⚠  {n_invalid} sirens outside expected Montgomery bounds — flagging")
        df["coords_suspect"] = (~valid).astype(int)

    # ── 7. Encode in_city_limits ──────────────────────────────
    city_col = [c for c in df.columns if "city_limits" in c or "in_city" in c]
    if city_col:
        col = city_col[0]
        log(f"  in_city_limits values: {df[col].value_counts().to_dict()}")
        df["in_city_limits"] = (df[col].str.upper().str.strip() == "YES").astype(int)
        df.drop(columns=[col], inplace=True)

    # ── 8. Encode pike_road flag ──────────────────────────────
    pike_col = [c for c in df.columns if "pike" in c.lower()]
    if pike_col:
        col = pike_col[0]
        log(f"  pike_road values: {df[col].value_counts().to_dict()}")
        df["in_pike_road"] = (df[col].str.upper().str.strip() == "YES").astype(int)
        df.drop(columns=[col], inplace=True)

    # ── 9. Clean zip ──────────────────────────────────────────
    zip_col = [c for c in df.columns if "zip" in c.lower()]
    if zip_col:
        df[zip_col[0]] = pd.to_numeric(df[zip_col[0]], errors="coerce").astype("Int64")

    log(f"\n✅ Final cleaned shape: {df.shape}")
    df.to_csv(cleaned_path, index=False)
    log(f"✅ Saved → Dataset/sirens_cleaned.csv")
    log(f"\n   Total sirens in reference layer: {len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 4 — FEATURE ENGINEERING  (Cell 4 of development.ipynb)
# ═══════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


GRID_SIZE = 0.005   # ~500 m


def assign_grid_cell(lat, lon, grid_size=GRID_SIZE):
    grid_lat = round(round(lat / grid_size) * grid_size, 6)
    grid_lon = round(round(lon / grid_size) * grid_size, 6)
    return f"{grid_lat:.4f}_{grid_lon:.4f}"


def step4_feature_matrix():
    log(SEP)
    log("STEP 4 — BUILDING FEATURE MATRIX")
    log(SEP)

    log("Loading cleaned datasets...")
    df_311    = pd.read_csv(os.path.join(DATASET_DIR, "311_requests_cleaned.csv"))
    df_viol   = pd.read_csv(os.path.join(DATASET_DIR, "violations_cleaned.csv"))
    df_sirens = pd.read_csv(os.path.join(DATASET_DIR, "sirens_cleaned.csv"))

    log(f"  311 shape:        {df_311.shape}")
    log(f"  Violations shape: {df_viol.shape}")
    log(f"  Sirens shape:     {df_sirens.shape}")

    # ── 2. Calculate true days_ago for 311 & Violations ───────
    log("\nCalculating days_ago...")
    if "Create_Date" in df_311.columns:
        df_311["Create_Date"] = pd.to_datetime(df_311["Create_Date"], errors="coerce")
        ref_date = df_311["Create_Date"].dropna().max()
        if pd.isna(ref_date):
            ref_date = pd.Timestamp.now()
        df_311["days_ago"] = (ref_date - df_311["Create_Date"]).dt.days
    else:
        # Fallback if Create_Date is missing
        df_311["proxy_date"] = pd.to_datetime(
            df_311["Year"].astype(str) + "-" +
            df_311["create_month"].astype(str).str.zfill(2) + "-01",
            errors="coerce"
        )
        ref_date = df_311["proxy_date"].max()
        df_311["days_ago"] = (ref_date - df_311["proxy_date"]).dt.days

    log(f"  Reference date (latest): {ref_date.date()}")
    if "days_ago" in df_311.columns:
        log(f"  days_ago range: {df_311['days_ago'].min()} - {df_311['days_ago'].max()}")

    # Violations true date
    if "CaseDate" in df_viol.columns:
        df_viol["CaseDate"] = pd.to_datetime(df_viol["CaseDate"], errors="coerce")
        df_viol["days_ago"] = (ref_date - df_viol["CaseDate"]).dt.days

    # ── 3. Build spatial grid ─────────────────────────────────
    log("\nBuilding spatial grid...")
    lat_col_311  = "Latitude"  if "Latitude"  in df_311.columns  else "latitude"
    lon_col_311  = "Longitude" if "Longitude" in df_311.columns  else "longitude"
    lat_col_viol = "latitude"  if "latitude"  in df_viol.columns else "Latitude"
    lon_col_viol = "longitude" if "longitude" in df_viol.columns else "Longitude"

    df_311["grid_cell"]  = [assign_grid_cell(lat, lon)
                             for lat, lon in zip(df_311[lat_col_311], df_311[lon_col_311])]
    df_viol["grid_cell"] = [assign_grid_cell(lat, lon)
                             for lat, lon in zip(df_viol[lat_col_viol], df_viol[lon_col_viol])]

    log(f"  Grid cells in 311 data:        {df_311['grid_cell'].nunique()}")
    log(f"  Grid cells in violations data: {df_viol['grid_cell'].nunique()}")

    # ── 4. Aggregate 311 features ─────────────────────────────
    log("\nAggregating 311 features per grid cell...")

    def agg_311_window(df, min_days, max_days, suffix):
        sub = df[(df["days_ago"] > min_days) & (df["days_ago"] <= max_days)]
        return sub.groupby("grid_cell").agg(**{
            f"complaint_count_{suffix}":   ("Request_ID", "count"),
            f"nuisance_count_{suffix}":    ("is_nuisance", "sum"),
            f"chronic_loc_count_{suffix}": ("is_chronic_location", "sum"),
        }).reset_index()

    TARGET_WINDOW = 30
    # Use NON-OVERLAPPING historical windows to prevent label leakage
    agg_30 = agg_311_window(df_311, TARGET_WINDOW, TARGET_WINDOW + 30, "30d")
    agg_60 = agg_311_window(df_311, TARGET_WINDOW, TARGET_WINDOW + 60, "60d")
    agg_90 = agg_311_window(df_311, TARGET_WINDOW, TARGET_WINDOW + 90, "90d")

    # For general aggregate stats (like most_common_dept), exclude the target window
    df_historical = df_311[df_311["days_ago"] > TARGET_WINDOW]

    agg_cols = {
        "total_complaints":         ("Request_ID",          "count"),
        "total_nuisance":           ("is_nuisance",          "sum"),
        "total_chronic_locations":  ("is_chronic_location",  "sum"),
        "days_since_last_complaint":("days_ago",             lambda x: x.min() - TARGET_WINDOW if len(x) > 0 else np.nan),
        "complaint_years_active":   ("Year",                 "nunique"),
        "most_common_dept":         ("department_encoded",
                                     lambda x: x.mode()[0] if len(x) > 0 else -1),
    }
    if "resolution_days" in df_historical.columns:
        agg_cols["avg_resolution_days"] = ("resolution_days", "mean")

    agg_all = df_historical.groupby("grid_cell").agg(**agg_cols).reset_index()
    agg_all["nuisance_rate"] = (agg_all["total_nuisance"] /
                                 agg_all["total_complaints"]).round(4)

    call_col = [c for c in df_historical.columns if "Call_Center" in c or "Call Center" in c]
    if call_col:
        call_agg = df_historical.groupby("grid_cell")[call_col[0]].mean().reset_index()
        call_agg.columns = ["grid_cell", "pct_call_center"]
        agg_all = agg_all.merge(call_agg, on="grid_cell", how="left")

    features_311 = agg_all.copy()
    for agg in [agg_30, agg_60, agg_90]:
        features_311 = features_311.merge(agg, on="grid_cell", how="left")

    fill_cols = [c for c in features_311.columns
                 if any(x in c for x in ["count", "nuisance", "chronic"])]
    features_311[fill_cols] = features_311[fill_cols].fillna(0)
    log(f"  311 feature matrix: {features_311.shape}")

    # ── 5. Aggregate violation features ──────────────────────
    log("\nAggregating violation features per grid cell...")
    viol_agg_cols = {
        "total_violations":      ("OffenceNum",        "count"),
        "open_violations":       ("is_open_case",       "sum"),
        "env_violations":        ("is_env_violation",   "sum"),
        "chronic_parcels":       ("is_chronic_parcel",  "sum"),
        "unique_parcels":        ("ParcelNo",           "nunique"),
        "violation_years_active":("Year",               "nunique"),
    }
    if "lien_status_encoded" in df_viol.columns:
        viol_agg_cols["lien_filed_count"] = (
            "lien_status_encoded", lambda x: (x == 2).sum())
        viol_agg_cols["avg_lien_status"]  = ("lien_status_encoded", "mean")

    # Restrict violations to purely historical data (> TARGET_WINDOW) to match 311 logic
    if "days_ago" in df_viol.columns:
        df_viol_hist = df_viol[df_viol["days_ago"] > TARGET_WINDOW]
    else:
        df_viol_hist = df_viol

    features_viol = df_viol_hist.groupby("grid_cell").agg(**viol_agg_cols).reset_index()
    features_viol["open_violation_rate"]  = (features_viol["open_violations"]  /
                                              features_viol["total_violations"]).round(4)
    features_viol["env_violation_rate"]   = (features_viol["env_violations"]   /
                                              features_viol["total_violations"]).round(4)
    features_viol["chronic_parcel_rate"]  = (features_viol["chronic_parcels"]  /
                                              features_viol["unique_parcels"]).round(4)
    log(f"  Violation feature matrix: {features_viol.shape}")

    # ── 6. Siren distances ────────────────────────────────────
    log("\nComputing siren coverage distances...")
    siren_lat_col = [c for c in df_sirens.columns if "lat"  in c.lower()][0]
    siren_lon_col = [c for c in df_sirens.columns if "lon"  in c.lower()][0]
    siren_coords  = list(zip(df_sirens[siren_lat_col], df_sirens[siren_lon_col]))

    def get_cell_center(cell):
        parts = cell.split("_")
        return float(parts[0]), float(parts[1])

    all_cells = set(features_311["grid_cell"]) | set(features_viol["grid_cell"])
    log(f"  Computing distances for {len(all_cells)} grid cells...")

    siren_dists = {}
    for cell in all_cells:
        try:
            clat, clon = get_cell_center(cell)
            siren_dists[cell] = round(
                min(haversine_km(clat, clon, s[0], s[1]) for s in siren_coords), 4
            )
        except Exception:
            siren_dists[cell] = np.nan

    df_siren_feat = pd.DataFrame(list(siren_dists.items()),
                                  columns=["grid_cell", "dist_to_nearest_siren_km"])
    df_siren_feat["siren_coverage_gap"] = (
        df_siren_feat["dist_to_nearest_siren_km"] > 3.0).astype(int)
    log(f"  Cells with siren gap (>3km): {df_siren_feat['siren_coverage_gap'].sum()}")

    # ── 7. Build target variable ──────────────────────────────
    log("\nBuilding target variable...")
    # TARGET_WINDOW is defined above, ensuring historical features don't leak label data
    df_future = df_311[df_311["days_ago"] <= TARGET_WINDOW]
    target = df_future.groupby("grid_cell").agg(
        target_nuisance_binary=("is_nuisance", lambda x: int(x.sum() > 0)),
        target_nuisance_count= ("is_nuisance", "sum"),
        target_any_complaint=  ("Request_ID",  lambda x: int(len(x) > 0)),
    ).reset_index()
    log(f"  Target positive rate (nuisance): {target['target_nuisance_binary'].mean():.2%}")
    log(f"  Target positive rate (any):      {target['target_any_complaint'].mean():.2%}")

    # ── 8. Merge into unified feature matrix ─────────────────
    log("\nMerging all features...")
    fm = features_311.merge(features_viol,  on="grid_cell", how="outer")
    fm = fm.merge(df_siren_feat,            on="grid_cell", how="left")
    fm = fm.merge(target,                   on="grid_cell", how="left")

    fm[["cell_lat", "cell_lon"]] = (
        fm["grid_cell"].str.split("_", expand=True).astype(float)
    )

    zero_fill = [c for c in fm.columns if any(x in c for x in
                 ["count", "nuisance", "violation", "chronic", "lien",
                  "parcel", "complaint", "target"])]
    fm[zero_fill] = fm[zero_fill].fillna(0)
    for col in fm.select_dtypes(include=[np.number]).columns:
        if fm[col].isnull().sum() > 0:
            fm[col] = fm[col].fillna(fm[col].median())

    log(f"\n  Final feature matrix shape: {fm.shape}")

    # ── 9. Summary ────────────────────────────────────────────
    groups = {
        "Rolling Windows (30/60/90d)": [c for c in fm.columns
                                         if any(x in c for x in ["_30d","_60d","_90d"])],
        "311 All-Time":   [c for c in fm.columns
                           if "complaint" in c or "resolution" in c or "nuisance_rate" in c],
        "Violations":     [c for c in fm.columns
                           if "violation" in c or "lien" in c or "parcel" in c],
        "Siren Coverage": [c for c in fm.columns if "siren" in c],
        "Spatial":        ["cell_lat", "cell_lon"],
        "Target":         [c for c in fm.columns if "target" in c],
    }
    log("\n--- Feature Groups ---")
    for g, cols in groups.items():
        log(f"  {g} ({len(cols)}): {cols}")

    nulls = fm.isnull().sum()
    nulls = nulls[nulls > 0]
    log("\n--- Null Check ---")
    log("  No nulls!" if len(nulls) == 0 else nulls.to_string())

    # ── 10. Save ──────────────────────────────────────────────
    out = os.path.join(DATASET_DIR, "feature_matrix.csv")
    fm.to_csv(out, index=False)
    log(f"\nSaved -> feature_matrix.csv")
    log(f"   Grid cells (rows): {len(fm)}")
    log(f"   Total columns:     {fm.shape[1]}")
    log("\nReady for ML model training!")
    return fm


# ═══════════════════════════════════════════════════════════════
# STEP 5 — TRAIN MODEL + SCORE  (Cell 5 of development.ipynb)
# ═══════════════════════════════════════════════════════════════

def step5_train_and_score():
    log(SEP)
    log("STEP 5 — TRAINING MODEL & SCORING GRID CELLS")
    log(SEP)

    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score,
        roc_curve, precision_recall_curve, average_precision_score,
    )
    import joblib

    # ── 1. Load feature matrix ────────────────────────────────
    log("Loading feature matrix...")
    df = pd.read_csv(os.path.join(DATASET_DIR, "feature_matrix.csv"))
    log(f"  Shape: {df.shape}")

    # ── 2. Define features & target ───────────────────────────
    DROP_COLS = ["grid_cell", "target_nuisance_binary",
                 "target_nuisance_count", "target_any_complaint"]
    TARGET = "target_nuisance_binary"

    feature_cols = [c for c in df.columns
                    if c not in DROP_COLS
                    and df[c].dtype in [np.float64, np.int64, float, int]]

    X = df[feature_cols].fillna(0)
    y = df[TARGET].fillna(0).astype(int)

    log(f"\n  Features:        {len(feature_cols)}")
    log(f"  Target positive: {y.sum()} / {len(y)} ({y.mean():.2%})")

    # ── 3. Train / test split (stratified) ───────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log(f"\n  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ── 4. Train Random Forest ────────────────────────────────
    log("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    log("  Done!")

    # ── 5. Cross-validation ───────────────────────────────────
    log("\nRunning 5-fold cross-validation...")
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
    log(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # ── 6. Evaluate on test set ───────────────────────────────
    log("\n" + "=" * 50)
    log("TEST SET EVALUATION")
    log("=" * 50)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc  = average_precision_score(y_test, y_pred_proba)

    log(f"\n  ROC-AUC:              {roc_auc:.4f}")
    log(f"  Precision-Recall AUC: {pr_auc:.4f}")
    log(f"\n  Classification Report:")
    log(classification_report(y_test, y_pred, target_names=["No Nuisance", "Nuisance"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    log(f"  Confusion Matrix:")
    log(f"                   Predicted No  Predicted Yes")
    log(f"  Actual No            {tn:>5}          {fp:>5}")
    log(f"  Actual Yes           {fn:>5}          {tp:>5}")
    log(f"\n  True Positives (caught hazards):    {tp}")
    log(f"  False Positives (unnecessary trips): {fp}")
    log(f"  False Negatives (missed hazards):    {fn}")

    # ── 7. Find optimal threshold (max F1) ────────────────────
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores      = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx       = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    log(f"\n  Optimal threshold: {best_threshold:.4f}  (F1 = {f1_scores[best_idx]:.4f})")

    # ── 8. Feature importance ─────────────────────────────────
    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    log(f"\n  Top 15 Features:")
    log(importance_df.head(15).to_string(index=False))

    # ── 9. Score all grid cells ───────────────────────────────
    log("\nScoring all grid cells...")
    X_all = df[feature_cols].fillna(0)
    df["risk_score"] = model.predict_proba(X_all)[:, 1]
    df["risk_label"] = pd.cut(
        df["risk_score"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    df["risk_flag"] = (df["risk_score"] >= best_threshold).astype(int)

    log(f"\n  Risk Distribution:")
    log(df["risk_label"].value_counts().to_string())
    log(f"\n  High-risk zones flagged: {df['risk_flag'].sum()}")

    # ── 10. Save evaluation charts ────────────────────────────
    log("\nGenerating charts...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Proactive Environmental Safety Predictor — Model Evaluation",
                 fontsize=14, fontweight="bold")

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].legend()

    axes[0, 1].plot(recalls, precisions, color="darkorange", lw=2,
                    label=f"PR AUC = {pr_auc:.3f}")
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend()

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0],
                xticklabels=["No Nuisance", "Nuisance"],
                yticklabels=["No Nuisance", "Nuisance"])
    axes[1, 0].set_title("Confusion Matrix")
    axes[1, 0].set_ylabel("Actual")
    axes[1, 0].set_xlabel("Predicted")

    top15 = importance_df.head(15)
    axes[1, 1].barh(top15["feature"][::-1], top15["importance"][::-1], color="steelblue")
    axes[1, 1].set_title("Top 15 Feature Importances")
    axes[1, 1].set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "model_evaluation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("  Saved -> model_evaluation.png")

    # ── 11. Save all outputs ──────────────────────────────────
    import joblib
    joblib.dump(model, os.path.join(BASE_DIR, "nuisance_predictor.pkl"))
    log("  Saved -> nuisance_predictor.pkl")

    dashboard_cols = ["grid_cell", "cell_lat", "cell_lon",
                      "risk_score", "risk_label", "risk_flag",
                      "target_nuisance_binary"]
    optional_cols  = ["total_complaints", "total_nuisance", "open_violations",
                      "dist_to_nearest_siren_km", "siren_coverage_gap",
                      "nuisance_rate", "open_violation_rate", "chronic_parcel_rate"]
    keep = [c for c in dashboard_cols + optional_cols if c in df.columns]
    df[keep].to_csv(os.path.join(DATASET_DIR, "risk_scores.csv"), index=False)
    log("  Saved -> risk_scores.csv")

    importance_df.to_csv(os.path.join(DATASET_DIR, "feature_importance.csv"), index=False)
    log("  Saved -> feature_importance.csv")

    log(f"""
{'=' * 50}
SUMMARY
{'=' * 50}
  Model:             Random Forest (200 trees)
  Features used:     {len(feature_cols)}
  Grid cells scored: {len(df)}
  CV ROC-AUC:        {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})
  Test ROC-AUC:      {roc_auc:.4f}
  Optimal threshold: {best_threshold:.4f}
  High-risk zones:   {df['risk_flag'].sum()}

  Output files:
    nuisance_predictor.pkl  -> trained model
    risk_scores.csv         -> risk scores per grid cell (dashboard)
    feature_importance.csv  -> feature rankings
    model_evaluation.png    -> evaluation charts
{'=' * 50}
""")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log(SEP)
    log("SafeCity Montgomery — AUTO PIPELINE")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(SEP)

    _, _, fetch_stats = step0_fetch_api()       # fetch fresh raw data
    step1_clean_311()       # Cell 1
    step2_clean_violations()# Cell 2
    step3_clean_sirens()    # Cell 3
    step4_feature_matrix()  # Cell 4
    step5_train_and_score() # Cell 5

    elapsed = round(time.time() - t0, 1)
    log(f"\nTotal elapsed: {elapsed}s")
    log("Dashboard files are up to date — refresh your Streamlit tab!")

    return {"fetch_stats": fetch_stats, "elapsed": elapsed}


if __name__ == "__main__":
    main()
