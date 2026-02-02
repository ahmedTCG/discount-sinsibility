#!/usr/bin/env python3
"""
01_prepare_data.py
------------------
Ingest raw CSV and reproduce Notebook 1 preprocessing exactly:

- Parse dates: as_of_date, first_order_date, last_order_date
- Drop 100% missing columns (e.g., account_age_days)
- For rolling-window columns (*_15d, *_30d, *_3m, *_6m, *_12m): fill NaN -> 0 (no activity)
- Fill avg_discount_per_order NaN -> 0 (no discount activity)
- Create target y_discount_sensitive:
    1 if share_of_orders_with_discount > 0 OR discount_abs_lifetime_eur > 0 else 0
- Drop leakage columns:
    discount_abs_lifetime_eur, discount_rate_lifetime, share_of_orders_with_discount,
    share_of_items_discounted, avg_discount_per_order, max_discount_single_order
- Country encoding:
    keep countries with >= 1% share in THIS training prep run; others -> OTHER; one-hot encode
    Save the kept list to artifacts/metadata.json for consistent scoring later.
- Drop gender completely
- Drop constant/useless flags if present: shops_included, registration_flag (if constant)
- Drop ID/date columns before export:
    externalcustomerkey, as_of_date, first_order_date, last_order_date
- Drop any remaining NaNs (should be extremely rare)
- Export parquet: data/df_model_clean.parquet

Usage:
  python scripts/01_prepare_data.py --input path/to/raw.csv --out data/df_model_clean.parquet

Notes:
- This script is intended to be run on the "training-like" dataset to establish metadata.
- For scoring new datasets later, we will reuse metadata.json to keep encoding consistent.
"""

import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd


LEAKAGE_COLS = [
    "discount_abs_lifetime_eur",
    "discount_rate_lifetime",
    "share_of_orders_with_discount",
    "share_of_items_discounted",
    "avg_discount_per_order",
    "max_discount_single_order",
]

DATE_COLS = ["as_of_date", "first_order_date", "last_order_date"]
ID_COLS = ["externalcustomerkey"]
DROP_GENDER = ["gender"]

WINDOW_TOKENS = ["_15d", "_30d", "_3m", "_6m", "_12m"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to raw CSV")
    p.add_argument("--out", default="data/df_model_clean.parquet", help="Output parquet path")
    p.add_argument("--metadata_out", default="artifacts/metadata.json", help="Metadata JSON output path")
    p.add_argument("--country_min_share", type=float, default=0.01, help="Min share to keep a country (default 1%)")
    return p.parse_args()


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.out)
    ensure_dir(args.metadata_out)

    print(f"Loading CSV: {args.input}")
    df = pd.read_csv(args.input)
    print("Raw shape:", df.shape)

    # 1) Parse dates
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # Validate date parsing quickly
    for c in DATE_COLS:
        if c in df.columns:
            if df[c].isna().any():
                # We don't expect NaNs here based on our notebook runs, but don't hard-fail.
                print(f"Warning: {c} has {int(df[c].isna().sum()):,} NaNs after parsing")

    # 2) Drop 100% missing columns
    all_null_cols = [c for c in df.columns if df[c].isna().all()]
    if all_null_cols:
        print("Dropping all-null columns:", all_null_cols)
        df = df.drop(columns=all_null_cols)

    # 3) Fill rolling-window numeric cols NaN -> 0 (no activity)
    window_cols = [
        c for c in df.columns
        if any(tok in c for tok in WINDOW_TOKENS) and c not in DATE_COLS
    ]
    # Only numeric-like (avoid accidental object fill)
    window_cols = [c for c in window_cols if c in df.columns and df[c].dtype != "object"]
    filled = 0
    for c in window_cols:
        n = int(df[c].isna().sum())
        if n:
            df[c] = df[c].fillna(0)
            filled += n
    print(f"Filled {filled:,} NaNs -> 0 across {len(window_cols)} window columns")

    # 4) Fill avg_discount_per_order NaN -> 0 (no discount activity)
    if "avg_discount_per_order" in df.columns:
        n = int(df["avg_discount_per_order"].isna().sum())
        if n:
            df["avg_discount_per_order"] = df["avg_discount_per_order"].fillna(0)
            print(f"Filled {n:,} NaNs -> 0 in avg_discount_per_order")

    # 5) Create target y_discount_sensitive
    required = ["share_of_orders_with_discount", "discount_abs_lifetime_eur"]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required columns for target: {missing_req}")

    df["y_discount_sensitive"] = (
        (df["share_of_orders_with_discount"] > 0) |
        (df["discount_abs_lifetime_eur"] > 0)
    ).astype(np.int8)

    print("Target distribution:")
    print(df["y_discount_sensitive"].value_counts())
    print("Positive rate:", float(df["y_discount_sensitive"].mean()))

    # 6) Drop leakage columns
    leakage_present = [c for c in LEAKAGE_COLS if c in df.columns]
    if leakage_present:
        df = df.drop(columns=leakage_present)
        print("Dropped leakage cols:", leakage_present)

    # 7) Country one-hot encoding (save kept list for consistent scoring later)
    if "country" not in df.columns:
        raise ValueError("Expected column 'country' not found")

    country_share = df["country"].value_counts(normalize=True)
    major_countries: List[str] = country_share[country_share >= args.country_min_share].index.tolist()

    df["country_grouped"] = df["country"].where(df["country"].isin(major_countries), "OTHER")
    country_dummies = pd.get_dummies(df["country_grouped"], prefix="country", dummy_na=False)
    df = pd.concat([df.drop(columns=["country", "country_grouped"]), country_dummies], axis=1)

    print(f"Country kept (>= {args.country_min_share:.2%} share): {len(major_countries)}")

    # 8) Drop gender completely
    for c in DROP_GENDER:
        if c in df.columns:
            df = df.drop(columns=[c])
            print("Dropped:", c)

    # 9) Drop constant/useless flags if present
    for c in ["shops_included", "registration_flag"]:
        if c in df.columns:
            if df[c].nunique(dropna=False) <= 1:
                df = df.drop(columns=[c])
                print("Dropped constant column:", c)

    # 10) Drop ID + dates before export
    drop_cols = [c for c in ID_COLS + DATE_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print("Dropped ID/date columns:", drop_cols)

    # 11) Final cleanup: drop remaining NaNs (should be extremely rare)
    nan_rows = int(df.isna().any(axis=1).sum())
    if nan_rows:
        print(f"Dropping {nan_rows:,} rows with remaining NaNs")
        df = df.dropna()

    # 12) Verify numeric-only (bools are OK)
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        raise ValueError(f"Non-numeric columns remain: {obj_cols}")

    print("Final shape:", df.shape)

    # Save parquet
    df.to_parquet(args.out, index=False)
    print("Saved parquet ->", args.out)

    # Save metadata (for consistent scoring later)
    metadata = {
        "country_min_share": args.country_min_share,
        "major_countries": major_countries,
        "feature_columns": [c for c in df.columns if c != "y_discount_sensitive"],
        "target_column": "y_discount_sensitive",
        "notes": "Generated by 01_prepare_data.py; reuse major_countries+feature_columns for scoring datasets.",
    }
    with open(args.metadata_out, "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata ->", args.metadata_out)


if __name__ == "__main__":
    main()
