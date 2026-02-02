#!/usr/bin/env python3
"""
03_score.py
-----------
Score a NEW raw CSV (Dataset B) using the trained LightGBM model.

Key requirements:
- Reproduce Notebook 1 transformations
- Use artifacts/metadata.json to:
    - reuse major_countries (consistent country grouping)
    - enforce feature column order and presence

Outputs:
- scores CSV with externalcustomerkey + score

Usage:
  python scripts/03_score.py \
    --input path/to/new_raw.csv \
    --model artifacts/model.txt \
    --metadata artifacts/metadata.json \
    --out data/scores.csv
"""

import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd
import lightgbm as lgb

DATE_COLS = ["as_of_date", "first_order_date", "last_order_date"]
WINDOW_TOKENS = ["_15d", "_30d", "_3m", "_6m", "_12m"]

LEAKAGE_COLS = [
    "discount_abs_lifetime_eur",
    "discount_rate_lifetime",
    "share_of_orders_with_discount",
    "share_of_items_discounted",
    "avg_discount_per_order",
    "max_discount_single_order",
]

DROP_ALWAYS = ["gender", "shops_included", "registration_flag"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to raw CSV to score")
    p.add_argument("--model", default="artifacts/model.txt", help="Path to saved LightGBM model.txt")
    p.add_argument("--metadata", default="artifacts/metadata.json", help="Path to metadata.json from training prep")
    p.add_argument("--out", default="data/scores.csv", help="Output scores CSV")
    return p.parse_args()


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.out)

    # Load metadata (feature list + country mapping)
    if not os.path.exists(args.metadata):
        raise FileNotFoundError(f"metadata not found: {args.metadata}")

    with open(args.metadata, "r") as f:
        meta = json.load(f)

    major_countries: List[str] = meta["major_countries"]
    feature_cols: List[str] = meta["feature_columns"]

    # Load model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"model not found: {args.model}")
    model = lgb.Booster(model_file=args.model)

    # Load raw scoring CSV
    df = pd.read_csv(args.input)
    print("Loaded scoring CSV shape:", df.shape)

    # Keep id for output
    if "externalcustomerkey" not in df.columns:
        raise ValueError("externalcustomerkey is required in scoring input")
    ids = df["externalcustomerkey"].copy()

    # Parse dates (even if we drop them, keeps parity and avoids type surprises)
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Drop 100% missing columns
    all_null_cols = [c for c in df.columns if df[c].isna().all()]
    if all_null_cols:
        df = df.drop(columns=all_null_cols)

    # Fill rolling-window numeric cols NaN -> 0
    window_cols = [
        c for c in df.columns
        if any(tok in c for tok in WINDOW_TOKENS) and c not in DATE_COLS
    ]
    window_cols = [c for c in window_cols if c in df.columns and df[c].dtype != "object"]
    for c in window_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(0)

    # Fill avg_discount_per_order NaN -> 0 (even though we drop leakage later, keeps parity)
    if "avg_discount_per_order" in df.columns:
        df["avg_discount_per_order"] = df["avg_discount_per_order"].fillna(0)

    # Drop leakage columns if present (target columns may exist in raw; we do NOT use them)
    leakage_present = [c for c in LEAKAGE_COLS if c in df.columns]
    if leakage_present:
        df = df.drop(columns=leakage_present)

    # Country encoding using TRAINING major_countries
    if "country" not in df.columns:
        raise ValueError("country column is required in scoring input (before encoding)")

    df["country_grouped"] = df["country"].where(df["country"].isin(major_countries), "OTHER")
    country_dummies = pd.get_dummies(df["country_grouped"], prefix="country", dummy_na=False)
    df = pd.concat([df.drop(columns=["country", "country_grouped"]), country_dummies], axis=1)

    # Drop unwanted cols if present
    for c in DROP_ALWAYS:
        if c in df.columns:
            # Only drop registration_flag if constant? Here we drop it always for consistency with training export.
            df = df.drop(columns=[c])

    # Drop ID/date cols (model never used them)
    drop_cols = [c for c in (["externalcustomerkey"] + DATE_COLS) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Align columns to training feature list:
    # - add missing columns as 0
    # - drop unexpected extra columns
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    extra_cols = [c for c in df.columns if c not in feature_cols]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    # Ensure correct column order
    df = df[feature_cols]

    # Final check: numeric-only
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        raise ValueError(f"Non-numeric columns remain after alignment: {obj_cols}")

    # Score
    scores = model.predict(df)

    out = pd.DataFrame({
        "externalcustomerkey": ids,
        "score": scores
    })
    out.to_csv(args.out, index=False)
    print("Saved scores ->", args.out)
    print(out.head())


if __name__ == "__main__":
    main()
