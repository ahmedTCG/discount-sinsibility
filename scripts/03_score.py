#!/usr/bin/env python3
"""
03_score.py - Score customers using trained model.
"""

import argparse
import json
import os

import pandas as pd
import lightgbm as lgb

from config import (
    ID_COLS, DATE_COLS, LEAKAGE_COLS, DROP_COLS, WINDOW_TOKENS, DEFAULT_PATHS,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to raw CSV to score")
    p.add_argument("--model", default=DEFAULT_PATHS["model"])
    p.add_argument("--metadata", default=DEFAULT_PATHS["metadata"])
    p.add_argument("--out", default=DEFAULT_PATHS["scores"])
    return p.parse_args()


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.out)

    if not os.path.exists(args.metadata):
        raise FileNotFoundError(f"Metadata not found: {args.metadata}")
    with open(args.metadata, "r") as f:
        meta = json.load(f)
    major_countries = meta["major_countries"]
    feature_cols = meta["feature_columns"]

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    model = lgb.Booster(model_file=args.model)

    df = pd.read_csv(args.input)
    print(f"Loaded scoring CSV shape: {df.shape}")

    if "externalcustomerkey" not in df.columns:
        raise ValueError("externalcustomerkey required in scoring input")
    ids = df["externalcustomerkey"].copy()

    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    all_null_cols = [c for c in df.columns if df[c].isna().all()]
    if all_null_cols:
        df = df.drop(columns=all_null_cols)

    window_cols = [c for c in df.columns if any(tok in c for tok in WINDOW_TOKENS) and c not in DATE_COLS and df[c].dtype != "object"]
    for c in window_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(0)

    if "avg_discount_per_order" in df.columns:
        df["avg_discount_per_order"] = df["avg_discount_per_order"].fillna(0)

    leakage_present = [c for c in LEAKAGE_COLS if c in df.columns]
    if leakage_present:
        df = df.drop(columns=leakage_present)

    if "country" not in df.columns:
        raise ValueError("Country column required for scoring")
    df["country_grouped"] = df["country"].where(df["country"].isin(major_countries), "OTHER")
    country_dummies = pd.get_dummies(df["country_grouped"], prefix="country", dummy_na=False)
    df = pd.concat([df.drop(columns=["country", "country_grouped"]), country_dummies], axis=1)

    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    drop_cols = [c for c in (ID_COLS + DATE_COLS) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    extra_cols = [c for c in df.columns if c not in feature_cols]
    if extra_cols:
        df = df.drop(columns=extra_cols)
    df = df[feature_cols]

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        raise ValueError(f"Non-numeric columns remain: {obj_cols}")

    scores = model.predict(df)

    out = pd.DataFrame({"externalcustomerkey": ids, "score": scores})
    out.to_csv(args.out, index=False)
    print(f"Saved scores -> {args.out}")
    print(out.head())


if __name__ == "__main__":
    main()
