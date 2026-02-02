#!/usr/bin/env python3
"""
02_train_model.py
-----------------
Train a LightGBM binary classifier on the prepared parquet from Script 1.

Inputs:
- data/df_model_clean.parquet (default)
- artifacts/metadata.json (optional, for feature list)

Outputs:
- artifacts/model.txt
- artifacts/feature_importance.csv
- artifacts/train_metrics.json

Usage:
  python scripts/02_train_model.py --data data/df_model_clean.parquet
"""

import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/df_model_clean.parquet", help="Prepared parquet path")
    p.add_argument("--metadata", default="artifacts/metadata.json", help="Metadata JSON path (optional)")
    p.add_argument("--model_out", default="artifacts/model.txt", help="Output model path")
    p.add_argument("--fi_out", default="artifacts/feature_importance.csv", help="Output feature importance CSV")
    p.add_argument("--metrics_out", default="artifacts/train_metrics.json", help="Output metrics JSON")
    p.add_argument("--test_size", type=float, default=0.2, help="Validation size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--num_boost_round", type=int, default=300, help="Max boosting rounds")
    p.add_argument("--early_stopping", type=int, default=30, help="Early stopping rounds")
    return p.parse_args()


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    args = parse_args()
    for pth in [args.model_out, args.fi_out, args.metrics_out]:
        ensure_dir(pth)

    print("Loading:", args.data)
    df = pd.read_parquet(args.data)
    print("Data shape:", df.shape)

    if "y_discount_sensitive" not in df.columns:
        raise ValueError("Target column y_discount_sensitive not found")

    # Use feature list from metadata if available (keeps strict ordering)
    feature_cols = [c for c in df.columns if c != "y_discount_sensitive"]
    if os.path.exists(args.metadata):
        try:
            with open(args.metadata, "r") as f:
                meta = json.load(f)
            if "feature_columns" in meta:
                feature_cols = meta["feature_columns"]
        except Exception as e:
            print("Warning: could not read metadata feature_columns:", e)

    X = df[feature_cols]
    y = df["y_discount_sensitive"].astype(int)

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )

    print("Train:", X_train.shape, " Val:", X_val.shape)
    print("Train pos rate:", float(y_train.mean()), " Val pos rate:", float(y_val.mean()))

    # LightGBM datasets
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    # Solid baseline parameters (robust to heavy tails, no scaling needed)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": args.seed,
    }

    # Train with callback-based early stopping (compatible with newer LightGBM)
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=args.num_boost_round,
        valid_sets=[lgb_val],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping)]
    )

    # Evaluate
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_val_pred)
    print(f"Validation ROC-AUC: {auc:.6f}")
    print("Best iteration:", model.best_iteration)

    # Save model
    model.save_model(args.model_out)
    print("Saved model ->", args.model_out)

    # Feature importance
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.feature_importance(importance_type="gain"),
        "importance_split": model.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    fi.to_csv(args.fi_out, index=False)
    print("Saved feature importance ->", args.fi_out)

    # Save metrics
    metrics = {
        "validation_auc": float(auc),
        "best_iteration": int(model.best_iteration),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "pos_rate_train": float(y_train.mean()),
        "pos_rate_val": float(y_val.mean()),
        "params": params,
        "feature_count": int(len(feature_cols)),
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics ->", args.metrics_out)


if __name__ == "__main__":
    main()
