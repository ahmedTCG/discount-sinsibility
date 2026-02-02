#!/usr/bin/env python3
"""
02_train_model.py - Train LightGBM classifier.
"""

import argparse
import json
import os

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from config import (
    TARGET_COLUMN, TEST_SIZE, RANDOM_SEED, LGBM_PARAMS,
    NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS, DEFAULT_PATHS,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_PATHS["data_clean"])
    p.add_argument("--metadata", default=DEFAULT_PATHS["metadata"])
    p.add_argument("--model_out", default=DEFAULT_PATHS["model"])
    p.add_argument("--fi_out", default=DEFAULT_PATHS["feature_importance"])
    p.add_argument("--metrics_out", default=DEFAULT_PATHS["metrics"])
    p.add_argument("--test_size", type=float, default=TEST_SIZE)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    args = parse_args()
    for pth in [args.model_out, args.fi_out, args.metrics_out]:
        ensure_dir(pth)

    print(f"Loading: {args.data}")
    df = pd.read_parquet(args.data)
    print(f"Data shape: {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column {TARGET_COLUMN} not found")

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    if os.path.exists(args.metadata):
        with open(args.metadata, "r") as f:
            meta = json.load(f)
        if "feature_columns" in meta:
            feature_cols = meta["feature_columns"]

    X = df[feature_cols]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    print(f"Train: {X_train.shape}  Val: {X_val.shape}")
    print(f"Train pos rate: {y_train.mean():.4f}  Val pos rate: {y_val.mean():.4f}")

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    params = {**LGBM_PARAMS, "seed": args.seed}
    model = lgb.train(params, lgb_train, num_boost_round=NUM_BOOST_ROUND, valid_sets=[lgb_val], valid_names=["val"], callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS)])

    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_val_pred)
    print(f"Validation ROC-AUC: {auc:.6f}")
    print(f"Best iteration: {model.best_iteration}")

    model.save_model(args.model_out)
    print(f"Saved model -> {args.model_out}")

    fi = pd.DataFrame({"feature": feature_cols, "importance_gain": model.feature_importance(importance_type="gain"), "importance_split": model.feature_importance(importance_type="split")}).sort_values("importance_gain", ascending=False)
    fi.to_csv(args.fi_out, index=False)
    print(f"Saved feature importance -> {args.fi_out}")

    metrics = {"validation_auc": float(auc), "best_iteration": int(model.best_iteration), "n_train": int(len(X_train)), "n_val": int(len(X_val)), "pos_rate_train": float(y_train.mean()), "pos_rate_val": float(y_val.mean()), "params": params, "feature_count": int(len(feature_cols))}
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
