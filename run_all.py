#!/usr/bin/env python3
"""
run_all.py
----------
Run the full discount sensitivity pipeline end-to-end.

Usage:
  python run_all.py --input path/to/raw.csv

Steps:
  1) Prepare data
  2) Train model
  3) Score data
  4) Bucketize scores
"""

import argparse
import subprocess
import sys


def run(cmd: list):
    print("\n>>", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to raw input CSV")
    return p.parse_args()


def main():
    args = parse_args()

    run([
        "python", "scripts/01_prepare_data.py",
        "--input", args.input,
        "--out", "data/df_model_clean.parquet",
        "--metadata_out", "artifacts/metadata.json",
    ])

    run([
        "python", "scripts/02_train_model.py",
        "--data", "data/df_model_clean.parquet",
    ])

    run([
        "python", "scripts/03_score.py",
        "--input", args.input,
        "--model", "artifacts/model.txt",
        "--metadata", "artifacts/metadata.json",
        "--out", "data/scores.csv",
    ])

    run([
        "python", "scripts/04_bucketize.py",
        "--scores", "data/scores.csv",
        "--out", "data/scores_bucketized.csv",
    ])

    print("\n✅ Pipeline finished successfully")


if __name__ == "__main__":
    main()
