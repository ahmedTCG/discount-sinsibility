#!/usr/bin/env python3
"""
run_all.py - Run the full discount sensitivity pipeline.
"""

import argparse
import subprocess
import sys

sys.path.insert(0, "scripts")
from config import DEFAULT_PATHS


def run(cmd):
    print("\n>>", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to raw input CSV")
    p.add_argument("--skip-train", action="store_true", help="Skip training, use existing model")
    return p.parse_args()


def main():
    args = parse_args()

    run(["python", "scripts/01_prepare_data.py", "--input", args.input, "--out", DEFAULT_PATHS["data_clean"], "--metadata_out", DEFAULT_PATHS["metadata"]])

    if not args.skip_train:
        run(["python", "scripts/02_train_model.py", "--data", DEFAULT_PATHS["data_clean"], "--metadata", DEFAULT_PATHS["metadata"], "--model_out", DEFAULT_PATHS["model"], "--fi_out", DEFAULT_PATHS["feature_importance"], "--metrics_out", DEFAULT_PATHS["metrics"]])

    run(["python", "scripts/03_score.py", "--input", args.input, "--model", DEFAULT_PATHS["model"], "--metadata", DEFAULT_PATHS["metadata"], "--out", DEFAULT_PATHS["scores"]])

    run(["python", "scripts/04_bucketize.py", "--scores", DEFAULT_PATHS["scores"], "--out", DEFAULT_PATHS["scores_bucketized"]])

    print("\n✅ Pipeline finished successfully")


if __name__ == "__main__":
    main()
