#!/usr/bin/env python3
"""
04_bucketize.py
---------------
Bucketize model scores into segments.

Default buckets:
  0.00–0.20 -> full_price
  0.20–0.60 -> conditional
  0.60–1.00 -> discount_driven

Inputs:
- scores CSV with columns: externalcustomerkey, score

Outputs:
- bucketized CSV with added column: segment
- prints segment counts + shares

Usage:
  python scripts/04_bucketize.py --scores data/scores_B.csv --out data/scores_B_bucketized.csv
"""

import argparse
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True, help="Input scores CSV (must contain externalcustomerkey, score)")
    p.add_argument("--out", required=True, help="Output bucketized CSV")
    p.add_argument("--t1", type=float, default=0.2, help="Threshold 1 (default 0.2)")
    p.add_argument("--t2", type=float, default=0.6, help="Threshold 2 (default 0.6)")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.scores)

    if "score" not in df.columns or "externalcustomerkey" not in df.columns:
        raise ValueError("Input must contain columns: externalcustomerkey, score")

    bins = [0.0, args.t1, args.t2, 1.0]
    labels = ["full_price", "conditional", "discount_driven"]

    df["segment"] = pd.cut(
        df["score"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True
    )

    df.to_csv(args.out, index=False)

    print("Saved ->", args.out)
    print("\nSegment distribution:")
    print(df["segment"].value_counts(dropna=False))
    print("\nSegment share:")
    print(df["segment"].value_counts(normalize=True, dropna=False))


if __name__ == "__main__":
    main()
