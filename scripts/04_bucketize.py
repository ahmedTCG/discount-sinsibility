#!/usr/bin/env python3
"""
04_bucketize.py - Assign customers to segments based on scores.
"""

import argparse
import pandas as pd

from config import SEGMENT_THRESHOLDS, SEGMENT_LABELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True, help="Input scores CSV")
    p.add_argument("--out", required=True, help="Output bucketized CSV")
    p.add_argument("--t1", type=float, default=SEGMENT_THRESHOLDS[1])
    p.add_argument("--t2", type=float, default=SEGMENT_THRESHOLDS[2])
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.scores)

    if "score" not in df.columns or "externalcustomerkey" not in df.columns:
        raise ValueError("Input must contain columns: externalcustomerkey, score")

    bins = [0.0, args.t1, args.t2, 1.0]
    df["segment"] = pd.cut(df["score"], bins=bins, labels=SEGMENT_LABELS, include_lowest=True, right=True)

    df.to_csv(args.out, index=False)

    print(f"Saved -> {args.out}")
    print("\nSegment distribution:")
    print(df["segment"].value_counts(dropna=False))
    print("\nSegment share:")
    print(df["segment"].value_counts(normalize=True, dropna=False))


if __name__ == "__main__":
    main()
