# -*- coding: utf-8 -*-
"""Chronological Train/Validation/Test Split for TrustShield

This script creates a deterministic 60/20/20 split of the synthetic logs
based on chronological order (date) and stable user ordering.  The split
is performed on the raw source `data/synthetic_logs.csv`.

Outputs (all placed under `data/evaluation/`):
* `train.csv`          – rows used for model fitting and threshold derivation
* `val.csv`            – rows for validation (optional downstream use)
* `test_input.csv`    – feature rows for the held‑out test set (no ground‑truth column)
* `test_ground_truth.csv` – ground‑truth labels for the test set (only `event_id` and `is_insider_ground_truth`)

The script also prints the date boundaries for each split so the user can
audit that no future information leaks into the training period.
"""

import os
import pandas as pd

def main():
    # Paths
    raw_path = os.path.join("data", "synthetic_logs.csv")
    out_dir = os.path.join("data", "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    # Load raw data
    df = pd.read_csv(raw_path)
    # Ensure deterministic ordering: sort by date then user_id then event_id if present
    df["date"] = pd.to_datetime(df["date"])
    sort_cols = ["date", "user_id"]
    if "event_id" in df.columns:
        sort_cols.append("event_id")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    total = len(df)
    train_end = int(0.60 * total)  # 60%
    val_end = train_end + int(0.20 * total)  # next 20%

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    # Save raw splits (optional – useful for debugging)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test_raw.csv"), index=False)

    # Produce test_input (all columns except the insider ground‑truth)
    gt_col = "is_insider_ground_truth"
    if gt_col in test.columns:
        test_input = test.drop(columns=[gt_col])
    else:
        test_input = test.copy()
    test_input.to_csv(os.path.join(out_dir, "test_input.csv"), index=False)

    # Produce test_ground_truth (only identifiers + ground‑truth)
    id_cols = []
    for cand in ["event_id", "user_id", "date"]:
        if cand in test.columns:
            id_cols.append(cand)
    gt_df = test[[gt_col] + id_cols]
    gt_df.to_csv(os.path.join(out_dir, "test_ground_truth.csv"), index=False)

    # Report split boundaries
    print("--- Chronological Split Summary ---")
    print(f"Total rows          : {total}")
    print(f"Train rows (60%)    : {len(train)}  – dates {train['date'].min().date()} to {train['date'].max().date()}")
    print(f"Val rows (20%)      : {len(val)}   – dates {val['date'].min().date()} to {val['date'].max().date()}")
    print(f"Test rows (20%)     : {len(test)}   – dates {test['date'].min().date()} to {test['date'].max().date()}")
    print("Outputs written to:")
    print(f"  {out_dir}/train.csv")
    print(f"  {out_dir}/val.csv")
    print(f"  {out_dir}/test_input.csv")
    print(f"  {out_dir}/test_ground_truth.csv")

if __name__ == "__main__":
    main()
