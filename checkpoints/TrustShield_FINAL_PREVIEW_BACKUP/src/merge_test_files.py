# -*- coding: utf-8 -*-
"""Merge test_input.csv with test_ground_truth.csv to produce a full test CSV
required for evaluation scripts that expect the ground truth column.
"""
import os
import pandas as pd

def main():
    base_dir = os.path.join("data", "evaluation")
    test_input_path = os.path.join(base_dir, "test_input.csv")
    gt_path = os.path.join(base_dir, "test_ground_truth.csv")
    out_path = os.path.join(base_dir, "test_full.csv")

    if not os.path.exists(test_input_path) or not os.path.exists(gt_path):
        raise FileNotFoundError("Test input or ground truth files not found.")

    test_input = pd.read_csv(test_input_path)
    gt = pd.read_csv(gt_path)
    # Merge on available identifier columns (event_id, user_id, date)
    merge_keys = [col for col in ["event_id", "user_id", "date"] if col in test_input.columns and col in gt.columns]
    merged = pd.merge(test_input, gt, on=merge_keys, how="left")
    merged.to_csv(out_path, index=False)
    print(f"Merged test file written to {out_path} (rows={len(merged)})")

if __name__ == "__main__":
    main()
