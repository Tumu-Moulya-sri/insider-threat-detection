# -*- coding: utf-8 -*-
"""Evaluate the merged test set (test_full.csv) for TrustShield.
This script loads the merged test CSV containing both features and ground truth,
computes confusion matrix and binary classification metrics, and prints a concise report.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ---- Configuration ----
TEST_CSV = Path("data/evaluation/test_full.csv")
# ---- End configuration ----

def main():
    if not TEST_CSV.is_file():
        sys.exit(f"[ERROR] Test file not found: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    required_cols = {"user_id", "date", "is_insider_ground_truth", "escalation_decision"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns in test file: {missing}")
    if df.isnull().any().any():
        sys.exit("[ERROR] NaN values detected in test file")
    y_true = df["is_insider_ground_truth"].astype(int)
    y_pred = (df["escalation_decision"] == "Alert").astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_used = tn + fp + fn + tp
    if total_used != len(df):
        sys.exit(f"[ERROR] Confusion‑matrix sum ({total_used}) does not equal number of rows ({len(df)})")
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    print("=== TRUSTSHIELD TEST EVALUATION (merged) ===")
    print(f"Test file                : {TEST_CSV}")
    print(f"Rows                     : {len(df)}")
    print(f"Unique users             : {df['user_id'].nunique()}")
    dates = pd.to_datetime(df['date'])
    print(f"Date range               : {dates.min().date()} .. {dates.max().date()}")
    print("\n--- Confusion matrix (TN, FP, FN, TP) ---")
    print(f"TN = {tn}\nFP = {fp}\nFN = {fn}\nTP = {tp}")
    print("\n--- Metrics ---")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1        = {f1:.4f}")
    print(f"FPR       = {fpr:.4f}")
    print(f"FNR       = {fnr:.4f}")

if __name__ == "__main__":
    main()
