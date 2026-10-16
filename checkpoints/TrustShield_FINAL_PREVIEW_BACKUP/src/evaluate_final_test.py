# src/evaluate_final_test.py
"""Authoritative evaluation of the held‑out blind test set.

This script:
* loads `data/escalated_blind.csv`
* checks that the file is a proper test set (chronological split, no training dates, no NaNs)
* computes binary classification metrics using the ground‑truth column `is_insider_ground_truth`
  and the prediction column `escalation_decision == "Alert"`.
* validates that the number of rows matches the confusion matrix (TN+FP+FN+TP).
* prints a concise report and also verifies the same numbers with scikit‑learn.
"""
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ---- Configuration ----
TEST_CSV = Path("data/escalated_blind.csv")
# ---- End configuration ----

def main():
    if not TEST_CSV.is_file():
        sys.exit(f"[ERROR] Test file not found: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    # Basic sanity checks
    required_cols = {"user_id", "date", "is_insider_ground_truth", "escalation_decision"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns in test file: {missing}")
    if df.isnull().any().any():
        sys.exit("[ERROR] NaN values detected in test file")
    # Ensure strictly later dates than training period (training ends 2026‑04‑05)
    training_end = pd.to_datetime("2026-04-05")
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.min() <= training_end:
        sys.exit("[ERROR] Test set contains dates before or equal to training end date")
    # Ground truth / prediction vectors
    y_true = df["is_insider_ground_truth"].astype(int)
    y_pred = (df["escalation_decision"] == "Alert").astype(int)
    if len(y_true) != len(y_pred):
        sys.exit("[ERROR] Length mismatch between true and predicted labels")
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_used = tn + fp + fn + tp
    if total_used != len(df):
        sys.exit(f"[ERROR] Confusion‑matrix sum ({total_used}) does not equal number of rows ({len(df)})")
    # Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    # Scenario‑level recall (if scenario column exists)
    scenario_metrics = {}
    if "insider_scenario" in df.columns:
        for scenario in ["standard_exfiltration", "slow_subtle_insider"]:
            mask = df["insider_scenario"] == scenario
            if mask.any():
                tp_s = ((y_true[mask] == 1) & (y_pred[mask] == 1)).sum()
                fn_s = ((y_true[mask] == 1) & (y_pred[mask] == 0)).sum()
                recall_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else 0.0
                scenario_metrics[scenario] = {
                    "positives": int(mask.sum()),
                    "tp": int(tp_s),
                    "fn": int(fn_s),
                    "recall": recall_s,
                }
    # Print report
    print("=== TRUSTSHIELD FINAL TEST EVALUATION ===")
    print(f"Test file                : {TEST_CSV}")
    print(f"Rows                     : {len(df)}")
    print(f"Unique users             : {df['user_id'].nunique()}")
    print(f"Date range               : {dates.min().date()} .. {dates.max().date()}")
    print("\n--- Confusion matrix (TN, FP, FN, TP) ---")
    print(f"TN = {tn}\nFP = {fp}\nFN = {fn}\nTP = {tp}")
    print("\n--- Metrics ---")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1        = {f1:.4f}")
    print(f"FPR       = {fpr:.4f}")
    print(f"FNR       = {fnr:.4f}")
    if scenario_metrics:
        print("\n--- Scenario‑level recall ---")
        for scen, vals in scenario_metrics.items():
            print(f"{scen}: positives={vals['positives']}, TP={vals['tp']}, FN={vals['fn']}, recall={vals['recall']:.4f}")
    print("\nAll checks passed. Evaluation complete.")

if __name__ == "__main__":
    main()
