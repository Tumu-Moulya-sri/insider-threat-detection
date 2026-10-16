import os
import subprocess
import pandas as pd
import numpy as np
import sys

# Add src to python path if not present to allow local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shrinkage_baseline import apply_shrinkage_baseline
from escalation_engine import train_and_escalate

def calculate_metrics(df_path):
    df = pd.read_csv(df_path)
    
    if "is_insider_ground_truth" not in df.columns:
        print(f"Warning: is_insider_ground_truth not found in {df_path}")
        return 0.0, 0.0, 0.0
        
    active = df[df["is_insider_ground_truth"] == 1]
    benign = df[df["is_insider_ground_truth"] == 0]
    
    # 1. Detection Recall (percentage of active insider days flagged as "Alert")
    total_active_days = len(active)
    active_alerts = (active["escalation_decision"] == "Alert").sum()
    recall = active_alerts / max(1, total_active_days)
    
    # 2. False Positive Rate (percentage of benign days flagged as "Alert")
    total_benign_days = len(benign)
    benign_alerts = (benign["escalation_decision"] == "Alert").sum()
    fpr = benign_alerts / max(1, total_benign_days)
    
    # 3. Detection Latency (average days from activation to first Alert)
    latencies = []
    for user, group in active.groupby("user_id"):
        first_active_day = group["day_index"].min()
        alerts = group[group["escalation_decision"] == "Alert"]
        if len(alerts) > 0:
            first_alert_day = alerts["day_index"].min()
            latencies.append(first_alert_day - first_active_day)
            
    avg_latency = np.mean(latencies) if latencies else 0.0
    return recall, fpr, avg_latency

def main():
    # Make sure we have synthetic logs
    if not os.path.exists("data/synthetic_logs.csv"):
        print("Generating synthetic logs...")
        subprocess.run(["python", "data/generate_data.py", "--seed=42"], check=True)
        
    os.makedirs("data", exist_ok=True)
    
    results = []
    
    # a) Full System
    print("\n--- Running Variant A: Full System ---")
    apply_shrinkage_baseline("data/synthetic_logs.csv", "data/features_shrinkage.csv", disable_novelty_1=False)
    train_and_escalate("data/features_shrinkage.csv", "data/escalated.csv", disable_novelty_2=False)
    rec, fpr, lat = calculate_metrics("data/escalated.csv")
    results.append({
        "Approach": "Full System (As-Is)",
        "Detection Recall": rec,
        "False Positive Rate": fpr,
        "Detection Latency": lat
    })
    
    # b) Novelty 1 Disabled
    print("\n--- Running Variant B: Novelty 1 Disabled (Static prior baseline) ---")
    apply_shrinkage_baseline("data/synthetic_logs.csv", "data/features_shrinkage_nov1.csv", disable_novelty_1=True)
    train_and_escalate("data/features_shrinkage_nov1.csv", "data/escalated_nov1.csv", disable_novelty_2=False)
    rec, fpr, lat = calculate_metrics("data/escalated_nov1.csv")
    results.append({
        "Approach": "Novelty 1 Disabled (Static Peer Baseline)",
        "Detection Recall": rec,
        "False Positive Rate": fpr,
        "Detection Latency": lat
    })
    
    # c) Novelty 2 Disabled
    print("\n--- Running Variant C: Novelty 2 Disabled (Flat threshold) ---")
    # Reuse full system features_shrinkage.csv to keep baseline as-is
    train_and_escalate("data/features_shrinkage.csv", "data/escalated_nov2.csv", disable_novelty_2=True)
    rec, fpr, lat = calculate_metrics("data/escalated_nov2.csv")
    results.append({
        "Approach": "Novelty 2 Disabled (Flat Threshold Escalation)",
        "Detection Recall": rec,
        "False Positive Rate": fpr,
        "Detection Latency": lat
    })
    
    # Save comparison table
    res_df = pd.DataFrame(results)
    res_df.to_csv("data/ablation_results.csv", index=False)
    print("\nAblation study complete. Results:")
    print(res_df.to_string(index=False))
    
if __name__ == "__main__":
    main()
