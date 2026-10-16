import os
import subprocess
import pandas as pd
import numpy as np

def run_experiments():
    seeds = [42, 43, 44, 45, 46]
    
    metrics = []
    
    for seed in seeds:
        print(f"\n======================================")
        print(f"Running full pipeline for seed {seed}")
        print(f"======================================")
        
        # 1. Generate Data
        subprocess.run(["python", "data/generate_data.py", f"--seed={seed}"], check=True)
        
        # 2. Adaptive Behaviour Baseline + Fingerprint export
        subprocess.run(["python", "src/shrinkage_baseline.py"], check=True)

        # 3. 3‑Model Consensus Detection (adds votes and scores)
        subprocess.run(["python", "src/consensus_baseline.py"], check=True)

        # 4. Escalation Engine – now consumes consensus_features.csv
        subprocess.run(["python", "src/escalation_engine.py", "--input_path=data/consensus_features.csv", "--output_path=data/escalated.csv"], check=True)

        # 4. Explainable AI + Reliability Score
        subprocess.run(["python", "src/explainability.py"], check=True)

        # 6. Extract Metrics
        df = pd.read_csv("data/escalated.csv")
        active_insiders = df[df["is_insider_ground_truth"] == 1]
        
        # Get alert rate for standard exfiltration
        std_exfil = active_insiders[active_insiders["insider_scenario"] == "standard_exfiltration"]
        std_alerts = len(std_exfil[std_exfil["escalation_decision"] == "Alert"])
        std_rate = std_alerts / max(1, len(std_exfil))
        
        # Get alert rate for slow subtle insider
        slow_exfil = active_insiders[active_insiders["insider_scenario"] == "slow_subtle_insider"]
        slow_alerts = len(slow_exfil[slow_exfil["escalation_decision"] == "Alert"])
        slow_rate = slow_alerts / max(1, len(slow_exfil))
        
        # Get false positive rate (alerts on benign activity)
        benign = df[df["is_insider_ground_truth"] == 0]
        fp_alerts = len(benign[benign["escalation_decision"] == "Alert"])
        fp_rate = fp_alerts / max(1, len(benign))
        
        metrics.append({
            "Seed": seed,
            "Std_Exfil_Alert_Rate": std_rate,
            "Slow_Subtle_Alert_Rate": slow_rate,
            "False_Positive_Rate": fp_rate
        })
        
    print("\n\n======================================")
    print("Multi-Seed Variance Results")
    print("======================================")
    
    res_df = pd.DataFrame(metrics)
    print(res_df.to_string(index=False))
    
    print("\nSummary Statistics:")
    print(f"Std Exfil Alert Rate:   {res_df['Std_Exfil_Alert_Rate'].mean():.3f} +/- {res_df['Std_Exfil_Alert_Rate'].std():.3f}")
    print(f"Slow Subtle Alert Rate: {res_df['Slow_Subtle_Alert_Rate'].mean():.3f} +/- {res_df['Slow_Subtle_Alert_Rate'].std():.3f}")
    print(f"False Positive Rate:    {res_df['False_Positive_Rate'].mean():.3f} +/- {res_df['False_Positive_Rate'].std():.3f}")

if __name__ == "__main__":
    run_experiments()
