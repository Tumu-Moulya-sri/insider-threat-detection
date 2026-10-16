import pandas as pd
import numpy as np

def evaluate_cost_sensitivity(input_path="data/escalated.csv"):
    df = pd.read_csv(input_path)
    print("\n--- Cost Sensitivity Check ---")
    
    BASE_COST_MATRIX = np.array([
        [ 0.0,   1.0,   5.0,  20.0 ],
        [ 1.0,   0.5,   2.0,  10.0 ],
        [ 5.0,   3.0,   0.5,   0.0 ]
    ])
    
    import os
    if not os.path.exists("data/beliefs.npy"):
        print("Beliefs not found. Please run 'python src/escalation_engine.py' first.")
        return

    belief = np.load("data/beliefs.npy")
    
    def get_decisions(cost_matrix_modifier_func):
        decisions = []
        for i, row in df.iterrows():
            b = belief[i]
            c_level = row.get("max_resource_classification", 1)
            
            risk_multiplier = 1.0
            if row.get("in_resignation_notice", 0) == 1: risk_multiplier += 0.5
            if row.get("has_open_disciplinary_case", 0) == 1: risk_multiplier += 0.5
            if row.get("recent_audit_failure", 0) == 1: risk_multiplier += 0.3
            
            dynamic_cost_matrix = cost_matrix_modifier_func(BASE_COST_MATRIX.copy())
            dynamic_cost_matrix[:, 2] *= (c_level ** 1.5) 
            dynamic_cost_matrix[:, 3] *= (c_level ** 2.0) 
            
            expected_costs = dynamic_cost_matrix.dot(b) * risk_multiplier
            
            if row.get("security_tampering_flag", 0) == 1:
                best_action = 2
            else:
                best_action = np.argmin(expected_costs)
            decisions.append(best_action)
        return decisions

    # Base alerts
    base_decisions = get_decisions(lambda x: x)
    base_alerts = sum(1 for d in base_decisions if d == 2)
    print(f"Base Configuration Alerts: {base_alerts}")
    
    # +25% False Alarm Cost (Shift costs of Alerting/Logging on Normal/Drift)
    def shift_up(matrix):
        matrix[1, 0:2] *= 1.25 # Log false alarm
        matrix[2, 0:2] *= 1.25 # Alert false alarm
        return matrix
        
    up_decisions = get_decisions(shift_up)
    up_alerts = sum(1 for d in up_decisions if d == 2)
    print(f"Tested False-Alarm Cost Shift (+25%): Alerts changed from {base_alerts} to {up_alerts}")

    # -25% False Alarm Cost
    def shift_down(matrix):
        matrix[1, 0:2] *= 0.75
        matrix[2, 0:2] *= 0.75
        return matrix
        
    down_decisions = get_decisions(shift_down)
    down_alerts = sum(1 for d in down_decisions if d == 2)
    print(f"Tested False-Alarm Cost Shift (-25%): Alerts changed from {base_alerts} to {down_alerts}")


def evaluate_insider_scenarios(input_path="data/escalated.csv"):
    df = pd.read_csv(input_path)
    print("\n--- Insider Scenario Detection Comparison ---")
    
    active_insiders = df[df["is_insider_ground_truth"] == 1]
    grouped = active_insiders.groupby("insider_scenario")
    
    results = []
    for scenario, group in grouped:
        total_days = len(group)
        alerts = len(group[group["escalation_decision"] == "Alert"])
        logs = len(group[group["escalation_decision"] == "Log"])
        monitors = len(group[group["escalation_decision"] == "Monitor"])
        
        results.append({
            "Scenario": scenario,
            "Total_Active_Days": total_days,
            "Alerts": alerts,
            "Alert_Rate": alerts / max(1, total_days)
        })
        
    res_df = pd.DataFrame(results).sort_values("Alert_Rate", ascending=False)
    print(res_df.to_string(index=False))

def evaluate_calibration_and_workload(input_path="data/escalated.csv"):
    df = pd.read_csv(input_path)
    print("\n--- Additional Evaluation Metrics ---")
    
    # 1. Analyst Workload Metric
    total_days = df["date"].nunique()
    total_alerts = (df["escalation_decision"] == "Alert").sum()
    total_logs = (df["escalation_decision"] == "Log").sum()
    
    avg_alerts_per_day = total_alerts / total_days
    avg_logs_per_day = total_logs / total_days
    print(f"Analyst Workload: {avg_alerts_per_day:.2f} Alerts/day | {avg_logs_per_day:.2f} Logs/day")
    
    # 2. Detection Latency
    if "is_insider_ground_truth" in df.columns:
        active_insiders = df[df["is_insider_ground_truth"] == 1]
        latencies = []
        
        for user, group in active_insiders.groupby("user_id"):
            first_active_day = group["day_index"].min()
            
            alerts = group[group["escalation_decision"] == "Alert"]
            if len(alerts) > 0:
                first_alert_day = alerts["day_index"].min()
                latencies.append({
                    "user_id": user,
                    "scenario": group.get("insider_scenario", pd.Series(["unknown"]*len(group))).iloc[0],
                    "latency_days": first_alert_day - first_active_day
                })
                
        if latencies:
            lat_df = pd.DataFrame(latencies)
            avg_latency = lat_df.groupby("scenario")["latency_days"].mean().reset_index()
            print("\nDetection Latency (Days from activation to first Alert):")
            print(avg_latency.to_string(index=False))
        
    # 3. Expected Calibration Error (ECE)
    import os
    if os.path.exists("data/beliefs.npy"):
        belief = np.load("data/beliefs.npy")
        if len(belief) != len(df):
            raise ValueError("beliefs.npy row count does not match escalated.csv -- re-run escalation_engine.py before evaluating")
            return
        # belief is shape (N, K)
        # We need the ground truth labels to check accuracy
        try:
            from src.escalation_engine import define_pseudo_labels
        except ModuleNotFoundError:
            from escalation_engine import define_pseudo_labels
        if "stage_label" not in df.columns:
            df["stage_label"] = df.apply(define_pseudo_labels, axis=1)
            
        y_true = df["stage_label"].values
        confidences = np.max(belief, axis=1)
        predictions = np.argmax(belief, axis=1)
        
        accuracies = (predictions == y_true)
        
        n_bins = 10
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        
        ece = 0.0
        print("\nCalibration Check (ECE across 10 bins):")
        for i in range(n_bins):
            bin_lower = bins[i]
            bin_upper = bins[i+1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        print(f"Expected Calibration Error (ECE): {ece:.4f}")
    else:
        print("\nCalibration Check skipped: beliefs.npy not found.")

if __name__ == "__main__":
    evaluate_cost_sensitivity()
    evaluate_insider_scenarios()
    evaluate_calibration_and_workload()
