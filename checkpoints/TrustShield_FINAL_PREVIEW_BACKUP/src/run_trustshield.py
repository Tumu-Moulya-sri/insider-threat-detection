import os
import sys
import argparse
import pandas as pd
import numpy as np

# Ensure src directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentiment_analysis import add_sentiment
from shrinkage_baseline import apply_shrinkage_baseline
from peer_analysis import add_peer_deviation
from consensus_baseline import compute_consensus_detection
from risk_scoring import compute_risk
from escalation_engine import train_and_escalate
from explainability import generate_explanations_with_reliability

def run_trustshield_pipeline(input_path="data/synthetic_logs.csv", output_dir="data"):
    """
    Executes the unified 7-stage TrustShield UEBA detection pipeline:
      Raw Logs -> Sentiment -> Adaptive EWMA Shrinkage -> Peer Analysis -> 
      3-Model Consensus (IF, AE, Stat) -> Fused Risk Engine -> 
      Evidential Escalation (Context & Cost) -> XAI (SHAP & Reliability) -> Dashboard Ready
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("STARTING TRUSTSHIELD UEBA UNIFIED PIPELINE EXECUTION")
    print("=" * 70)

    # 1. Verify Input File
    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found. Auto-generating synthetic data...")
        import subprocess
        subprocess.run(["python", "data/generate_data.py", "--seed=42"], check=True)
        input_path = "data/synthetic_logs.csv"

    print(f"\n[Stage 1/7] Reading input logs: {input_path}")
    raw_df = pd.read_csv(input_path)
    total_records = len(raw_df)
    unique_users = raw_df["user_id"].nunique()
    print(f"Loaded {total_records} activity records for {unique_users} unique users.")

    # 2. Sentiment Analysis
    print("\n[Stage 2/7] Running Sentiment Analysis & Resignation Intent Detection...")
    df = add_sentiment(input_path=input_path, output_path=None)

    # 3. Adaptive Baseline & Empirical Bayes Shrinkage
    print("\n[Stage 3/7] Computing Adaptive EWMA Baselines & Shrinkage Z-Scores...")
    temp_raw = os.path.join(output_dir, "temp_raw_with_sentiment.csv")
    df.to_csv(temp_raw, index=False)
    features_df = apply_shrinkage_baseline(
        input_path=temp_raw, 
        output_path=os.path.join(output_dir, "trustshield_features.csv")
    )
    if os.path.exists(temp_raw):
        os.remove(temp_raw)
    print(f"Saved Stage 3 output -> {os.path.join(output_dir, 'trustshield_features.csv')}")

    # 4. Peer Cohort Analysis
    print("\n[Stage 4/7] Computing Peer Group Department-Level Deviations...")
    temp_feat = os.path.join(output_dir, "trustshield_features.csv")
    features_df = add_peer_deviation(input_path=temp_feat, output_path=temp_feat)

    # 5. Three-Model Consensus Anomaly Detection
    print("\n[Stage 5/7] Executing 3-Model Consensus Anomaly Detection (IForest, Autoencoder, Stat Z-Score)...")
    detection_df = compute_consensus_detection(features_df, save_models=True)
    detection_path = os.path.join(output_dir, "trustshield_detection.csv")
    detection_df.to_csv(detection_path, index=False)
    print(f"Saved Stage 5 output -> {detection_path}")

    # 6. Multi-Signal Fused Risk Scoring & Temporal Decay
    print("\n[Stage 6/7] Fusing Consensus Scores with Peer, Rule, Sentiment, and Temporal Risk Decay...")
    risk_path = os.path.join(output_dir, "trustshield_risk.csv")
    risk_df = compute_risk(input_path_or_df=detection_df, output_path=risk_path)
    print(f"Saved Stage 6 output -> {risk_path}")

    # 7. Evidential Decision Escalation Engine (Context & Cost Minimization)
    print("\n[Stage 7/7] Running Dirichlet Evidential Escalation Engine & Cost Minimization...")
    alerts_path = os.path.join(output_dir, "trustshield_alerts.csv")
    escalated_df = train_and_escalate(input_path_or_df=risk_df, output_path=alerts_path, disable_novelty_2=False)

    # 8. Explainable AI (SHAP attributions & Explanation Reliability Score)
    print("\n[Stage 8/8] Computing SHAP Explanations & Explanation Reliability Scores...")
    try:
        final_df = generate_explanations_with_reliability(input_path_or_df=escalated_df, output_path=alerts_path)
    except Exception as e:
        print(f"Explainability step failed ({e}), proceeding without SHAP explanations.")
        final_df = escalated_df
        final_df.to_csv(alerts_path, index=False)
    
    # Mirror to legacy data/escalated.csv so existing dashboard/evaluation scripts continue to work
    legacy_path = os.path.join(output_dir, "escalated.csv")
    final_df.to_csv(legacy_path, index=False)
    print(f"Mirrored final output to legacy path -> {legacy_path}")

    # =========================================================================
    # TERMINAL SUMMARY REPORTING
    # =========================================================================
    alert_counts = final_df["escalation_decision"].value_counts().to_dict()
    alerts = alert_counts.get("Alert", 0)
    logs = alert_counts.get("Log", 0)
    monitors = alert_counts.get("Monitor", 0)

    risk_counts = final_df["risk_level"].value_counts().to_dict()
    crit = risk_counts.get("Critical", 0)
    high = risk_counts.get("High", 0)
    med = risk_counts.get("Medium", 0)
    low = risk_counts.get("Low", 0)

    vote_counts = final_df["consensus_votes"].value_counts().to_dict()
    v3 = vote_counts.get(3, 0)
    v2 = vote_counts.get(2, 0)
    v1 = vote_counts.get(1, 0)
    v0 = vote_counts.get(0, 0)

    avg_risk = final_df["cumulative_risk_score"].mean()

    print("\n" + "=" * 70)
    print("TRUSTSHIELD UEBA PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Input records:         {total_records}")
    print(f"Users:                 {unique_users}")
    print(f"Alerts:                {alerts}")
    print(f"Log:                   {logs}")
    print(f"Monitor:               {monitors}")
    print("-" * 70)
    print(f"Risk Breakdown (Critical/High/Medium/Low):")
    print(f"  Critical: {crit} | High: {high} | Medium: {med} | Low: {low}")
    print(f"Average risk score:    {avg_risk:.2f}")
    print("-" * 70)
    print(f"3-Model Agreement Distribution:")
    print(f"  3/3 votes (Unanimous Anomaly): {v3}")
    print(f"  2/3 votes (Majority Anomaly):  {v2}")
    print(f"  1/3 votes (Minority Anomaly):  {v1}")
    print(f"  0/3 votes (Unanimous Normal):  {v0}")
    print("=" * 70)

    # Sample tracing: 1 normal user/day vs 1 suspicious user/day
    print("\nSINGLE-DAY RECORD TRACING DEMONSTRATION")
    print("-" * 70)
    normal_sample = final_df[final_df["escalation_decision"] == "Monitor"].head(1)
    susp_sample = final_df[final_df["escalation_decision"] == "Alert"].head(1)

    if len(normal_sample) > 0:
        row_n = normal_sample.iloc[0]
        print(f"[BENIGN USER DAY]")
        print(f"  Employee:             {row_n['user_id']} ({row_n.get('department','N/A')}, Role: {row_n.get('role','N/A')})")
        print(f"  Date:                 {row_n['date']}")
        print(f"  Model Scores:         IF={row_n.get('if_anomaly_score',0)}, AE={row_n.get('ae_anomaly_score',0)}, Stat={row_n.get('stat_anomaly_score',0)}")
        print(f"  Model Votes:          IF={row_n.get('vote_iforest',0)}, AE={row_n.get('vote_autoencoder',0)}, Stat={row_n.get('vote_statistical',0)}")
        print(f"  Consensus:            {row_n.get('consensus_votes',0)}/3 votes (Alert={row_n.get('consensus_alert',0)})")
        print(f"  Context Flags:        SecurityTampering={row_n.get('security_tampering_flag',0)}, Resignation={row_n.get('in_resignation_notice',0)}")
        print(f"  Risk Scores:          Daily={row_n.get('daily_risk_score',0)}, Cumulative={row_n.get('cumulative_risk_score',0)} ({row_n.get('risk_level','Low')})")
        print(f"  Final Decision:       {row_n['escalation_decision']} (Expected Cost={row_n.get('expected_cost',0):.2f})")
        print(f"  Explanation:          {row_n['evidence_chain_explanation']}")
        print(f"  Reliability Score:    {row_n.get('explanation_reliability_score',1.0):.2f}")

    print("-" * 70)
    if len(susp_sample) > 0:
        row_s = susp_sample.iloc[0]
        print(f"[FLAGGED SUSPICIOUS USER DAY]")
        print(f"  Employee:             {row_s['user_id']} ({row_s.get('department','N/A')}, Role: {row_s.get('role','N/A')})")
        print(f"  Date:                 {row_s['date']}")
        print(f"  Model Scores:         IF={row_s.get('if_anomaly_score',0)}, AE={row_s.get('ae_anomaly_score',0)}, Stat={row_s.get('stat_anomaly_score',0)}")
        print(f"  Model Votes:          IF={row_s.get('vote_iforest',0)}, AE={row_s.get('vote_autoencoder',0)}, Stat={row_s.get('vote_statistical',0)}")
        print(f"  Consensus:            {row_s.get('consensus_votes',0)}/3 votes (Alert={row_s.get('consensus_alert',0)})")
        print(f"  Context Flags:        SecurityTampering={row_s.get('security_tampering_flag',0)}, Resignation={row_s.get('in_resignation_notice',0)}")
        print(f"  Risk Scores:          Daily={row_s.get('daily_risk_score',0)}, Cumulative={row_s.get('cumulative_risk_score',0)} ({row_s.get('risk_level','Low')})")
        print(f"  Final Decision:       {row_s['escalation_decision']} (Expected Cost={row_s.get('expected_cost',0):.2f})")
        print(f"  Explanation:          {row_s['evidence_chain_explanation']}")
        print(f"  Reliability Score:    {row_s.get('explanation_reliability_score',1.0):.2f}")

    print("=" * 70)
    print("TRUSTSHIELD UNIFIED PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrustShield UEBA Unified Pipeline Orchestrator")
    parser.add_argument("--input", type=str, default="data/synthetic_logs.csv", help="Path to input raw activity logs CSV")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save pipeline outputs")
    args = parser.parse_args()

    run_trustshield_pipeline(input_path=args.input, output_dir=args.output_dir)
