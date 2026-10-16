import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib

# Ensure src directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentiment_analysis import add_sentiment
from shrinkage_baseline import apply_shrinkage_baseline
from peer_analysis import add_peer_deviation
from consensus_baseline import compute_consensus_detection
from risk_scoring import compute_risk
from escalation_engine import train_and_escalate
from explainability import generate_explanations_with_reliability
from spedia_pipeline import run_spedia_pipeline


def run_v2_pipeline(input_path="data/evaluation/test_input.csv", output_dir="data"):
    """
    Executes the true inference-only TrustShield UEBA detection pipeline on V2 synthetic benchmark:
      Stage 1: feature preparation (Raw Logs & Linguistic/Sentiment Analysis)
      Stage 2: warm-start baseline (Adaptive EWMA Shrinkage from Training State)
      Stage 3: peer analysis (Cohort Department-Level Deviations)
      Stage 4: model inference (Loaded Pre-Trained Scaler, IF, AE with Fixed TRAIN Threshold)
      Stage 5: consensus (3-Model Anomaly Consensus Engine)
      Stage 6: risk/context (Fused Risk Engine & Temporal Accumulation)
      Stage 7: escalation (Consensus-Gated Production Decision Rule)
      Stage 8: XAI (SHAP & Evidence Chain Explanations)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 75)
    print("TRUSTSHIELD UEBA PRODUCTION INFERENCE ENGINE (V2 BENCHMARK)")
    print("=" * 75)
    print("MODEL MODE:           INFERENCE ONLY")
    print("BASELINE MODE:        WARM START (models/training_baseline_state.pkl)")
    print("AE THRESHOLD SOURCE:  TRAIN SAVED STATE (0.870013)")
    print("DECISION MODE:        CONSENSUS GATED")
    print(f"OUTPUT DIRECTORY:     {output_dir}")
    print("=" * 75)

    # 1. Verify Input File
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file {input_path} not found.")
        sys.exit(1)

    print(f"\n[Stage 1: feature preparation] Reading input logs from {input_path}...")
    raw_df = pd.read_csv(input_path)
    total_records = len(raw_df)
    unique_users = raw_df["user_id"].nunique()
    print(f"Loaded {total_records} activity records for {unique_users} unique users.")

    # Handle comm_text vs message_text schema compatibility
    if "comm_text" not in raw_df.columns and "message_text" in raw_df.columns:
        raw_df["comm_text"] = raw_df["message_text"]

    temp_raw = os.path.join(output_dir, "temp_raw_with_sentiment.csv")
    raw_df.to_csv(temp_raw, index=False)
    df_sent = add_sentiment(input_path=temp_raw, output_path=temp_raw)

    # 2. Warm-Start Baseline
    print("\n[Stage 2: warm-start baseline] Computing Adaptive EWMA Shrinkage from saved training state...")
    features_path = os.path.join(output_dir, "trustshield_features.csv")
    state_file = "models/training_baseline_state.pkl" if os.path.exists("models/training_baseline_state.pkl") else None
    features_df = apply_shrinkage_baseline(
        input_path=temp_raw,
        output_path=features_path,
        disable_novelty_1=False,
        baseline_state_path=state_file
    )
    if os.path.exists(temp_raw):
        os.remove(temp_raw)
    print(f"Saved -> {features_path}")

    # 3. Peer Analysis
    print("\n[Stage 3: peer analysis] Calculating peer department-level baseline deviations...")
    features_df = add_peer_deviation(input_path=features_path, output_path=features_path)

    # 4. Model Inference & 5. Consensus
    print("\n[Stage 4: model inference] Loading pre-trained Scaler, IF, and AE models...")
    print("[Stage 5: consensus] Evaluating 3-model anomaly consensus with fixed TRAIN threshold...")
    detection_df = compute_consensus_detection(features_df.copy(), save_models=False, is_training=False)
    detection_path = os.path.join(output_dir, "trustshield_detection.csv")
    detection_df.to_csv(detection_path, index=False)
    print(f"Saved -> {detection_path}")

    # 6. Risk & Context
    print("\n[Stage 6: risk/context] Fusing consensus scores with peer, rule, and temporal risk decay...")
    risk_path = os.path.join(output_dir, "trustshield_risk.csv")
    risk_df = compute_risk(input_path_or_df=detection_df, output_path=risk_path)
    print(f"Saved -> {risk_path}")

    # 7. Escalation (Consensus-Gated Decision Rule)
    print("\n[Stage 7: escalation] Executing Consensus-Gated Escalation Production Rule...")
    alerts_path = os.path.join(output_dir, "trustshield_alerts.csv")
    
    escalated_df = train_and_escalate(input_path_or_df=risk_df, output_path=None, disable_novelty_2=False)
    
    def apply_consensus_gated_rule(row):
        if row.get("security_tampering_flag", 0) == 1:
            return "Alert"
        cv = int(row.get("consensus_votes", 0))
        if cv >= 2:
            return "Alert"
        elif cv == 1:
            return "Log"
        else:
            return "Monitor"

    escalated_df["escalation_decision"] = escalated_df.apply(apply_consensus_gated_rule, axis=1)
    escalated_df.to_csv(alerts_path, index=False)
    print(f"Saved -> {alerts_path}")

    # 8. XAI
    print("\n[Stage 8: XAI] Generating SHAP explanations & Evidence Chain Reliability...")
    try:
        final_df = generate_explanations_with_reliability(input_path_or_df=escalated_df, output_path=alerts_path)
    except Exception as e:
        print(f"Explainability fallback ({e}), generating standard evidence chains.")
        final_df = escalated_df
        final_df.to_csv(alerts_path, index=False)

    print(f"Saved final annotated results -> {alerts_path}")

    # Terminal summary reporting
    alert_counts = final_df["escalation_decision"].value_counts().to_dict()
    alerts = alert_counts.get("Alert", 0)
    logs = alert_counts.get("Log", 0)
    monitors = alert_counts.get("Monitor", 0)

    vote_counts = final_df["consensus_votes"].value_counts().to_dict()
    v3 = vote_counts.get(3, 0)
    v2 = vote_counts.get(2, 0)
    v1 = vote_counts.get(1, 0)
    v0 = vote_counts.get(0, 0)

    print("\n" + "=" * 75)
    print("V2 BENCHMARK INFERENCE EXECUTION SUMMARY")
    print("=" * 75)
    print(f"Input records:         {total_records}")
    print(f"Unique Users:          {unique_users}")
    print(f"Alert Actions:         {alerts} ({alerts/total_records*100:.2f}%)")
    print(f"Log Actions:           {logs} ({logs/total_records*100:.2f}%)")
    print(f"Monitor Actions:       {monitors} ({monitors/total_records*100:.2f}%)")
    print("-" * 75)
    print(f"Consensus Breakdown:")
    print(f"  3/3 votes (Unanimous Anomaly): {v3}")
    print(f"  2/3 votes (Majority Anomaly):  {v2}")
    print(f"  1/3 votes (Minority Anomaly):  {v1}")
    print(f"  0/3 votes (Unanimous Normal):  {v0}")
    print("=" * 75)
    print("PIPELINE EXECUTION COMPLETE.")
    print("=" * 75)

    return final_df


def main():
    parser = argparse.ArgumentParser(description="TrustShield UEBA Production Inference Runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default="spedia",
        choices=["spedia", "v2"],
        help="Dataset pipeline to execute: 'spedia' (Primary Live Telemetry) or 'v2' (Synthetic Benchmark)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input raw activity logs CSV (optional override)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save pipeline outputs (defaults to data/demo/spedia or data/demo)"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=None,
        help="Directory containing pre-trained models"
    )
    args = parser.parse_args()

    if args.dataset == "spedia":
        raw_path = args.input or "data/spedia dataset/logs_SPEDIA_annotated_en.csv"
        out_dir = args.output_dir or "data/demo/spedia"
        m_dir = args.models_dir or "models/spedia"
        run_spedia_pipeline(raw_path=raw_path, output_dir=out_dir, models_dir=m_dir)
    else:
        in_path = args.input or "data/evaluation/test_input.csv"
        out_dir = args.output_dir or "data/demo"
        run_v2_pipeline(input_path=in_path, output_dir=out_dir)


if __name__ == "__main__":
    main()
