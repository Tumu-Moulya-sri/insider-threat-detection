import os
import sys
import copy
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Features used for behavioral anomaly scoring in SPEDIA
SPEDIA_BEHAVIORAL_METRICS = [
    "total_events",
    "off_hours_events",
    "num_logins",
    "login_failed",
    "num_logoffs",
    "session_events",
    "commands_executed",
    "unique_commands",
    "high_level_commands",
    "files_accessed",
    "files_modified",
    "files_added",
    "files_deleted",
    "total_file_events",
    "emails_sent",
    "external_emails",
    "total_attachments",
    "total_email_size_kb",
    "http_requests",
    "unique_urls",
    "usb_connects",
    "usb_disconnects",
    "device_events"
]

def compute_spedia_adaptive_baseline(df, is_training=False, state_path=None, save_state_path=None, alpha=0.2):
    """
    Computes chronological per-user EWMA adaptive baseline on SPEDIA features.
    Strictly leakage-safe: Each record is scored BEFORE its observation updates the baseline.
    """
    df = df.sort_values(["date", "user_id"]).reset_index(drop=True).copy()
    
    # Global priors computed only from training data if training, or loaded from state
    if state_path and os.path.exists(state_path):
        saved = joblib.load(state_path)
        user_history = copy.deepcopy(saved["user_history"])
        global_priors = copy.deepcopy(saved["global_priors"])
    else:
        global_priors = {}
        for m in SPEDIA_BEHAVIORAL_METRICS:
            if m in df.columns:
                mean_val = float(df[m].mean())
                var_val = float(df[m].var())
                if np.isnan(var_val) or var_val == 0:
                    var_val = 1.0
                global_priors[m] = {"mean": mean_val, "var": var_val}
        user_history = {}

    zscore_rows = []
    
    for idx, row in df.iterrows():
        u = row["user_id"]
        if u not in user_history:
            # Initialize with global prior
            user_history[u] = {
                m: {"mean": global_priors[m]["mean"], "var": global_priors[m]["var"], "count": 0}
                for m in SPEDIA_BEHAVIORAL_METRICS if m in df.columns
            }
        
        row_zscores = {}
        for m in SPEDIA_BEHAVIORAL_METRICS:
            if m not in df.columns:
                continue
            val = float(row[m])
            cur_mean = user_history[u][m]["mean"]
            cur_var = user_history[u][m]["var"]
            std = np.sqrt(max(cur_var, 1e-4))
            
            # Score BEFORE update
            z = (val - cur_mean) / std
            row_zscores[f"{m}_zscore"] = float(z)
            
            # Update EWMA state
            new_mean = (1 - alpha) * cur_mean + alpha * val
            new_var = (1 - alpha) * cur_var + alpha * ((val - new_mean) ** 2)
            user_history[u][m]["mean"] = new_mean
            user_history[u][m]["var"] = max(new_var, 1e-4)
            user_history[u][m]["count"] += 1
            
        zscore_rows.append(row_zscores)
        
    z_df = pd.DataFrame(zscore_rows)
    for col in z_df.columns:
        df[col] = z_df[col]

    if save_state_path:
        os.makedirs(os.path.dirname(save_state_path), exist_ok=True)
        joblib.dump({"user_history": user_history, "global_priors": global_priors}, save_state_path)
        print(f"Saved SPEDIA baseline state to {save_state_path} ({len(user_history)} users)")

    return df, user_history, global_priors


def train_spedia_models(features_path="data/external/spedia/spedia_features.csv", models_dir="models/spedia"):
    """
    Trains the SPEDIA-specific model suite on the chronological TRAIN split (<= 2025-03-20).
    Saves:
      - models/spedia/training_baseline_state.pkl
      - models/spedia/scaler.pkl
      - models/spedia/iforest.pkl
      - models/spedia/autoencoder.pkl (with fixed TRAIN threshold)
    """
    os.makedirs(models_dir, exist_ok=True)
    if not os.path.exists(features_path):
        from spedia_adapter import extract_spedia_features
        extract_spedia_features(output_path=features_path)
        
    df = pd.read_csv(features_path)
    dates = sorted(df["date"].unique())
    train_dates = dates[:15] # 60% of 25 dates
    train_df = df[df["date"].isin(train_dates)].copy()
    
    print(f"\n[SPEDIA Training] Chronological Split: 15 Train dates ({len(train_df)} user-day records).")
    
    # 1. Train Adaptive Baseline
    state_path = os.path.join(models_dir, "training_baseline_state.pkl")
    train_feat_df, _, _ = compute_spedia_adaptive_baseline(
        train_df, is_training=True, save_state_path=state_path
    )
    
    zscore_cols = [f"{m}_zscore" for m in SPEDIA_BEHAVIORAL_METRICS if f"{m}_zscore" in train_feat_df.columns]
    X_train = train_feat_df[zscore_cols].fillna(0).values
    
    # 2. Train Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    
    # 3. Train Isolation Forest
    iforest = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
    iforest.fit(X_scaled)
    joblib.dump(iforest, os.path.join(models_dir, "iforest.pkl"))
    
    # 4. Train Autoencoder
    ae = MLPRegressor(
        hidden_layer_sizes=(16, 8, 16),
        activation="relu",
        max_iter=500,
        random_state=42
    )
    ae.fit(X_scaled, X_scaled)
    
    # Fixed TRAIN-derived threshold (95th percentile of TRAIN reconstruction errors)
    recon_err_train = np.mean((X_scaled - ae.predict(X_scaled)) ** 2, axis=1)
    ae.threshold_ = float(np.percentile(recon_err_train, 95))
    joblib.dump(ae, os.path.join(models_dir, "autoencoder.pkl"))
    
    print(f"Trained SPEDIA Scaler, IF, and Autoencoder.")
    print(f"Fixed SPEDIA AE Reconstruction Threshold (95th percentile TRAIN): {ae.threshold_:.6f}")
    return ae.threshold_


def run_spedia_pipeline(
    raw_path=None,
    features_path="data/external/spedia/spedia_features.csv",
    output_dir="data/demo/spedia",
    models_dir="models/spedia"
):
    """
    Executes the end-to-end TrustShield UEBA detection pipeline on SPEDIA:
      1. Preprocess & Feature Extraction
      2. Adaptive Baseline Warm-Start
      3. Anomaly Inference (IF, AE, Stat Detector)
      4. Consensus Gated Escalation
      5. Risk Scoring & Temporal Decay
      6. XAI Explanations & Reliability
      7. Saves canonical outputs to data/demo/spedia/
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 75)
    print("TRUSTSHIELD UEBA — LIVE SPEDIA ANALYSIS PIPELINE")
    print("=" * 75)
    print("DATASET:              SPEDIA (Multi-source Linux/Network/Host Telemetry)")
    print("BASELINE MODE:        CHRONOLOGICAL WARM-START EWMA")
    print(f"MODELS DIRECTORY:     {models_dir}")
    print(f"OUTPUT DIRECTORY:     {output_dir}")
    print("=" * 75)
    
    # Ensure features exist
    if not os.path.exists(features_path) or raw_path is not None:
        from spedia_adapter import extract_spedia_features
        extract_spedia_features(input_path=raw_path, output_path=features_path)
        
    # Ensure models exist
    state_file = os.path.join(models_dir, "training_baseline_state.pkl")
    scaler_file = os.path.join(models_dir, "scaler.pkl")
    iforest_file = os.path.join(models_dir, "iforest.pkl")
    ae_file = os.path.join(models_dir, "autoencoder.pkl")
    
    if not (os.path.exists(state_file) and os.path.exists(scaler_file) and os.path.exists(iforest_file) and os.path.exists(ae_file)):
        print("\n[Step 0: Training] SPEDIA model state missing. Training SPEDIA models on TRAIN split...")
        train_spedia_models(features_path=features_path, models_dir=models_dir)
        
    df = pd.read_csv(features_path)
    total_records = len(df)
    unique_users = df["user_id"].nunique()
    print(f"\n[Stage 1: Ingestion] Loaded {total_records} daily user records across {unique_users} actors.")

    # 1. Compute Chronological Adaptive Baseline
    print("\n[Stage 2: Adaptive Baseline] Propagating chronological per-user EWMA baseline...")
    features_df, _, _ = compute_spedia_adaptive_baseline(
        df, is_training=False, state_path=state_file
    )
    features_output = os.path.join(output_dir, "trustshield_features.csv")
    features_df.to_csv(features_output, index=False)
    print(f"Saved -> {features_output}")

    # 2. Model Inference
    print("\n[Stage 3: Inference] Loading pre-trained SPEDIA Scaler, IF, and AE models...")
    scaler = joblib.load(scaler_file)
    iforest = joblib.load(iforest_file)
    ae = joblib.load(ae_file)
    
    zscore_cols = [f"{m}_zscore" for m in SPEDIA_BEHAVIORAL_METRICS if f"{m}_zscore" in features_df.columns]
    X = features_df[zscore_cols].fillna(0).values
    X_scaled = scaler.transform(X)

    # Model 1 — Isolation Forest
    if_raw = iforest.decision_function(X_scaled)
    min_s, max_s = if_raw.min(), if_raw.max()
    if_score = 100 * (max_s - if_raw) / (max_s - min_s + 1e-9)
    vote_iforest = (iforest.predict(X_scaled) == -1).astype(int)

    # Model 2 — Autoencoder with FIXED TRAIN THRESHOLD
    recon_err = np.mean((X_scaled - ae.predict(X_scaled)) ** 2, axis=1)
    ae_thresh = getattr(ae, 'threshold_', float(np.percentile(recon_err, 95)))
    vote_autoencoder = (recon_err > ae_thresh).astype(int)
    min_e, max_e = recon_err.min(), recon_err.max()
    ae_score = 100 * (recon_err - min_e) / (max_e - min_e + 1e-9)

    # Model 3 — Statistical Z-Score Detector
    stat_dev = np.max(np.abs(X), axis=1)
    vote_statistical = (stat_dev > 3.0).astype(int)
    stat_score = np.clip((stat_dev / 5.0) * 100, 0, 100)

    # Detection DataFrame
    detection_df = features_df.copy()
    detection_df["vote_iforest"] = vote_iforest
    detection_df["vote_autoencoder"] = vote_autoencoder
    detection_df["vote_statistical"] = vote_statistical
    detection_df["if_anomaly_score"] = np.round(if_score, 2)
    detection_df["ae_anomaly_score"] = np.round(ae_score, 2)
    detection_df["stat_anomaly_score"] = np.round(stat_score, 2)
    detection_df["ae_reconstruction_error"] = np.round(recon_err, 6)

    # 3. Consensus Engine
    print("\n[Stage 4: Consensus] Computing 3-model anomaly consensus...")
    detection_df["consensus_votes"] = (
        detection_df["vote_iforest"] + 
        detection_df["vote_autoencoder"] + 
        detection_df["vote_statistical"]
    )
    detection_df["consensus_confidence"] = np.round(detection_df["consensus_votes"] / 3.0, 3)
    detection_df["consensus_alert"] = (detection_df["consensus_votes"] >= 2).astype(int)
    
    detection_output = os.path.join(output_dir, "trustshield_detection.csv")
    detection_df.to_csv(detection_output, index=False)
    print(f"Saved -> {detection_output}")

    # 4. Risk Engine & Temporal Decay
    print("\n[Stage 5: Risk & Context] Fusing consensus, statistical severity, and temporal risk decay...")
    risk_df = detection_df.copy()
    
    # Daily Risk Score (0 - 100)
    base_risk = (
        0.40 * (risk_df["consensus_confidence"] * 100) +
        0.30 * risk_df["stat_anomaly_score"] +
        0.15 * risk_df["if_anomaly_score"] +
        0.15 * risk_df["ae_anomaly_score"]
    )
    
    # Tampering bump
    tamper_bump = risk_df["security_tampering_flag"].fillna(0) * 25.0
    risk_df["daily_risk"] = np.round(np.clip(base_risk + tamper_bump, 0, 100), 2)
    
    # Temporal Cumulative Risk (per user exponential decay across days)
    risk_df = risk_df.sort_values(["user_id", "date"]).reset_index(drop=True)
    decay = 0.85
    cum_risk = []
    user_running_risk = {}
    
    for _, row in risk_df.iterrows():
        u = row["user_id"]
        d_risk = row["daily_risk"]
        prev = user_running_risk.get(u, 0.0)
        curr = d_risk + decay * prev
        user_running_risk[u] = curr
        cum_risk.append(round(curr, 2))
        
    risk_df["cumulative_risk"] = cum_risk
    risk_output = os.path.join(output_dir, "trustshield_risk.csv")
    risk_df.to_csv(risk_output, index=False)
    print(f"Saved -> {risk_output}")

    # 5. Consensus-Gated Escalation Engine
    print("\n[Stage 6: Escalation] Enforcing Production Consensus-Gated Escalation Rule...")
    def apply_spedia_escalation(row):
        if row.get("security_tampering_flag", 0) == 1:
            return "Alert"
        cv = int(row.get("consensus_votes", 0))
        if cv >= 2:
            return "Alert"
        elif cv == 1:
            return "Log"
        else:
            return "Monitor"

    risk_df["escalation_decision"] = risk_df.apply(apply_spedia_escalation, axis=1)

    # 6. Explainable AI (XAI)
    print("\n[Stage 7: XAI] Generating Human-Readable Attributions & Explanations...")
    explanations = []
    recommendations = []
    top_reasons = []
    reliability_scores = []
    
    # Fit surrogate on Z-scores for SHAP if multi-class
    y_target = (risk_df["consensus_votes"] >= 2).astype(int)
    use_shap = (y_target.nunique() > 1)
    
    shap_vals = None
    if use_shap:
        try:
            import shap
            surrogate = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
            surrogate.fit(X, y_target)
            explainer = shap.TreeExplainer(surrogate)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list) and len(shap_vals) > 1:
                shap_vals = shap_vals[1]
        except Exception:
            shap_vals = None

    for i, row in risk_df.iterrows():
        action = row["escalation_decision"]
        cv = int(row["consensus_votes"])
        d_risk = float(row["daily_risk"])
        tamper = int(row.get("security_tampering_flag", 0))
        
        # Determine top deviating features
        row_z = {m: abs(float(row.get(f"{m}_zscore", 0))) for m in SPEDIA_BEHAVIORAL_METRICS if f"{m}_zscore" in row}
        sorted_feats = sorted(row_z.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_feats[:3]
        
        reasons_list = [f"{feat} (Z={score:.1f})" for feat, score in top_3 if score > 1.5]
        if not reasons_list:
            reasons_list = [f"{sorted_feats[0][0]} (Z={sorted_feats[0][1]:.1f})"] if sorted_feats else ["Nominal baseline activity"]
        
        top_reasons_str = "; ".join(reasons_list)
        top_reasons.append(top_reasons_str)
        
        # Build narrative explanation
        if tamper == 1:
            exp = f"SECURITY OVERRIDE: High-level privilege command manipulation or authentication abuse detected. Escalated immediately."
            rec = "Immediate SOC Analyst Review & Endpoint Containment"
            rel = 0.95
        elif action == "Alert":
            exp = f"Majority detector consensus ({cv}/3 votes) flagged anomalous activity driven by {top_reasons_str}. Daily Risk score: {d_risk:.1f}/100."
            rec = "Prioritize for Active Threat Investigation & User Access Review"
            rel = 0.90 if cv == 3 else 0.85
        elif action == "Log":
            exp = f"Single detector signal ({cv}/3 votes) observed with moderate deviation in {top_reasons_str}. Recorded for temporal monitoring."
            rec = "Log to Telemetry & Continue Behavioral Baseline Tracking"
            rel = 0.80
        else:
            exp = f"Nominal user activity consistent with established behavioral baseline (0/3 detector votes). Daily Risk: {d_risk:.1f}/100."
            rec = "Routine Automated Monitoring"
            rel = 0.95

        explanations.append(exp)
        recommendations.append(rec)
        reliability_scores.append(rel)

    risk_df["top_behavioral_reasons"] = top_reasons
    risk_df["explanation"] = explanations
    risk_df["recommended_action"] = recommendations
    risk_df["xai_reliability_score"] = reliability_scores

    alerts_output = os.path.join(output_dir, "trustshield_alerts.csv")
    risk_df.to_csv(alerts_output, index=False)
    print(f"Saved -> {alerts_output}")

    # 7. Summary & Offline Validation Stats
    action_counts = risk_df["escalation_decision"].value_counts().to_dict()
    alerts = action_counts.get("Alert", 0)
    logs = action_counts.get("Log", 0)
    monitors = action_counts.get("Monitor", 0)

    dates = sorted(risk_df["date"].unique())
    train_dates = dates[:15]
    val_dates = dates[15:20]
    test_dates = dates[20:]

    test_df = risk_df[risk_df["date"].isin(test_dates)].copy()
    
    print("\n" + "=" * 75)
    print("SPEDIA TRUSTSHIELD INFERENCE SUMMARY")
    print("=" * 75)
    print(f"Total User-Day Records: {total_records}")
    print(f"Unique Actors:          {unique_users}")
    print(f"Date Coverage:          {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print(f"Alert Actions:          {alerts} ({alerts/total_records*100:.2f}%)")
    print(f"Log Actions:            {logs} ({logs/total_records*100:.2f}%)")
    print(f"Monitor Actions:        {monitors} ({monitors/total_records*100:.2f}%)")
    print("-" * 75)
    
    # Offline Research Evaluation on Test Split (if ground truth available)
    if "spedia_is_malicious_day" in test_df.columns:
        y_test_true = test_df["spedia_is_malicious_day"].values
        y_test_pred = (test_df["escalation_decision"] == "Alert").astype(int).values
        
        tn, fp, fn, tp = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1]).ravel()
        prec = precision_score(y_test_true, y_test_pred, zero_division=0)
        rec = recall_score(y_test_true, y_test_pred, zero_division=0)
        f1 = f1_score(y_test_true, y_test_pred, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        print("RESEARCH EVALUATION (TEST SPLIT ONLY):")
        print(f"  Test Records: {len(test_df)}")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
        print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  FPR:       {fpr:.4f}, FNR: {fnr:.4f}")
    print("=" * 75)
    
    return risk_df

if __name__ == "__main__":
    run_spedia_pipeline()
