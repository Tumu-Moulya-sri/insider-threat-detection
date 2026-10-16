import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

STATES  = ["Normal", "Drift", "Suspicious", "Exfiltration"]
ACTIONS = ["Monitor", "Log", "Alert"]

# Cost Matrix — rows = actions, columns = true states (Normal, Drift, Susp, Exfil)
BASE_COST_MATRIX = np.array([
    [ 0.0,  1.0,  5.0, 20.0 ],  # Monitor
    [ 1.0,  0.5,  2.0, 10.0 ],  # Log
    [ 5.0,  3.0,  0.5,  0.0 ],  # Alert
])

def define_pseudo_labels(row):
    """
    Derives 4-class stage labels (0: Normal, 1: Drift, 2: Suspicious, 3: Exfiltration)
    using purely unsupervised behavioral anomaly signals, 3-model consensus votes, and security flags.
    Does NOT use any ground-truth target columns.
    """
    z_cols = [c for c in row.index if c.endswith("_zscore")]
    max_z = max(abs(row.get(c, 0)) for c in z_cols) if z_cols else 0.0
    consensus_votes = row.get("consensus_votes", 0)

    # Exfiltration signal: severe anomaly in data transfer, downloads, security tampering, or unanimous consensus
    exfil_signal = (
        row.get("data_transferred_mb_zscore", 0) > 3.0 or 
        row.get("files_downloaded_zscore", 0) > 3.0 or 
        row.get("security_tampering_flag", 0) == 1 or
        consensus_votes == 3
    )
    
    if exfil_signal or max_z > 3.5:
        return 3  # Exfiltration
    elif consensus_votes >= 2 or max_z > 2.5:
        return 2  # Suspicious
    elif consensus_votes == 1 or max_z > 1.5:
        return 1  # Drift
    return 0  # Normal

def train_and_escalate(input_path_or_df=None, input_path="data/features_shrinkage.csv", output_path="data/escalated.csv", disable_novelty_2=False):
    if input_path_or_df is not None:
        if isinstance(input_path_or_df, pd.DataFrame):
            df = input_path_or_df.copy()
        else:
            df = pd.read_csv(input_path_or_df)
    else:
        df = pd.read_csv(input_path)

    # ── 1. Pseudo-labels (Unsupervised) ───────────────────────────────────────
    if "stage_label" not in df.columns:
        df["stage_label"] = df.apply(define_pseudo_labels, axis=1)

    features = [c for c in df.columns if c.endswith("_zscore")]
    # Include consensus detection scores if available
    extra_feats = [c for c in ["consensus_votes", "vote_iforest", "vote_autoencoder", "vote_statistical"] if c in df.columns]
    feature_set = features + extra_feats

    X = df[feature_set].fillna(0)
    y = df["stage_label"]


    # ── 2. Behavioral Classifier ──────────────────────────────────────────────
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    raw_probs = model.predict_proba(X)
    
    # Map to guaranteed (N, 4) probability shape across 4 states
    probs = np.zeros((len(df), 4))
    for idx_c, cls in enumerate(model.classes_):
        if 0 <= cls < 4:
            probs[:, int(cls)] = raw_probs[:, idx_c]

    # ── 3. Evidential Uncertainty (Dirichlet surrogate) ───────────────────────
    K = 4
    evidence  = probs * model.n_estimators   # treat vote counts as evidence
    alpha     = evidence + 1
    S         = np.sum(alpha, axis=1, keepdims=True)
    belief    = evidence / S                 # normalized belief (N×K)
    uncertainty = (K / S).squeeze()

    # Persist model + beliefs so evaluate_system.py can load them exactly
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/behavioral_model.pkl")
    # NOTE: Save beliefs after the final escalated CSV to guarantee row order alignment.
    # The DataFrame `df` is not reordered after this point, so saving now ensures
    # `beliefs.npy` matches the rows written to `data/escalated.csv`.
    # beliefs.npy will be saved after CSV to ensure row order sync

    # ── 4. Context-Aware Expected-Cost Minimisation ───────────────────────────
    decisions          = []
    expected_costs_list = []

    for i, row in df.iterrows():
        b = belief[i]

        # Asset sensitivity (scales miss costs superlinearly)
        c_level = row.get("max_resource_classification", 1)

        # Organisational context multiplier
        risk_multiplier = 1.0
        if row.get("in_resignation_notice",       0) == 1: risk_multiplier += 0.50
        if row.get("has_open_disciplinary_case",  0) == 1: risk_multiplier += 0.50
        if row.get("recent_audit_failure",        0) == 1: risk_multiplier += 0.30

        # Role-based multiplier — admins are higher stakes on exfil
        if row.get("role") == "administrator":
            risk_multiplier *= 1.20

        dynamic_cost = BASE_COST_MATRIX.copy()
        dynamic_cost[:, 2] *= (c_level ** 1.5)   # Suspicious miss
        dynamic_cost[:, 3] *= (c_level ** 2.0)   # Exfil miss

        expected_costs = dynamic_cost.dot(b) * risk_multiplier

        # Hard override: security-tampering always forces an Alert
        if row.get("security_tampering_flag", 0) == 1:
            best_action = 2
            expected_cost = 0.0
        elif disable_novelty_2:
            # Expected anomaly score
            expected_anomaly_score = np.dot(b, [0.0, 1.0, 2.0, 3.0])
            if expected_anomaly_score >= 1.5:
                best_action = 2 # Alert
            elif expected_anomaly_score >= 0.5:
                best_action = 1 # Log
            else:
                best_action = 0 # Monitor
            expected_cost = expected_anomaly_score
        else:
            best_action = int(np.argmin(expected_costs))
            expected_cost = expected_costs[best_action]

        decisions.append(ACTIONS[best_action])
        expected_costs_list.append(expected_cost)

    df["predicted_stage"]   = [STATES[int(np.argmax(b))] for b in belief]
    df["uncertainty"]        = uncertainty
    df["escalation_decision"] = decisions
    df["expected_cost"]      = expected_costs_list

    # ── 5. Plain-language evidence strings ────────────────────────────────────
    explanations = []
    for i, row in df.iterrows():
        action = df.loc[i, "escalation_decision"]
        if action == "Monitor":
            explanations.append("Normal behavior observed.")
            continue

        parts = []
        if row.get("security_tampering_flag", 0) == 1:
            parts.append("[CRITICAL CONTEXT] User attempted security tampering (e.g., log deletion).")
        if row.get("in_resignation_notice", 0) == 1:
            parts.append("[CONTEXT] User is in resignation notice period.")

        deviations = [
            f"{f.replace('_zscore','')} deviated {row[f]:.1f}σ from peer-baseline"
            for f in features if abs(row.get(f, 0)) > 2.5
        ]
        if deviations:
            parts.append("[BEHAVIOR] " + "; ".join(deviations) + ".")

        parts.append(f"[ASSET] Max resource classification: {row.get('max_resource_classification', 1)}.")
        parts.append(f"[CONFIDENCE] Uncertainty: {df.loc[i, 'uncertainty']:.2f}.")

        explanations.append(" ".join(parts))

    df["evidence_chain_explanation"] = explanations

    df.to_csv(output_path, index=False)
    # Save beliefs after CSV to keep synchronization between files
    np.save("data/beliefs.npy", belief)
    print(f"Escalation engine complete. Saved to {output_path} (Probability matrix shape: {probs.shape})")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run escalation engine with specified input and output CSV files.")
    parser.add_argument("--input_path", type=str, default="data/features_shrinkage.csv", help="Path to input feature CSV.")
    parser.add_argument("--output_path", type=str, default="data/escalated.csv", help="Path to output escalated CSV.")
    parser.add_argument("--disable_novelty_2", action="store_true", help="Disable novelty 2 behavior.")
    args = parser.parse_args()
    train_and_escalate(input_path=args.input_path, output_path=args.output_path, disable_novelty_2=args.disable_novelty_2)

