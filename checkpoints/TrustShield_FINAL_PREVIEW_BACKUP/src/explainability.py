import pandas as pd
import numpy as np

# pyrefly: ignore [missing-import]
import shap
from sklearn.ensemble import RandomForestClassifier

DEFAULT_FEATURE_COLS = [
    "off_hours_logins_zscore", "files_accessed_zscore", "files_downloaded_zscore",
    "data_transferred_mb_zscore", "usb_connections_zscore", "emails_sent_zscore",
    "emails_external_zscore", "unique_devices_zscore", "unique_locations_zscore"
]

def generate_explanations_with_reliability(input_path_or_df="data/escalated.csv", output_path="data/escalated.csv"):
    if isinstance(input_path_or_df, pd.DataFrame):
        df = input_path_or_df.copy()
    else:
        df = pd.read_csv(input_path_or_df)

    
    # Dynamically detect feature z-score columns present in the input DataFrame
    feature_cols = [c for c in df.columns if c.endswith("_zscore")]
    if not feature_cols:
        # Fall back to DEFAULT_FEATURE_COLS that exist in the DataFrame
        feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]
    
    if not feature_cols:
        # No feature columns available; cannot compute explanations.
        print("No feature columns found for SHAP explanations. Skipping explanation generation.")
        df["evidence_chain_explanation"] = "Explanation unavailable due to missing features."
        df["explanation_reliability_score"] = np.nan
        df.to_csv(output_path, index=False)
        return df
        
    X = df[feature_cols].fillna(0)
    
    # We are explaining the escalation decision (Alert/Log = 1, Monitor = 0)
    y_target = df["escalation_decision"].isin(["Alert", "Log"]).astype(int)
    
    unique_classes = np.unique(y_target)

    if len(y_target) == 0:
        print("No records to explain.")
        df["evidence_chain_explanation"] = "No anomaly detected."
        df["explanation_reliability_score"] = 1.0
        if output_path:
            df.to_csv(output_path, index=False)
        return df

    # Check if surrogate has two classes required for genuine SHAP explanations
    if len(unique_classes) >= 2:
        surrogate = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        surrogate.fit(X, y_target)
        
        class_1_idx = int(np.where(surrogate.classes_ == 1)[0][0])
        surrogate_probs = surrogate.predict_proba(X)[:, class_1_idx]
        
        try:
            explainer = shap.TreeExplainer(surrogate)
            shap_values = explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_pos = shap_values[class_1_idx]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_pos = shap_values[:, :, class_1_idx]
            else:
                shap_pos = shap_values
            has_shap = True
        except Exception as e:
            print(f"SHAP TreeExplainer failed ({e}); falling back to evidence chain.")
            has_shap = False
            shap_pos = None
    else:
        # Single-class surrogate case: Genuine SHAP is impossible (zero variance)
        has_shap = False
        shap_pos = None
        surrogate_probs = np.ones(len(df)) if (len(unique_classes) == 1 and unique_classes[0] == 1) else np.zeros(len(df))

    explanations = []
    reliability_scores = []
    
    for i, row in df.iterrows():
        action = row["escalation_decision"]
        if action == "Monitor":
            explanations.append("Normal behavior observed. No explanation required.")
            reliability_scores.append(1.0)
            continue
            
        # =====================================================================
        # EXPLANATION RELIABILITY SCORE (SURROGATE AGREEMENT HEURISTIC)
        # =====================================================================
        # WHAT THIS SCORE MEASURES:
        #   This score measures local alignment between a shallow TreeExplainer
        #   surrogate model and the main Risk Engine's decision.
        # WHAT THIS SCORE DOES NOT PROVE:
        #   It does NOT prove ground-truth causal truth or absolute mathematical
        #   correctness of the explanation. A score of 1.0 means the surrogate
        #   surrogates the engine's flag with 100% confidence.
        # =====================================================================
        surrogate_prob = surrogate_probs[i] if i < len(surrogate_probs) else 1.0
        
        # Empirical decision target: Alert expects ~1.0, Log expects ~0.50 threshold
        target_prob = 1.0 if action == "Alert" else 0.50
        
        # Faithfulness heuristic = 1.0 - absolute divergence from target probability
        faithfulness = 1.0 - abs(surrogate_prob - target_prob)
        reliability_scores.append(float(np.clip(faithfulness, 0.1, 1.0)))
        
        # Generate Evidence Chain
        exp_parts = []
        if row.get("security_tampering_flag", 0) == 1:
            exp_parts.append("[CRITICAL CONTEXT] Security tampering detected (e.g., log deletion).")
            
        if row.get("in_resignation_notice", 0) == 1:
            exp_parts.append("[CONTEXT] User is in resignation notice period.")
            
        # Behavior drivers: Genuine SHAP if available, otherwise baseline deviation fallback
        if has_shap and shap_pos is not None:
            contributions = list(zip(feature_cols, shap_pos[i]))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            top_reasons = [
                f"{f.replace('_zscore', '')} (SHAP impact: {v:+.2f})" 
                for f, v in contributions[:3] if abs(v) > 0.05
            ]
            if top_reasons:
                exp_parts.append("[BEHAVIOR] Primary drivers: " + "; ".join(top_reasons) + ".")
            else:
                z_devs = [(f, row[f]) for f in feature_cols if f in row and abs(row[f]) > 0.5]
                z_devs.sort(key=lambda x: abs(x[1]), reverse=True)
                top_devs = [f"{f.replace('_zscore', '')} (z-score: {v:+.2f})" for f, v in z_devs[:3]]
                if top_devs:
                    exp_parts.append("[BEHAVIOR] Primary baseline deviations: " + "; ".join(top_devs) + ".")
        else:
            # Fallback when SHAP is unavailable (single-class or SHAP failure):
            z_devs = [(f, row[f]) for f in feature_cols if f in row and abs(row[f]) > 0.5]
            z_devs.sort(key=lambda x: abs(x[1]), reverse=True)
            top_devs = [f"{f.replace('_zscore', '')} (z-score: {v:+.2f})" for f, v in z_devs[:3]]
            if top_devs:
                exp_parts.append("[BEHAVIOR] Primary baseline deviations: " + "; ".join(top_devs) + ".")

        exp_parts.append(f"[ASSET] Target resource classification: {row.get('max_resource_classification', 1)}.")
        
        # Multi-model consensus context
        if "consensus_confidence" in row:
            exp_parts.append(f"[CONSENSUS] {int(row['consensus_confidence']*3)}/3 models flagged anomaly.")
            
        explanations.append(" ".join(exp_parts))
        
    df["evidence_chain_explanation"] = explanations
    df["explanation_reliability_score"] = reliability_scores
    
    if output_path:
        df.to_csv(output_path, index=False)
    print("SHAP explanations and Reliability Scores added.")
    return df

if __name__ == "__main__":
    generate_explanations_with_reliability()