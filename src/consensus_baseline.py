"""
consensus_baseline.py  (formerly dynamic_risk_engine.py)

Runs the three-model consensus vote independently, producing
data/consensus_features.csv for comparison against the main
escalation_engine.py results.

This is NOT part of the production pipeline — run it separately
to generate the "consensus-vote baseline" row in your results table.

Usage:
    python src/consensus_baseline.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURE_COLS = [
    "off_hours_logins_zscore", "files_accessed_zscore", "files_downloaded_zscore",
    "data_transferred_mb_zscore", "usb_connections_zscore", "emails_sent_zscore",
    "emails_external_zscore", "unique_devices_zscore", "unique_locations_zscore",
]


def compute_consensus_detection(df, save_models=False, is_training=False):
    """
    Computes 3-model consensus detection across Isolation Forest, Autoencoder, and Statistical Z-Score features.
    Adds model votes (vote_iforest, vote_autoencoder, vote_statistical), anomaly scores, consensus_votes, 
    consensus_confidence, and consensus_alert to the DataFrame.
    
    Leakage-safe: Models and AE reconstruction threshold (95th percentile) are computed on TRAIN data
    and saved to models/. Inference loads the fixed TRAIN-derived scaler, models, and threshold.
    """
    feature_cols = [c for c in df.columns if c.endswith('_zscore') and not c.endswith('_peer_zscore')]
    if not feature_cols:
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        
    X = df[feature_cols].fillna(0).values

    models_exist = (
        os.path.exists("models/scaler.pkl") and 
        os.path.exists("models/iforest.pkl") and 
        os.path.exists("models/autoencoder.pkl")
    )

    if is_training or not models_exist:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Model 1 — Isolation Forest
        iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        iforest.fit(X_scaled)
        
        # Model 2 — Autoencoder (MLP reconstruction)
        ae = MLPRegressor(hidden_layer_sizes=(6, 3, 6), activation="relu",
                          max_iter=500, random_state=42)
        ae.fit(X_scaled, X_scaled)
        recon_err_train = np.mean((X_scaled - ae.predict(X_scaled)) ** 2, axis=1)
        ae.threshold_ = float(np.percentile(recon_err_train, 95))

        if save_models or is_training:
            os.makedirs("models", exist_ok=True)
            joblib.dump(iforest, "models/iforest.pkl")
            joblib.dump(ae, "models/autoencoder.pkl")
            joblib.dump(scaler, "models/scaler.pkl")
    else:
        scaler = joblib.load("models/scaler.pkl")
        iforest = joblib.load("models/iforest.pkl")
        ae = joblib.load("models/autoencoder.pkl")
        X_scaled = scaler.transform(X)

    # Model 1 — Isolation Forest predictions
    if_votes = (iforest.predict(X_scaled) == -1).astype(int)
    if_raw = iforest.decision_function(X_scaled)
    min_s, max_s = if_raw.min(), if_raw.max()
    if_score = 100 * (max_s - if_raw) / (max_s - min_s + 1e-9)

    # Model 2 — Autoencoder predictions using TRAIN fixed threshold
    recon_err = np.mean((X_scaled - ae.predict(X_scaled)) ** 2, axis=1)
    ae_threshold = getattr(ae, 'threshold_', float(np.percentile(recon_err, 95)))
    ae_votes = (recon_err > ae_threshold).astype(int)
    min_e, max_e = recon_err.min(), recon_err.max()
    ae_score = 100 * (recon_err - min_e) / (max_e - min_e + 1e-9)

    # Model 3 — Statistical Z-score threshold
    stat_dev = np.max(np.abs(X), axis=1)
    stat_votes = (stat_dev > 3.0).astype(int)
    stat_score = np.clip((stat_dev / 5.0) * 100, 0, 100)

    df["vote_iforest"] = if_votes
    df["vote_autoencoder"] = ae_votes
    df["vote_statistical"] = stat_votes
    
    df["if_anomaly_score"] = np.round(if_score, 2)
    df["ae_anomaly_score"] = np.round(ae_score, 2)
    df["stat_anomaly_score"] = np.round(stat_score, 2)

    df["consensus_votes"] = df["vote_iforest"] + df["vote_autoencoder"] + df["vote_statistical"]
    df["consensus_confidence"] = np.round(df["consensus_votes"] / 3.0, 3)
    df["consensus_alert"] = (df["consensus_votes"] >= 2).astype(int)

    return df


def run_consensus_baseline(
    input_path="data/features_shrinkage.csv",
    output_path="data/consensus_features.csv",
):
    df = pd.read_csv(input_path)
    df = compute_consensus_detection(df, save_models=True)
    df.to_csv(output_path, index=False)
    print(f"Consensus baseline written to {output_path}")
    print(df["consensus_votes"].value_counts().to_string())
    return df


if __name__ == "__main__":
    run_consensus_baseline()

