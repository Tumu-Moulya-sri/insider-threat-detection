import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

FEATURE_COLS = [
    "off_hours_logins_zscore", "files_accessed_zscore", "files_downloaded_zscore",
    "data_transferred_mb_zscore", "usb_connections_zscore", "emails_sent_zscore",
    "emails_external_zscore", "unique_devices_zscore", "unique_locations_zscore"
]

def train_model(input_path="data/features.csv", model_path="models/isolation_forest.pkl"):
    df = pd.read_csv(input_path)
    X = df[FEATURE_COLS]

    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(X)

    df["anomaly_score_raw"] = model.decision_function(X)
    df["is_anomaly"] = model.predict(X)

    min_s, max_s = df["anomaly_score_raw"].min(), df["anomaly_score_raw"].max()
    df["anomaly_score"] = 100 * (max_s - df["anomaly_score_raw"]) / (max_s - min_s)

    joblib.dump(model, model_path)
    df.to_csv("data/scored.csv", index=False)
    print("Model trained and saved to", model_path)

if __name__ == "__main__":
    train_model()