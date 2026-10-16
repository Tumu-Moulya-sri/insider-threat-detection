import pandas as pd
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "off_hours_logins_zscore", "files_accessed_zscore", "files_downloaded_zscore",
    "data_transferred_mb_zscore", "usb_connections_zscore", "emails_sent_zscore",
    "emails_external_zscore", "unique_devices_zscore", "unique_locations_zscore"
]

def train_autoencoder(input_path="data/features.csv", output_path="data/features.csv"):
    df = pd.read_csv(input_path)
    X = df[FEATURE_COLS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Autoencoder-style reconstruction network: input -> bottleneck -> output
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(6, 3, 6),
        activation="relu",
        max_iter=500,
        random_state=42
    )
    autoencoder.fit(X_scaled, X_scaled)

    reconstructed = autoencoder.predict(X_scaled)
    reconstruction_error = np.mean((X_scaled - reconstructed) ** 2, axis=1)

    df["ae_reconstruction_error"] = reconstruction_error
    min_e, max_e = reconstruction_error.min(), reconstruction_error.max()
    df["ae_anomaly_score"] = 100 * (reconstruction_error - min_e) / (max_e - min_e + 1e-9)

    joblib.dump(autoencoder, "models/autoencoder.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    df.to_csv(output_path, index=False)
    print("Autoencoder trained. Reconstruction-error anomaly scores added.")
    return df

if __name__ == "__main__":
    train_autoencoder()