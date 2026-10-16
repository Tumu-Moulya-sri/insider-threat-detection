import pandas as pd
import numpy as np

def engineer_features(input_path="data/synthetic_logs.csv", output_path="data/features.csv"):
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["user_id", "date"])

    numeric_cols = [
        "off_hours_logins", "files_accessed", "files_downloaded",
        "data_transferred_mb", "usb_connections", "emails_sent",
        "emails_external", "unique_devices", "unique_locations"
    ]

    for col in numeric_cols:
        df[f"{col}_roll_mean"] = (
            df.groupby("user_id")[col]
            .transform(lambda x: x.shift(1).rolling(window=14, min_periods=5).mean())
        )
        df[f"{col}_roll_std"] = (
            df.groupby("user_id")[col]
            .transform(lambda x: x.shift(1).rolling(window=14, min_periods=5).std())
        )
        df[f"{col}_zscore"] = (
            (df[col] - df[f"{col}_roll_mean"]) / df[f"{col}_roll_std"].replace(0, np.nan)
        )

    zscore_cols = [f"{c}_zscore" for c in numeric_cols]
    roll_mean_cols = [f"{c}_roll_mean" for c in numeric_cols]

    # Only drop rows where there's genuinely not enough history yet (first ~13 days per user)
    df = df.dropna(subset=roll_mean_cols)

    # For rows with a valid history but zero variance (std=0), zscore is NaN — treat as "no deviation"
    df[zscore_cols] = df[zscore_cols].fillna(0)

    df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Saved {len(df)} rows to {output_path}")
    return df

if __name__ == "__main__":
    engineer_features()