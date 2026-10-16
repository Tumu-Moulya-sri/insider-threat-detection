import pandas as pd

METRICS = [
    "off_hours_logins", "files_downloaded", "data_transferred_mb",
    "usb_connections", "emails_external"
]

def add_peer_deviation(input_path="data/features.csv", output_path="data/features.csv"):
    df = pd.read_csv(input_path)

    for col in METRICS:
        peer_stats = df.groupby(["department", "date"])[col].agg(["mean", "std"]).reset_index()
        peer_stats.columns = ["department", "date", f"{col}_peer_mean", f"{col}_peer_std"]
        df = df.merge(peer_stats, on=["department", "date"], how="left")
        df[f"{col}_peer_zscore"] = (
            (df[col] - df[f"{col}_peer_mean"]) / df[f"{col}_peer_std"].replace(0, 1)
        ).fillna(0)

    peer_cols = [f"{c}_peer_zscore" for c in METRICS]
    df["peer_deviation_score"] = df[peer_cols].abs().mean(axis=1)

    df.to_csv(output_path, index=False)
    print("Peer-cohort deviation added.")
    return df

if __name__ == "__main__":
    add_peer_deviation()