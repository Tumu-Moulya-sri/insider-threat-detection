import pandas as pd
import numpy as np

def compute_risk(input_path_or_df="data/scored.csv", output_path="data/risk_scored.csv"):
    if isinstance(input_path_or_df, pd.DataFrame):
        df = input_path_or_df.copy()
    else:
        df = pd.read_csv(input_path_or_df)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["user_id", "date"])

    # --- Rule-based signals ---
    df["flag_off_hours"] = (df["off_hours_logins"] >= 2).astype(int) * 10
    df["flag_bulk_download"] = (df["files_downloaded"] >= 20).astype(int) * 15
    df["flag_large_transfer"] = (df["data_transferred_mb"] >= 250).astype(int) * 20
    df["flag_usb"] = (df["usb_connections"] >= 2).astype(int) * 10
    df["flag_external_email"] = (df.get("emails_external", 0) >= 3).astype(int) * 10
    
    res_flag = df.get("resignation_flag", 0)
    df["flag_resignation_language"] = res_flag * 15
    rule_score = df[[c for c in df.columns if c.startswith("flag_")]].sum(axis=1).clip(0, 100)

    # --- Sentiment risk signal ---
    sentiment_comp = df.get("sentiment_compound", 0.0)
    sentiment_risk = ((-sentiment_comp.clip(upper=0)) * 50).clip(0, 100)

    # --- Peer deviation signal ---
    peer_dev = df.get("peer_deviation_score", 0.0)
    peer_risk = (peer_dev * 20).clip(0, 100)

    # --- Multi-signal fusion (Fusing 3-model consensus with peer, rule, and sentiment) ---
    if "if_anomaly_score" in df.columns and "ae_anomaly_score" in df.columns and "stat_anomaly_score" in df.columns:
        df["daily_risk_score"] = (
            0.25 * df["if_anomaly_score"] +       # Model 1: Isolation Forest
            0.20 * df["ae_anomaly_score"] +       # Model 2: Autoencoder
            0.15 * df["stat_anomaly_score"] +     # Model 3: Statistical Z-Score
            0.15 * peer_risk +                    # Peer-cohort deviation
            0.15 * rule_score +                   # Domain rules
            0.10 * sentiment_risk                 # Linguistic/sentiment signal
        ).clip(0, 100)
    else:
        # Fallback for legacy single-model scored.csv
        anom = df.get("anomaly_score", df.get("ae_anomaly_score", 0.0))
        ae_anom = df.get("ae_anomaly_score", anom)
        df["daily_risk_score"] = (
            0.35 * anom +
            0.25 * ae_anom +
            0.20 * peer_risk +
            0.10 * rule_score +
            0.10 * sentiment_risk
        ).clip(0, 100)

    df["daily_risk_score"] = np.round(df["daily_risk_score"], 2)

    # --- Temporal risk-decay (EWMA) — models threat as an evolving process ---
    ALPHA = 0.3  # weight on today's evidence vs accumulated history
    cumulative = []
    for user, group in df.groupby("user_id"):
        group = group.sort_values("date")
        running = 0.0
        vals = []
        for score in group["daily_risk_score"]:
            running = ALPHA * score + (1 - ALPHA) * running
            vals.append(running)
        cumulative.extend(list(zip(group.index, vals)))
    cum_df = pd.DataFrame(cumulative, columns=["idx", "cumulative_risk_score"]).set_index("idx")
    df["cumulative_risk_score"] = np.round(cum_df["cumulative_risk_score"], 2)

    # Dynamic, data-driven thresholds
    p75 = df["cumulative_risk_score"].quantile(0.75)
    p90 = df["cumulative_risk_score"].quantile(0.90)
    p97 = df["cumulative_risk_score"].quantile(0.97)

    def risk_level(score):
        if score >= p97: return "Critical"
        elif score >= p90: return "High"
        elif score >= p75: return "Medium"
        return "Low"

    df["risk_level"] = df["cumulative_risk_score"].apply(risk_level)

    # Escalation flag: sustained elevated risk for 3+ consecutive days
    df["escalating"] = (
        df.groupby("user_id")["cumulative_risk_score"]
        .transform(lambda s: s.rolling(3).apply(lambda x: all(v >= p90 for v in x), raw=True))
        .fillna(0).astype(int)
    )

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Fused, temporally-adaptive risk scoring complete. Output saved to {output_path}")

    return df

if __name__ == "__main__":
    compute_risk()