import os
import pandas as pd
import numpy as np

def extract_spedia_features(
    input_path=None,
    output_path="data/external/spedia/spedia_features.csv"
):
    """
    SPEDIA Behavioral Feature Adapter.
    Extracts daily user-level behavioral aggregates genuinely present in the SPEDIA dataset.
    
    Raw Log Source: logs_SPEDIA_annotated_en.csv (72,250 events across 17 users/actors)
    Target Granularity: (User, Date) daily behavioral vectors.
    """
    candidate_paths = [
        input_path,
        "data/spedia dataset/logs_SPEDIA_annotated_en.csv",
        "data/external/spedia/logs_SPEDIA_annotated_en.csv"
    ]
    
    resolved_path = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            resolved_path = p
            break
            
    if not resolved_path:
        print(f"[ERROR] SPEDIA source file not found in candidates: {candidate_paths}")
        return None

    print(f"Reading raw SPEDIA dataset from {resolved_path}...")
    df = pd.read_csv(resolved_path, low_memory=False)
    
    # Clean timestamps and users
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "User"])
    
    df["date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["Timestamp"].dt.hour
    df["is_weekend"] = df["Timestamp"].dt.weekday.isin([5, 6]).astype(int)
    df["is_off_hours"] = ((df["hour"] < 8) | (df["hour"] >= 18) | (df["is_weekend"] == 1)).astype(int)

    total_valid_events = len(df)
    unique_actors = df["User"].nunique()
    print(f"Loaded {total_valid_events:,} valid events for {unique_actors} actors.")

    # -------------------------------------------------------------------------
    # DAILY USER AGGREGATIONS (Genuinely Supported Fields Only)
    # -------------------------------------------------------------------------
    daily_records = []

    for (user, date), group in df.groupby(["User", "date"]):
        total_events = len(group)
        off_hours_events = int(group["is_off_hours"].sum())
        
        # Primary Agent Host Category
        agent_names = group["Agent_name"].dropna().unique()
        primary_agent = str(agent_names[0]) if len(agent_names) > 0 else "unknown"
        
        if "public-host" in primary_agent.lower():
            department = "Public Host / DMZ"
            role = "Public Server Operator"
        elif "internal-server" in primary_agent.lower():
            department = "Internal Enterprise Server"
            role = "Internal Systems Operator"
        elif "attacker" in primary_agent.lower():
            department = "Red Team / Target Host"
            role = "Endpoint Host Account"
        else:
            department = "General Infrastructure"
            role = "System Operator"

        # 1. Authentication / Session Activity
        num_logins = int((group["Action"] == "Login").sum())
        login_failed = int((group["Action"].isin(["Login Failed", "Multiple failed logins"])).sum())
        num_logoffs = int((group["Action"] == "Logoff").sum())
        session_events = int((group["Activity"] == "session").sum())

        # 2. Command Activity (Auditd / System Execution)
        commands_executed = int((group["Action"] == "Command executed").sum())
        unique_commands = int(group["Command"].dropna().nunique())
        high_level_commands = int((group["Level"].fillna(0) >= 8.0).sum())

        # 3. File Activity (Syscheck / Integrity)
        files_accessed = int((group["Action"] == "File accessed").sum())
        files_modified = int((group["Action"] == "File modified").sum())
        files_added = int((group["Action"] == "File added").sum())
        files_deleted = int((group["Action"] == "File deleted").sum())
        total_file_events = int((group["Activity"] == "file").sum())
        
        # 4. Email Activity (CERT / Communications)
        emails_sent = int((group["Action"] == "Email sent").sum())
        total_attachments = int(group["Attachments"].fillna(0).sum())
        total_email_size_kb = float(group["Size"].fillna(0).sum() / 1024.0)

        # External email recipient check (if domain != dtaa.com)
        external_emails = 0
        for to_addr in group["To"].dropna():
            if any(domain not in str(to_addr).lower() for domain in ["@dtaa.com", "dtaa"]):
                external_emails += 1

        # 5. Network / HTTP Activity
        http_requests = int((group["Activity"] == "http").sum())
        unique_urls = int(group["Url"].dropna().nunique())

        # 6. Device / Hardware Connections
        usb_connects = int((group["Action"] == "Connect").sum())
        usb_disconnects = int((group["Action"] == "Disconnect").sum())
        device_events = int((group["Activity"] == "device").sum())

        # Security Tampering Indicators
        # In SPEDIA: privilege manipulation or high-volume failed auth / systemctl kills
        security_tampering_flag = int((high_level_commands >= 15) or (login_failed >= 5))

        # 7. Ground Truth Malicious Event Proportion (for offline research evaluation only)
        anomaly_events = int((group["Anomaly"] == 1).sum()) if "Anomaly" in group.columns else 0
        is_malicious_day = int(anomaly_events > 0)

        daily_records.append({
            "user_id": user,
            "date": date,
            "department": department,
            "role": role,
            "total_events": total_events,
            "off_hours_events": off_hours_events,
            "num_logins": num_logins,
            "login_failed": login_failed,
            "num_logoffs": num_logoffs,
            "session_events": session_events,
            "commands_executed": commands_executed,
            "unique_commands": unique_commands,
            "high_level_commands": high_level_commands,
            "files_accessed": files_accessed,
            "files_modified": files_modified,
            "files_added": files_added,
            "files_deleted": files_deleted,
            "total_file_events": total_file_events,
            "emails_sent": emails_sent,
            "external_emails": external_emails,
            "total_attachments": total_attachments,
            "total_email_size_kb": round(total_email_size_kb, 2),
            "http_requests": http_requests,
            "unique_urls": unique_urls,
            "usb_connects": usb_connects,
            "usb_disconnects": usb_disconnects,
            "device_events": device_events,
            "security_tampering_flag": security_tampering_flag,
            "spedia_anomaly_event_count": anomaly_events,
            "spedia_is_malicious_day": is_malicious_day
        })

    feat_df = pd.DataFrame(daily_records)
    feat_df = feat_df.sort_values(["date", "user_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feat_df.to_csv(output_path, index=False)
    print(f"Extracted {len(feat_df)} daily user behavioral records across {feat_df['user_id'].nunique()} actors.")
    print(f"Saved daily features to {output_path}")

    return feat_df

if __name__ == "__main__":
    extract_spedia_features()
