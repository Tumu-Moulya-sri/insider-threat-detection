import pandas as pd
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
import sys
import argparse

# Add src to python path if not present to allow local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shrinkage_baseline import apply_shrinkage_baseline
from escalation_engine import train_and_escalate

def map_cert_to_cohorts(cert_dir="data/CERT"):
    """
    Reads the CMU CERT dataset files and maps them to the project cohort schema.
    """
    print(f"Mapping CERT dataset from: {cert_dir}")
    
    # Discover required files recursively within the CERT directory
    def find_file(filename):
        matches = glob.glob(os.path.join(cert_dir, "**", filename), recursive=True)
        return matches[0] if matches else None

    def find_ldap_dir():
        # Look for a directory named LDAP at any depth
        matches = glob.glob(os.path.join(cert_dir, "**", "LDAP"), recursive=True)
        return matches[0] if matches else None

    ldap_dir = find_ldap_dir()
    logon_path = find_file("logon.csv")
    file_path = find_file("file.csv")
    http_path = find_file("http.csv")
    device_path = find_file("device.csv")
    email_path = find_file("email.csv")
    psychometric_path = find_file("psychometric.csv")  # not used currently but discovered for completeness

    # Verify required files exist and report any missing ones
    missing = []
    if not logon_path:
        missing.append("logon.csv")
    if not file_path:
        missing.append("file.csv")
    if not http_path:
        missing.append("http.csv")
    if not device_path:
        missing.append("device.csv")
    if not email_path:
        missing.append("email.csv")
    if not ldap_dir:
        missing.append("LDAP directory")

    if missing:
        print("Error: The following required CERT files/directories were not found:")
        for m in missing:
            print(f"  - {m}")
        print("Please ensure the dataset is extracted correctly under data/CERT.")
        return pd.DataFrame()

        
    # 1. Load LDAP to map user roles, departments, and functional units
    print("Loading LDAP metadata...")
    user_map = {}
    company_domains = set()
    if os.path.exists(ldap_dir):
        for path in glob.glob(os.path.join(ldap_dir, "*.csv")):
            try:
                ldap_df = pd.read_csv(path)
                for _, row in ldap_df.iterrows():
                    uid = row.get("user_id")
                    if pd.notna(uid):
                        email = str(row.get("email", ""))
                        if "@" in email:
                            domain = email.split("@")[1]
                            company_domains.add(domain)
                            
                        user_map[uid] = {
                            "role": row.get("role", "Unknown"),
                            "department": row.get("department", "Unknown"),
                            "project_id": row.get("functional_unit", row.get("projects", row.get("department", "Unknown"))),
                        }
            except Exception as e:
                print(f"Error reading LDAP file {path}: {e}")
                
    if not company_domains:
        company_domains.add("draper.com")  # default fallback domain
        
    print(f"Detected company domains for external email logic: {company_domains}")
    
    # 2. Load Ground Truth Answers (XML files or insiders.csv)
    print("Loading Ground Truth Answers...")
    insiders = {}  # user_id -> set of dates
    insider_scenarios = {}  # user_id -> scenario
    answers_dir = os.path.join(cert_dir, "answers")
    if os.path.exists(answers_dir):
        for path in glob.glob(os.path.join(answers_dir, "*.xml")):
            try:
                tree = ET.parse(path)
                root = tree.getroot()
                for insider in root.findall(".//insider"):
                    uid = insider.attrib.get("id")
                    sc_num = insider.attrib.get("scenario", "1")
                    sc_map = {
                        "1": "standard_exfiltration",
                        "2": "contractor_lingering",
                        "3": "slow_subtle_insider",
                        "4": "credential_misuse"
                    }
                    scenario = sc_map.get(str(sc_num), f"scenario_{sc_num}")
                    if uid:
                        insider_scenarios[uid] = scenario
                        if uid not in insiders:
                            insiders[uid] = set()
                        for event in insider.findall(".//event"):
                            date_str = event.attrib.get("date")
                            if date_str:
                                dt = pd.to_datetime(date_str)
                                insiders[uid].add(dt.strftime("%Y-%m-%d"))
            except Exception as e:
                print(f"Error parsing xml {path}: {e}")
                
        # Support fallback insiders.csv
        csv_answers = os.path.join(cert_dir, "insiders.csv")
        if os.path.exists(csv_answers):
            try:
                ans_df = pd.read_csv(csv_answers)
                for _, row in ans_df.iterrows():
                    uid = row.get("user")
                    scenario = row.get("scenario", "standard_exfiltration")
                    start_date = pd.to_datetime(row.get("start")).strftime("%Y-%m-%d")
                    end_date = pd.to_datetime(row.get("end")).strftime("%Y-%m-%d")
                    insider_scenarios[uid] = scenario
                    if uid not in insiders:
                        insiders[uid] = set()
                    date_range = pd.date_range(start_date, end_date)
                    for d in date_range:
                        insiders[uid].add(d.strftime("%Y-%m-%d"))
            except Exception as e:
                print(f"Error reading insiders.csv: {e}")
                
    # 3. Read and Aggregate logon.csv
    print("Processing Logon logs...")
    logon_df = pd.read_csv(logon_path)
    logon_df["datetime"] = pd.to_datetime(logon_df["date"])
    logon_df["date_str"] = logon_df["datetime"].dt.strftime("%Y-%m-%d")
    logon_df["hour"] = logon_df["datetime"].dt.hour
    logon_df["is_weekend"] = logon_df["datetime"].dt.dayofweek >= 5
    
    unique_users = logon_df["user"].unique()
    min_date = logon_df["datetime"].min()
    max_date = logon_df["datetime"].max()
    all_dates = pd.date_range(start=min_date, end=max_date)
    
    daily_stats = {}
    for u in unique_users:
        u_meta = user_map.get(u, {"role": "employee", "department": "Unknown", "project_id": "Unknown"})
        for d in all_dates:
            d_str = d.strftime("%Y-%m-%d")
            is_we = int(d.dayofweek >= 5)
            
            is_insider = 0
            scenario = "none"
            if u in insiders and d_str in insiders[u]:
                is_insider = 1
                scenario = insider_scenarios.get(u, "standard_exfiltration")
                
            daily_stats[(u, d_str)] = {
                "user_id": u,
                "date": d_str,
                "role": u_meta["role"],
                "department": u_meta["department"],
                "project_id": u_meta["project_id"],
                # Clearance-tier placeholder: Explicitly noted placeholder. CERT has no clearance level equivalent,
                # so we assign a constant placeholder level of 1.
                "clearance_level": 1,
                "is_weekend": is_we,
                "off_hours_logins": 0,
                "login_hour": 9.0,
                "logout_hour": 18.0,
                "num_logins": 0,
                "files_accessed": 0,
                "files_downloaded": 0,
                "data_transferred_mb": 0.0,
                "usb_connections": 0,
                "emails_external": 0,
                "cross_project_access_count": 0,
                "max_resource_classification": 1,
                "emails_sent": 0,
                "unique_devices": 1,
                "unique_locations": 1,
                "security_tampering_flag": 0,
                "in_resignation_notice": 0,
                "has_open_disciplinary_case": 0,
                "recent_audit_failure": 0,
                "is_insider_ground_truth": is_insider,
                "insider_scenario": scenario
            }
            
    for _, row in logon_df.iterrows():
        u = row["user"]
        d_str = row["date_str"]
        h = row["hour"]
        act = str(row["activity"]).lower()
        pc = row["pc"]
        is_we = row["is_weekend"]
        
        key = (u, d_str)
        if key in daily_stats:
            stats = daily_stats[key]
            
            if "pc_list" not in stats:
                stats["pc_list"] = set()
            stats["pc_list"].add(pc)
            
            if act == "logon":
                stats["num_logins"] += 1
                if h < 7 or h > 19 or is_we:
                    stats["off_hours_logins"] += 1
                if stats["num_logins"] == 1 or h < stats["login_hour"]:
                    stats["login_hour"] = float(h)
            elif act == "logoff":
                if stats["logout_hour"] == 18.0 or h > stats["logout_hour"]:
                    stats["logout_hour"] = float(h)

    # 4. Process file.csv
    if os.path.exists(file_path):
        print("Processing File logs...")
        file_df = pd.read_csv(file_path)
        file_df["datetime"] = pd.to_datetime(file_df["date"])
        file_df["date_str"] = file_df["datetime"].dt.strftime("%Y-%m-%d")
        for _, row in file_df.iterrows():
            u = row["user"]
            d_str = row["date_str"]
            pc = row["pc"]
            to_usb = row.get("to_removable_media", False)
            
            if isinstance(to_usb, str):
                to_usb = to_usb.lower() in ["true", "1", "y", "yes"]
                
            key = (u, d_str)
            if key in daily_stats:
                stats = daily_stats[key]
                stats["files_accessed"] += 1
                if to_usb:
                    stats["files_downloaded"] += 1
                    stats["usb_connections"] += 1  # Fallback USB connect indicator
                    
                if "pc_list" not in stats:
                    stats["pc_list"] = set()
                stats["pc_list"].add(pc)

    # 5. Process device.csv (Direct Connect/Disconnect)
    if os.path.exists(device_path):
        print("Processing Device/USB logs...")
        device_df = pd.read_csv(device_path)
        device_df["datetime"] = pd.to_datetime(device_df["date"])
        device_df["date_str"] = device_df["datetime"].dt.strftime("%Y-%m-%d")
        for _, row in device_df.iterrows():
            u = row["user"]
            d_str = row["date_str"]
            act = str(row["activity"]).lower()
            pc = row["pc"]
            
            key = (u, d_str)
            if key in daily_stats:
                stats = daily_stats[key]
                if act == "connect":
                    stats["usb_connections"] += 1
                if "pc_list" not in stats:
                    stats["pc_list"] = set()
                stats["pc_list"].add(pc)

    # 6. Process http.csv
    http_count = {}
    if os.path.exists(http_path):
        print("Processing HTTP logs (chunked to optimize memory)...")
        for chunk in pd.read_csv(http_path, chunksize=100000):
            chunk["datetime"] = pd.to_datetime(chunk["date"])
            chunk["date_str"] = chunk["datetime"].dt.strftime("%Y-%m-%d")
            for _, row in chunk.iterrows():
                u = row["user"]
                d_str = row["date_str"]
                pc = row["pc"]
                
                key = (u, d_str)
                if key in daily_stats:
                    stats = daily_stats[key]
                    http_count[key] = http_count.get(key, 0) + 1
                    if "pc_list" not in stats:
                        stats["pc_list"] = set()
                    stats["pc_list"].add(pc)

    # 7. Process email.csv
    if os.path.exists(email_path):
        print("Processing Email logs...")
        email_df = pd.read_csv(email_path)
        email_df["datetime"] = pd.to_datetime(email_df["date"])
        email_df["date_str"] = email_df["datetime"].dt.strftime("%Y-%m-%d")
        for _, row in email_df.iterrows():
            u = row["user"]
            d_str = row["date_str"]
            to_field = str(row.get("to", ""))
            
            is_external = 0
            recipients = [r.strip() for r in to_field.split(";") if r.strip()]
            for r in recipients:
                if "@" in r:
                    r_domain = r.split("@")[1]
                    if r_domain not in company_domains:
                        is_external = 1
                        break
                        
            key = (u, d_str)
            if key in daily_stats:
                stats = daily_stats[key]
                stats["emails_sent"] += 1
                if is_external:
                    stats["emails_external"] += 1

    # Compile mapped data
    print("Compiling final mapped CERT DataFrame...")
    rows = []
    min_date_val = min_date
    
    for (u, d_str), stats in daily_stats.items():
        pc_set = stats.pop("pc_list", set())
        if pc_set:
            stats["unique_devices"] = len(pc_set)
            stats["unique_locations"] = len(pc_set)
            
        cur_date = pd.to_datetime(d_str)
        stats["day_index"] = (cur_date - min_date_val).days
        
        hc = http_count.get((u, d_str), 0)
        stats["data_transferred_mb"] = round(
            stats["files_downloaded"] * 10.0 + stats["files_accessed"] * 0.1 + hc * 0.05, 2
        )
        rows.append(stats)
        
    mapped_df = pd.DataFrame(rows)
    mapped_df = mapped_df.sort_values(["user_id", "date"])
    print(f"Successfully mapped {len(mapped_df)} daily records for {len(unique_users)} users.")
    return mapped_df

def report_side_by_side(synth_path="data/escalated.csv", cert_path="data/cert_escalated.csv"):
    def get_stats(df_path):
        if not os.path.exists(df_path):
            return None
        df = pd.read_csv(df_path)
        active = df[df["is_insider_ground_truth"] == 1]
        benign = df[df["is_insider_ground_truth"] == 0]
        
        recall = (active["escalation_decision"] == "Alert").mean() if len(active) > 0 else 0.0
        fpr = (benign["escalation_decision"] == "Alert").mean() if len(benign) > 0 else 0.0
        
        latencies = []
        for user, group in active.groupby("user_id"):
            first_active_day = group["day_index"].min()
            alerts = group[group["escalation_decision"] == "Alert"]
            if len(alerts) > 0:
                first_alert_day = alerts["day_index"].min()
                latencies.append(first_alert_day - first_active_day)
        avg_lat = np.mean(latencies) if latencies else 0.0
        return len(df), len(active), recall, fpr, avg_lat
        
    synth_stats = get_stats(synth_path)
    cert_stats = get_stats(cert_path)
    
    print("\n==================================================================")
    print("SIDE-BY-SIDE PIPELINE EVALUATION")
    print("==================================================================")
    
    headers = ["Metric", "Synthetic Dataset", "CERT Dataset"]
    rows = []
    
    metrics_list = [
        ("Total Log Days", lambda s, c: (f"{s[0]}" if s else "N/A", f"{c[0]}" if c else "N/A")),
        ("Active Insider Days", lambda s, c: (f"{s[1]}" if s else "N/A", f"{c[1]}" if c else "N/A")),
        ("Detection Recall", lambda s, c: (f"{s[2]:.3%}" if s else "N/A", f"{c[2]:.3%}" if c else "N/A")),
        ("False Positive Rate", lambda s, c: (f"{s[3]:.3%}" if s else "N/A", f"{c[3]:.3%}" if c else "N/A")),
        ("Detection Latency", lambda s, c: (f"{s[4]:.2f} days" if s else "N/A", f"{c[4]:.2f} days" if c else "N/A")),
    ]
    
    for name, extractor in metrics_list:
        s_val, c_val = extractor(synth_stats, cert_stats)
        rows.append([name, s_val, c_val])
        
    res_df = pd.DataFrame(rows, columns=headers)
    print(res_df.to_string(index=False))
    print("==================================================================")

def run_cert_evaluation(cert_dir="data/CERT"):
    mapped_df = map_cert_to_cohorts(cert_dir)
    
    if len(mapped_df) > 0:
        os.makedirs("data", exist_ok=True)
        mapped_df.to_csv("data/cert_mapped.csv", index=False)
        print("CERT data mapped successfully and saved to data/cert_mapped.csv")
        
        print("\n--- Running Shrinkage Baseline on CERT ---")
        features_df = apply_shrinkage_baseline(input_path="data/cert_mapped.csv", output_path="data/cert_features.csv")
        
        print("\n--- Running Escalation Engine on CERT ---")
        # Ensure pseudo-labels exist for training
        from escalation_engine import define_pseudo_labels
        features_df["stage_label"] = features_df.apply(define_pseudo_labels, axis=1)
        features_df.to_csv("data/cert_features.csv", index=False)
        
        train_and_escalate(input_path="data/cert_features.csv", output_path="data/cert_escalated.csv")
        print("\nCERT evaluation complete. Saved results to data/cert_escalated.csv")
        
        report_side_by_side(synth_path="data/escalated.csv", cert_path="data/cert_escalated.csv")
    else:
        print("Could not evaluate CERT: dataset mapping returned empty data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cert_dir", type=str, default="data/CERT", help="Path to extract of CERT logs directory")
    args = parser.parse_args()
    
    run_cert_evaluation(args.cert_dir)
