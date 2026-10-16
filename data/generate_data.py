import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import argparse

NEUTRAL_PHRASES = [
    "Please review the attached report by EOD.",
    "Meeting rescheduled to 3 PM tomorrow.",
    "Thanks for the update on the project status.",
    "Can you share the latest budget figures?",
    "Great work on the deployment this week."
]
NEGATIVE_PHRASES = [
    "I'm really frustrated with how this team is being treated.",
    "This workload is completely unmanageable and unfair.",
    "Nobody listens to my concerns around here.",
    "I don't think my contributions are valued at all.",
    "This place makes me want to quit every single day."
]
RESIGNATION_PHRASES = [
    "I'm seriously considering leaving the organization soon.",
    "Looking at other job offers outside the company right now.",
    "This will probably be my last project here.",
    "Talking to a recruiter about opportunities elsewhere."
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data generation")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    random.seed(args.seed)

    NUM_USERS = 60
    NUM_DAYS = 90
    START_DATE = datetime(2026, 4, 1)

    DEPARTMENTS = ["IT", "HR", "Finance", "Defense-Ops", "R&D", "Logistics"]
    ROLES = ["researcher", "senior_researcher", "administrator", "contractor", "project_manager"]
    PROJECTS = ["P-Alpha", "P-Beta", "P-Gamma", "P-Delta", "Shared-Ops"]
    CLEARANCE_LEVELS = [1, 2, 3, 4]

    ALL_USERS = [f"U{1000+i}" for i in range(NUM_USERS)]

    # Define insider users with start day and daily activation probability for gradual appearance
    INSIDER_USERS = random.sample(ALL_USERS, 6)
    # Each insider starts potentially malicious activity on a random start day (0-89)
    INSIDER_START_DAY = {u: random.randint(0, NUM_DAYS - 1) for u in INSIDER_USERS}
    # Probability of being active on any given day after start day (gradual, not abrupt)
    INSIDER_DAILY_PROB = 0.15
    INSIDER_SCENARIOS = {
        INSIDER_USERS[0]: "standard_exfiltration",
        INSIDER_USERS[1]: "standard_exfiltration",
        INSIDER_USERS[2]: "contractor_lingering",
        INSIDER_USERS[3]: "credential_misuse",
        INSIDER_USERS[4]: "slow_subtle_insider",
        INSIDER_USERS[5]: "cross_project_exploration"
    }

    user_dept = {}
    user_role = {}
    user_project = {}
    user_clearance = {}
    user_context = {}

    for u in ALL_USERS:
        user_dept[u] = random.choice(DEPARTMENTS)
        
        if u == INSIDER_USERS[2]:
            user_role[u] = "contractor"
        else:
            user_role[u] = random.choice(ROLES)
            
        user_project[u] = random.choice(PROJECTS)
        user_clearance[u] = random.choice(CLEARANCE_LEVELS)
        
        user_context[u] = {
            "in_resignation_notice": int(random.random() < 0.05),
            "has_open_disciplinary_case": int(random.random() < 0.03),
            "recent_audit_failure": int(random.random() < 0.02)
        }

    BENIGN_ANOMALY_USERS = random.sample([u for u in ALL_USERS if u not in INSIDER_USERS], 4)
    BENIGN_SCENARIOS = {
        BENIGN_ANOMALY_USERS[0]: ("new_onboarding", random.randint(10, 80)),
        BENIGN_ANOMALY_USERS[1]: ("cross_project_collab", random.randint(20, 70)),
        BENIGN_ANOMALY_USERS[2]: ("emergency_access", random.randint(30, 60)),
        BENIGN_ANOMALY_USERS[3]: ("deadline_crunch", random.randint(40, 70))
    }

    rows = []

    for day_idx in range(NUM_DAYS):
        date = START_DATE + timedelta(days=day_idx)
        is_weekend = date.weekday() >= 5
        
        for user in ALL_USERS:
            dept = user_dept[user]
            role = user_role[user]
            project = user_project[user]
            clearance = user_clearance[user]
            context = user_context[user]
            
            # Determine if insider is active on this day based on start day and probability
            if user in INSIDER_USERS:
                start_day = INSIDER_START_DAY[user]
                is_insider_active = day_idx >= start_day and random.random() < INSIDER_DAILY_PROB
                scenario = INSIDER_SCENARIOS.get(user, "none")
            else:
                is_insider_active = False
                scenario = "none"
            
            base_login_hour = np.random.normal(9, 0.7)
            base_logout_hour = np.random.normal(18, 0.8)
            num_logins = max(1, int(np.random.normal(3, 1)))
            off_hours_logins = np.random.poisson(0.1)
            files_accessed = max(0, int(np.random.normal(20, 6)))
            files_downloaded = max(0, int(np.random.normal(5, 2)))
            data_transferred_mb = max(0, np.random.normal(50, 20))
            usb_connections = np.random.poisson(0.2)
            emails_sent = max(0, int(np.random.normal(10, 3)))
            emails_external = max(0, int(np.random.normal(1, 0.5)))
            unique_devices = 1
            unique_locations = 1
            
            max_resource_classification = min(clearance, max(1, int(np.random.normal(clearance - 1, 0.5))))
            cross_project_access_count = np.random.poisson(0.05)
            security_tampering_flag = 0

            if is_weekend:
                num_logins = np.random.poisson(0.3)
                files_accessed = int(files_accessed * 0.2)
                files_downloaded = int(files_downloaded * 0.2)
                data_transferred_mb *= 0.2

            if user in BENIGN_SCENARIOS:
                b_scenario, b_day = BENIGN_SCENARIOS[user]
                if b_scenario == "new_onboarding" and day_idx < b_day:
                    files_accessed = 0; files_downloaded = 0; data_transferred_mb = 0; emails_sent = 0
                elif b_scenario == "new_onboarding" and day_idx == b_day:
                    files_accessed += 50
                    files_downloaded += 20
                elif b_scenario == "cross_project_collab" and b_day <= day_idx <= b_day + 5:
                    cross_project_access_count += np.random.randint(5, 15)
                    files_accessed += 20
                elif b_scenario == "emergency_access" and day_idx == b_day:
                    off_hours_logins += 1
                    base_login_hour = 3.0
                    max_resource_classification = 4
                    cross_project_access_count += 5
                elif b_scenario == "deadline_crunch" and b_day <= day_idx <= b_day + 7:
                    off_hours_logins += 1
                    base_logout_hour = 22.0
                    files_accessed += 30

            if is_insider_active:
                if scenario == "standard_exfiltration":
                    off_hours_logins += np.random.poisson(2)
                    files_downloaded += np.random.randint(15, 40)
                    data_transferred_mb += np.random.uniform(300, 900)
                    usb_connections += np.random.poisson(2)
                    emails_external += np.random.randint(2, 6)
                    unique_devices += np.random.randint(0, 2)
                    unique_locations += np.random.randint(0, 2)
                    base_login_hour = np.random.normal(2, 1.5) % 24
                    
                elif scenario == "contractor_lingering":
                    off_hours_logins += np.random.poisson(1)
                    data_transferred_mb += np.random.uniform(100, 300)
                    
                elif scenario == "credential_misuse":
                    unique_devices += 2
                    unique_locations += 2
                    max_resource_classification = 4
                    cross_project_access_count += 10
                    
                elif scenario == "slow_subtle_insider":
                    files_downloaded += np.random.randint(2, 6)
                    data_transferred_mb += np.random.uniform(20, 50)
                    emails_external += np.random.randint(0, 2)
                    
                elif scenario == "cross_project_exploration":
                    cross_project_access_count += np.random.randint(10, 30)
                    files_accessed += np.random.randint(30, 80)
                
                if random.random() < 0.10:
                    security_tampering_flag = 1
                    
            # FIX: Decouple sentiment from insider truth.
            # Sentiment is purely driven by organizational context now, plus a tiny baseline.
            prob_neg = 0.05
            prob_res = 0.01
            
            if context["in_resignation_notice"]:
                prob_neg = 0.60
                prob_res = 0.40
            elif context["has_open_disciplinary_case"]:
                prob_neg = 0.50
                prob_res = 0.05
            
            # Tiny (+2%) noise for active insiders just so it's not strictly 0 correlation,
            # but small enough it won't dominate the model.
            if is_insider_active:
                prob_neg += 0.02
                prob_res += 0.01

            rand_val = random.random()
            if rand_val < prob_res:
                comm_text = random.choice(RESIGNATION_PHRASES)
            elif rand_val < prob_res + prob_neg:
                comm_text = random.choice(NEGATIVE_PHRASES)
            else:
                comm_text = random.choice(NEUTRAL_PHRASES)

            rows.append({
                "user_id": user,
                "department": dept,
                "role": role,
                "project_id": project,
                "clearance_level": clearance,
                "date": date.strftime("%Y-%m-%d"),
                "day_index": day_idx,
                "is_weekend": int(is_weekend),
                "login_hour": round(base_login_hour % 24, 2),
                "logout_hour": round(base_logout_hour % 24, 2),
                "num_logins": num_logins,
                "off_hours_logins": off_hours_logins,
                "files_accessed": files_accessed,
                "files_downloaded": files_downloaded,
                "data_transferred_mb": round(data_transferred_mb, 2),
                "usb_connections": usb_connections,
                "emails_sent": emails_sent,
                "comm_text": comm_text,
                "emails_external": emails_external,
                "unique_devices": unique_devices,
                "unique_locations": unique_locations,
                "max_resource_classification": max_resource_classification,
                "cross_project_access_count": cross_project_access_count,
                "security_tampering_flag": security_tampering_flag,
                "in_resignation_notice": context["in_resignation_notice"],
                "has_open_disciplinary_case": context["has_open_disciplinary_case"],
                "recent_audit_failure": context["recent_audit_failure"],
                "is_insider_ground_truth": int(is_insider_active),
                "insider_scenario": scenario if is_insider_active else "none"
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/synthetic_logs.csv", index=False)
    print(f"Generated {len(df)} rows for {NUM_USERS} users over {NUM_DAYS} days with seed {args.seed}.")

if __name__ == "__main__":
    main()