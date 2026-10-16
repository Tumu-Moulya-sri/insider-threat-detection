import pandas as pd
import numpy as np
import copy
import joblib

# Default candidate metric names if explicitly queried
DEFAULT_METRICS = [
    "off_hours_logins", "files_accessed", "files_downloaded",
    "data_transferred_mb", "usb_connections", "emails_external",
    "cross_project_access_count", "max_resource_classification",
    "emails_sent", "unique_devices", "unique_locations"
]

PROJECT_LINKED_METRICS = [
    "files_accessed", "files_downloaded", "data_transferred_mb", 
    "cross_project_access_count", "max_resource_classification"
]

def get_dynamic_metrics(df):
    exclude_cols = {
        "user_id", "date", "department", "role", "project_id", "clearance_level",
        "day_index", "is_weekend", "login_hour", "logout_hour", "num_logins",
        "security_tampering_flag", "in_resignation_notice", "has_open_disciplinary_case",
        "recent_audit_failure", "is_insider_ground_truth", "insider_scenario",
        "message_text", "comm_text", "stage_label"
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    metrics = [c for c in numeric_cols if c not in exclude_cols]
    return metrics if len(metrics) > 0 else [m for m in DEFAULT_METRICS if m in df.columns]

def save_baseline_state(user_history, priors_lookup, metrics, output_path="models/training_baseline_state.pkl"):
    """Persist the training-phase baseline state so that test inference can warm-start.
    
    Saves:
      - user_history: per-user per-metric EWMA mean/var, observation count, shrinkage weight,
                      current project, department
      - priors_lookup: cohort prior means and variances keyed by (role, project_id, clearance_level)
      - metrics: ordered list of metric column names used during training
    """
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    state = {
        "user_history": copy.deepcopy(user_history),
        "priors_lookup": copy.deepcopy(priors_lookup),
        "metrics": list(metrics),
    }
    joblib.dump(state, output_path)
    print(f"Baseline state saved to {output_path} ({len(user_history)} users, {len(metrics)} metrics)")


def apply_shrinkage_baseline(input_path="data/synthetic_logs.csv", output_path="data/features_shrinkage.csv", disable_novelty_1=False, train_end_date=None, baseline_state_path=None, save_state_path=None):
    import os
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    # Dynamically derive day_index if missing
    if "day_index" not in df.columns:
        df["day_index"] = (df["date"] - df["date"].min()).dt.days

    # Dynamically detect present metric columns
    metrics = get_dynamic_metrics(df)
    project_linked = [m for m in PROJECT_LINKED_METRICS if m in metrics]

    # Compute cohort priors from training period only if provided
    if train_end_date is not None:
        try:
            train_cutoff = pd.to_datetime(train_end_date)
        except Exception:
            train_cutoff = pd.to_datetime(train_end_date, errors='coerce')
        prior_df = df[df["date"] <= train_cutoff]
    else:
        prior_df = df[df["day_index"] < 30]
    cohort_priors = prior_df.groupby(["role", "project_id", "clearance_level"])[metrics].agg(["mean", "var"]).reset_index()
    cohort_priors.columns = ["_".join(col).strip('_') for col in cohort_priors.columns.values]
    
    for m in metrics:
        if f"{m}_var" in cohort_priors.columns:
            cohort_priors[f"{m}_var"] = cohort_priors[f"{m}_var"].fillna(0.1).replace(0, 0.1)

    priors_lookup = {}
    for _, cp_row in cohort_priors.iterrows():
        key = (cp_row["role"], cp_row["project_id"], cp_row["clearance_level"])
        priors_lookup[key] = cp_row.to_dict()

    # ── Warm-start: load persisted training state if provided ──
    if baseline_state_path is not None and os.path.exists(baseline_state_path):
        saved = joblib.load(baseline_state_path)
        user_history = saved["user_history"]
        priors_lookup = saved["priors_lookup"]
        metrics = saved["metrics"]
        project_linked = [m for m in PROJECT_LINKED_METRICS if m in metrics]
        print(f"Warm-started baseline from {baseline_state_path} ({len(user_history)} users)")
        # Count how many test users are new (cold-start via cohort priors)
        test_users = set(df["user_id"].unique())
        known_users = set(user_history.keys())
        new_users = test_users - known_users
        if new_users:
            print(f"  {len(new_users)} test user(s) have no training history -> cohort-prior cold start")
        else:
            print(f"  All {len(test_users)} test users have training history")
    else:
        user_history = {} 
    results = []
    pending_updates = []
    
    # Load analyst feedback once at the beginning to avoid major I/O bottlenecks in the loops
    approved_set = set()
    feedback_path = "data/analyst_feedback.csv"
    if os.path.exists(feedback_path):
        try:
            feedback_df = pd.read_csv(feedback_path)
            if "user_id" in feedback_df.columns and "metric" in feedback_df.columns and "approved" in feedback_df.columns:
                approved_rows = feedback_df[feedback_df["approved"] == 1]
                for _, row_fb in approved_rows.iterrows():
                    approved_set.add((row_fb["user_id"], row_fb["metric"]))
        except Exception as e:
            print(f"Error reading analyst feedback: {e}")

    # EWMA parameters
    ALPHA = 0.1 # Half-life of roughly ~7 days for adapting to new baselines
    
    df_records = df.to_dict('records')
    for row in df_records:
        user = row["user_id"]
        role = row["role"]
        project = row["project_id"]
        clearance = row["clearance_level"]
        department = row["department"]
        
        if user not in user_history:
            user_history[user] = {m: {"n": 0, "ewma_mean": None, "ewma_var": 0.0, "prev_W": 0.0} for m in metrics}
            user_history[user]["current_project"] = project
            user_history[user]["department"] = department
            
        if user_history[user]["current_project"] != project:
            # Reset project-linked metrics on project switch
            for m in project_linked:
                user_history[user][m] = {"n": 0, "ewma_mean": None, "ewma_var": 0.0, "prev_W": 0.0}
            user_history[user]["current_project"] = project
            
        cohort_info = priors_lookup.get((role, project, clearance), None)
        
        out_row = dict(row)
        
        for m in metrics:
            val = row[m]
            hist = user_history[user][m]
            n = hist["n"]
            
            if cohort_info and f"{m}_mean" in cohort_info and pd.notna(cohort_info[f"{m}_mean"]):
                mu_prior = cohort_info[f"{m}_mean"]
            else:
                mu_prior = 0.0

            if cohort_info and f"{m}_var" in cohort_info and pd.notna(cohort_info[f"{m}_var"]):
                var_prior = cohort_info[f"{m}_var"]
            else:
                var_prior = 1.0

            
            static_mean = mu_prior
            
            # Naive Baseline uses EWMA but starts from 0
            if n == 0:
                naive_mean = 0.0
                hist["ewma_mean"] = val # Initialize EWMA
                hist["ewma_var"] = 0.0
            else:
                naive_mean = hist["ewma_mean"]
                # Update EWMA
                diff = val - hist["ewma_mean"]
                
                # --- Hardening: Defense against gradual poisoning (the "boiling frog" attack class) ---
                # Cap how much any single day's evidence can shift a user's baseline
                raw_shift = ALPHA * diff
                max_daily_shift = 0.5  # configurable daily shift limit
                capped_shift = np.clip(raw_shift, -max_daily_shift, max_daily_shift)
                
                hist["ewma_mean"] += capped_shift
                hist["ewma_var"] = (1 - ALPHA) * (hist["ewma_var"] + ALPHA * diff**2)
                
            x_bar = hist["ewma_mean"]
            s2 = hist["ewma_var"]
            
            # Shrinkage Baseline (combining prior with EWMA user history)
            if n < 3: 
                shrunk_mean = mu_prior
                shrunk_var = var_prior
                W = 0.0
            else:
                epsilon = 0.01
                prior_precision = 1.0 / max(var_prior, epsilon)
                user_precision = n / max(s2, epsilon)
                
                W_raw = user_precision / (prior_precision + user_precision)
                
                # --- Hardening: Defense against gradual poisoning (the "boiling frog" attack class) ---
                # Cap how much any single day's evidence can shift a user's shrinkage weight W
                prev_W = hist.get("prev_W", 0.0)
                max_w_change = 0.05  # configurable weight shift limit
                W = np.clip(W_raw, prev_W - max_w_change, prev_W + max_w_change)
                hist["prev_W"] = W
                
                if disable_novelty_1:
                    shrunk_mean = mu_prior
                    shrunk_var = var_prior
                else:
                    shrunk_mean = W * x_bar + (1 - W) * mu_prior
                    shrunk_var = W * max(s2, epsilon) + (1 - W) * var_prior
            
            # --- Hardening: Defense against gradual poisoning (the "boiling frog" attack class) ---
            # Require analyst confirmation before a cumulative baseline shift beyond a threshold is fully trusted.
            # Simulated via Delta Buffer / analyst feedback mechanism.
            if not disable_novelty_1:
                cumulative_shift = shrunk_mean - mu_prior
                cumulative_shift_threshold = 2.0  # configurable threshold
                
                if abs(cumulative_shift) > cumulative_shift_threshold:
                    approved = (user, m) in approved_set
                    
                    if not approved:
                        # Cap baseline shift (untrusted cumulative baseline shift is capped/frozen)
                        shrunk_mean = mu_prior + np.sign(cumulative_shift) * cumulative_shift_threshold
                        
                        # Store in Delta Buffer for analyst confirmation
                        d_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10]
                        pending_updates.append({
                            "date": d_str,
                            "user_id": user,
                            "metric": m,
                            "prior_mean": mu_prior,
                            "attempted_mean": mu_prior + cumulative_shift,
                            "cumulative_shift": cumulative_shift,
                            "status": "Pending Confirmation"
                        })
            
            user_zscore = (val - shrunk_mean) / np.sqrt(max(shrunk_var, 0.001))
            
            out_row[f"{m}_static_mean"] = static_mean
            out_row[f"{m}_naive_mean"] = naive_mean
            out_row[f"{m}_shrunk_mean"] = shrunk_mean
            out_row[f"{m}_zscore"] = user_zscore
            
            hist["n"] += 1
            
        results.append(out_row)
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_path, index=False)
    print(f"Adaptive Baselines (EWMA) calculated across {len(metrics)} metrics. Saved to {output_path}")
    
    # Export Delta Buffer pending updates if any
    if pending_updates:
        pending_df = pd.DataFrame(pending_updates)
        pending_path = "data/pending_baseline_updates.csv"
        pending_df.to_csv(pending_path, index=False)
        print(f"Boiling frog defense: {len(pending_df)} updates queued in Delta Buffer ({pending_path})")
    
    # Export Behavioural Fingerprints
    fingerprint_rows = []
    for user, h in user_history.items():
        row_dict = {"user_id": user, "department": h["department"]}
        for m in metrics:
            row_dict[f"{m}_baseline"] = h[m]["ewma_mean"] if h[m]["ewma_mean"] is not None else 0.0
        fingerprint_rows.append(row_dict)
        
    fp_df = pd.DataFrame(fingerprint_rows)
    fp_df.to_csv("data/behavioural_fingerprints.csv", index=False)
    print("Behavioural Fingerprints exported to data/behavioural_fingerprints.csv")
    
    # ── Persist state for downstream warm-start if requested ──
    if save_state_path is not None:
        save_baseline_state(user_history, priors_lookup, metrics, output_path=save_state_path)
    
    return res_df

if __name__ == "__main__":
    apply_shrinkage_baseline()

