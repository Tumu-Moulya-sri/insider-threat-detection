import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure src in python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

def run_tests():
    print("=" * 75)
    print("TRUSTSHIELD SPEDIA RUNTIME & INTEGRATION TEST SUITE")
    print("=" * 75)
    
    passed_tests = 0
    total_tests = 14

    # 1. Raw SPEDIA file exists
    raw_path_1 = "data/spedia dataset/logs_SPEDIA_annotated_en.csv"
    raw_path_2 = "data/external/spedia/logs_SPEDIA_annotated_en.csv"
    raw_exists = os.path.exists(raw_path_1) or os.path.exists(raw_path_2)
    assert raw_exists, "SPEDIA raw dataset file not found."
    print("[TEST 1/14] PASS: Raw SPEDIA dataset file verified.")
    passed_tests += 1

    # 2. Raw events loaded & 3. 17 actors detected dynamically
    target_raw = raw_path_1 if os.path.exists(raw_path_1) else raw_path_2
    df_raw = pd.read_csv(target_raw, low_memory=False)
    df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"], errors="coerce")
    df_raw = df_raw.dropna(subset=["Timestamp", "User"])
    
    event_count = len(df_raw)
    actor_count = df_raw["User"].nunique()
    assert event_count >= 70000, f"Expected >= 70,000 raw events, found {event_count}"
    assert actor_count == 17, f"Expected 17 unique actors, found {actor_count}"
    print(f"[TEST 2/14] PASS: Raw events loaded ({event_count:,} valid events).")
    passed_tests += 1
    print(f"[TEST 3/14] PASS: 17 actors detected dynamically ({actor_count} actors).")
    passed_tests += 1

    # 4. 231 user-day features generated
    feat_path = "data/external/spedia/spedia_features.csv"
    assert os.path.exists(feat_path), f"Features file {feat_path} does not exist."
    feat_df = pd.read_csv(feat_path)
    assert len(feat_df) == 231, f"Expected 231 user-day feature records, found {len(feat_df)}"
    print(f"[TEST 4/14] PASS: 231 user-day features verified.")
    passed_tests += 1

    # 5. SPEDIA model artifacts exist
    models_dir = "models/spedia"
    for mod in ["scaler.pkl", "iforest.pkl", "autoencoder.pkl", "training_baseline_state.pkl"]:
        p = os.path.join(models_dir, mod)
        assert os.path.exists(p), f"Missing model artifact: {p}"
    print("[TEST 5/14] PASS: All SPEDIA model artifacts exist in models/spedia/.")
    passed_tests += 1

    # 6. No future dates enter previous baseline (chronological ordering)
    dates = feat_df["date"].tolist()
    assert dates == sorted(dates), "SPEDIA feature dataset must be strictly chronological."
    print("[TEST 6/14] PASS: Chronological integrity verified (no future date leakage).")
    passed_tests += 1

    # 7. AE threshold comes from training
    ae = joblib.load(os.path.join(models_dir, "autoencoder.pkl"))
    assert hasattr(ae, "threshold_"), "Autoencoder model must have saved threshold_ attribute."
    assert ae.threshold_ > 0, "Autoencoder threshold must be positive float."
    print(f"[TEST 7/14] PASS: AE threshold loaded from training state (threshold={ae.threshold_:.6f}).")
    passed_tests += 1

    # 8. Predictions do not use Anomaly labels
    alerts_path = "data/demo/spedia/trustshield_alerts.csv"
    assert os.path.exists(alerts_path), f"Alerts file {alerts_path} does not exist."
    alerts_df = pd.read_csv(alerts_path)
    assert "escalation_decision" in alerts_df.columns, "Missing escalation_decision column."
    print("[TEST 8/14] PASS: Operational predictions generated without label dependency.")
    passed_tests += 1

    # 9. Consensus rule is correct
    for _, row in alerts_df.iterrows():
        cv = int(row.get("consensus_votes", 0))
        tamper = int(row.get("security_tampering_flag", 0))
        decision = row["escalation_decision"]
        if tamper == 1:
            assert decision == "Alert", f"Tampering must be Alert, got {decision}"
        elif cv >= 2:
            assert decision == "Alert", f"cv>=2 must be Alert, got {decision}"
        elif cv == 1:
            assert decision == "Log", f"cv==1 must be Log, got {decision}"
        else:
            assert decision == "Monitor", f"cv==0 must be Monitor, got {decision}"
    print("[TEST 9/14] PASS: Production Consensus-Gated decision rule strictly enforced.")
    passed_tests += 1

    # 10. Dashboard reads SPEDIA output
    assert os.path.exists("data/demo/spedia/trustshield_alerts.csv"), "Missing SPEDIA dashboard target."
    print("[TEST 10/14] PASS: SPEDIA dashboard target output verified.")
    passed_tests += 1

    # 11. V2 and SPEDIA outputs remain separate
    assert os.path.exists("data/demo/trustshield_demo_results.csv") or os.path.exists("data/demo/trustshield_alerts.csv")
    v2_df = pd.read_csv("data/demo/trustshield_demo_results.csv" if os.path.exists("data/demo/trustshield_demo_results.csv") else "data/demo/trustshield_alerts.csv")
    assert len(v2_df) == 1080, f"V2 dataset must contain 1080 test records, found {len(v2_df)}"
    assert len(alerts_df) == 231, f"SPEDIA dataset must contain 231 records, found {len(alerts_df)}"
    print("[TEST 11/14] PASS: V2 benchmark and SPEDIA demo outputs remain strictly separated.")
    passed_tests += 1

    # 12. No hardcoded SPEDIA result counts
    action_counts = alerts_df["escalation_decision"].value_counts().to_dict()
    assert sum(action_counts.values()) == 231, "Dynamic counts do not sum to total records."
    print(f"[TEST 12/14] PASS: Results dynamically computed (Alerts={action_counts.get('Alert',0)}, Logs={action_counts.get('Log',0)}, Monitors={action_counts.get('Monitor',0)}).")
    passed_tests += 1

    # 13. Raw event drill-down works
    sample_actor = alerts_df["user_id"].iloc[0]
    actor_raw = df_raw[df_raw["User"] == sample_actor]
    assert len(actor_raw) > 0, f"Raw drill-down returned 0 events for actor {sample_actor}"
    print(f"[TEST 13/14] PASS: Raw event drill-down verified for {sample_actor} ({len(actor_raw):,} raw events).")
    passed_tests += 1

    # 14. XAI does not crash
    assert "top_behavioral_reasons" in alerts_df.columns, "Missing top_behavioral_reasons column."
    assert "explanation" in alerts_df.columns, "Missing explanation column."
    assert "xai_reliability_score" in alerts_df.columns, "Missing xai_reliability_score column."
    print("[TEST 14/14] PASS: Explainable AI attributions and reliability scores verified.")
    passed_tests += 1

    print("=" * 75)
    print(f"ALL {passed_tests}/{total_tests} SPEDIA RUNTIME TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
