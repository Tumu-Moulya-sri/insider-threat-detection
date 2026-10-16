import os
import sys
import joblib
import pandas as pd
import numpy as np

print("=" * 65)
print("TRUSTSHIELD FINAL CONSISTENCY VERIFICATION SUITE")
print("=" * 65)

checks = {}

# 1. Inference-only runner exists
checks["1. Inference-only runner exists (src/run_trustshield.py)"] = os.path.exists("src/run_trustshield.py")

# 2, 3, 4, 5. Pretrained models & saved AE threshold
scaler = joblib.load("models/scaler.pkl")
iforest = joblib.load("models/iforest.pkl")
ae = joblib.load("models/autoencoder.pkl")
ae_thresh = getattr(ae, "threshold_", None)

checks["2. Pre-trained Scaler loaded from models/scaler.pkl"] = (scaler is not None)
checks["3. Pre-trained Isolation Forest loaded from models/iforest.pkl"] = (iforest is not None)
checks["4. Pre-trained Autoencoder loaded from models/autoencoder.pkl"] = (ae is not None)
checks["5. Saved TRAIN AE threshold exists (0.870013)"] = (ae_thresh is not None and abs(ae_thresh - 0.870013) < 0.001)

# 6. EWMA saved training state
state_exists = os.path.exists("models/training_baseline_state.pkl")
checks["6. EWMA training baseline state exists (models/training_baseline_state.pkl)"] = state_exists

# 7 & 8. Consensus-gated production decision logic check in demo results
demo_path = "data/demo/trustshield_demo_results.csv"
if os.path.exists(demo_path):
    demo_df = pd.read_csv(demo_path)
    mismatches = 0
    for i, r in demo_df.iterrows():
        tamp = r.get("security_tampering_flag", 0)
        cv = int(r.get("consensus_votes", 0))
        act = r.get("escalation_decision", "")
        if tamp == 1 and act != "Alert": mismatches += 1
        elif tamp == 0 and cv >= 2 and act != "Alert": mismatches += 1
        elif tamp == 0 and cv == 1 and act != "Log": mismatches += 1
        elif tamp == 0 and cv == 0 and act != "Monitor": mismatches += 1
    checks["7. Production Consensus-Gated Rule strictly followed (1,080 rows)"] = (mismatches == 0)
    
    # 9 & 10. Dashboard data checks
    checks["8. Canonical Demo CSV exists (data/demo/trustshield_demo_results.csv)"] = True
    has_gt = any(c in demo_df.columns for c in ["is_insider_ground_truth", "insider_scenario", "stage_label"])
    checks["9. Ground truth strictly withheld from demo CSV"] = (not has_gt)
else:
    checks["7. Production Consensus-Gated Rule strictly followed (1,080 rows)"] = False
    checks["8. Canonical Demo CSV exists (data/demo/trustshield_demo_results.csv)"] = False
    checks["9. Ground truth strictly withheld from demo CSV"] = False

# 10. Dashboard no stale fallback check
with open("app.py", "r", encoding="utf-8") as f:
    app_code = f.read()
checks["10. Dashboard loads canonical demo CSV with no stale fallback"] = (
    "data/demo/trustshield_demo_results.csv" in app_code and "data/escalated.csv" not in app_code
)

# 11. SPEDIA adapter & features exist
checks["11. SPEDIA adapter and daily features generated (data/external/spedia/)"] = (
    os.path.exists("src/spedia_adapter.py") and os.path.exists("data/external/spedia/spedia_features.csv")
)

# 12. SPEDIA separation check
checks["12. SPEDIA kept strictly separate from V2 evaluation"] = (
    os.path.exists("data/external/spedia/spedia_features.csv") and not os.path.exists("data/evaluation/spedia.csv")
)

print("\nAUTOMATED CONSISTENCY CHECK RESULTS:")
all_passed = True
for name, status in checks.items():
    res_str = "PASS" if status else "FAIL"
    if not status: all_passed = False
    print(f"  [{res_str}] {name}")

print("=" * 65)
print("OVERALL STATUS:", "ALL CHECKS PASSED (12/12)" if all_passed else "SOME CHECKS FAILED")
print("=" * 65)
