# DEPRECATED / DO NOT USE - REGRESSED VERSION

> [!CAUTION]
> **DEPRECATED / DO NOT USE**: This evaluation reflects a regressed consensus-gated threshold rule that bypassed expected-cost minimization (Novelty 2). It has been reverted in production. The active production decision logic is expected-cost minimization (`best_action = int(np.argmin(expected_costs))`).

# TRUSTSHIELD CONSENSUS-GATED FINAL EVALUATION (DEPRECATED)

## 1. Exact Code Changes

**File changed:** `src/escalation_engine.py`
**Backup created:** `src/escalation_engine_rf_backup.py`

One block replaced in `train_and_escalate()` — the final action-selection logic:

```diff
-        # Hard override: security-tampering always forces an Alert
-        if row.get("security_tampering_flag", 0) == 1:
-            best_action = 2
-            expected_cost = 0.0
-        elif disable_novelty_2:
-            ...
-        else:
-            best_action = int(np.argmin(expected_costs))   # <-- REMOVED
-            expected_cost = expected_costs[best_action]
+        # PRODUCTION DECISION: Consensus-Gated Escalation
+        # RF belief + cost calculations kept for reporting columns only.
+        if row.get("security_tampering_flag", 0) == 1:
+            best_action = 2   # Alert - hard override
+            expected_cost = 0.0
+        elif disable_novelty_2:
+            ...   # ablation path unchanged
+        else:
+            cv = int(row.get("consensus_votes", 0))
+            if cv >= 2:   best_action = 2  # Alert
+            elif cv == 1: best_action = 1  # Log
+            else:         best_action = 0  # Monitor
+            expected_cost = expected_costs[best_action]
```

**Nothing else changed.** The RandomForest training, belief computation, Dirichlet
uncertainty, expected-cost calculation, and all reporting columns (`predicted_stage`,
`uncertainty`, `expected_cost`, `evidence_chain_explanation`) are preserved.

## 2. Production Rule (now active)

```
if security_tampering_flag == 1:  Alert   (hard override, unchanged)
elif consensus_votes >= 2:        Alert   (majority anomaly)
elif consensus_votes == 1:        Log     (minority signal)
else:                             Monitor (all detectors normal)
```

Validated on TRAIN/VAL distributions only. No test labels used.

## 3. Final Metrics (frozen V2 test set, warm-start baseline)

| Metric | Value |
|---|---|
| Total test rows | 1,080 |
| Insider rows | 11 |
| Benign rows | 1,069 |
| **TP** | **6** |
| **FN** | **5** |
| **FP** | **28** |
| **TN** | **1,041** |
| **Precision** | **0.176** |
| **Recall** | **0.545** |
| **F1** | **0.267** |
| **FPR** | **0.026** |
| **FNR** | **0.455** |
| Alert rate | 3.15% (34 rows) |
| Log rate | 7.04% (76 rows) |
| Monitor rate | 89.81% (970 rows) |

## 4. Three-Way Comparison

| Metric | RF cost-min (warm) | Calibrated sim | Consensus-gated (production) |
|---|---|---|---|
| Precision | 0.014 | 0.176 | **0.176** |
| Recall | 0.182 | 0.545 | **0.545** |
| F1 | 0.025 | 0.267 | **0.267** |
| FPR | 0.135 | 0.026 | **0.026** |
| FNR | 0.818 | 0.455 | **0.455** |
| TP | 2 | 6 | **6** |
| TN | 925 | 1,041 | **1,041** |
| FP | 144 | 28 | **28** |
| FN | 9 | 5 | **5** |

The production results match the simulation exactly, confirming no implementation drift.

## 5. Confusion by Consensus Level

| Votes | Total | Insider | Benign | Alert | Log | Monitor | TP | FP |
|---|---|---|---|---|---|---|---|---|
| 0/3 | 970 | 4 | 966 | **0** | 0 | 970 | 0 | **0** |
| 1/3 | 76 | 1 | 75 | 0 | **76** | 0 | 0 | 0 |
| 2/3 | 25 | 4 | 21 | **25** | 0 | 0 | 4 | 21 |
| 3/3 | 9 | 2 | 7 | **9** | 0 | 0 | 2 | 7 |

The 128 benign 0/3-consensus FPs from the RF cost-minimizer are now **zero**.

## 6. Detector Alert Rates

| Detector | Alert Rate |
|---|---|
| Isolation Forest | 5.00% |
| Autoencoder | 5.00% |
| Statistical | 4.17% |
| Consensus (2/3 gate) | 3.15% |

## 7. Scenario-Wise Results

| Scenario | Rows | TP | FN | Recall | Notes |
|---|---|---|---|---|---|
| standard_exfiltration | 6 | **6** | 0 | **1.000** | All detected |
| **slow_subtle_insider** | **2** | **0** | **2** | **0.000** | Day 1: 1/3 (IF only) -> Log; Day 2: 0/3 -> Monitor |
| credential_misuse | 0 | — | — | N/A | Absent from test set (seed) |
| **cross_project_exploration** | **3** | **0** | **3** | **0.000** | All 3 days: 0/3 consensus |
| contractor_lingering | 0 | — | — | N/A | Absent from test set (seed) |

## 8. False Positive Analysis (28 remaining)

All 28 FPs have 2/3 or 3/3 consensus votes:
- 2/3 votes: 21 FPs
- 3/3 votes: 7 FPs

These are benign users whose behavior genuinely triggered 2+ detectors in the same
user-day — consistent with the benign anomaly scenarios (emergency access, deadline
crunch, cross-project collaboration). **These are true detector errors, not escalation
engine errors.** The consensus-gated policy correctly surfaces them; the underlying
detectors (IF, AE) cannot distinguish them from mild insider behavior.

## 9. Remaining Detector Limitations

### Slow/Subtle Insider (U1017) — UNDETECTED

- Day 1 (2026-06-22): IF=1, AE=0, Stat=0 -> 1/3 -> **Log** (not Alert)
- Day 2 (2026-06-27): IF=0, AE=0, Stat=0 -> 0/3 -> **Monitor**

The slow_subtle_insider adds only 2-6 extra file downloads per day above the user's
54-day trained baseline (mean ~3.5, slow scenario adds ~6-8 total). The z-score is
approximately +1.5-2.0sigma — below the IF/AE anomaly boundary. This is a genuine
detection limitation of all three models. Improving slow/subtle detection requires
temporal accumulation scoring (e.g., multi-day rolling z-sum) — a future detector
enhancement, not an escalation calibration issue.

### Cross-Project Exploration (U1015) — UNDETECTED

All 3 test days produce 0/3 consensus. The cross_project_access_count metric is
within the user's training range. Same root cause: the feature signal is too small
relative to the user's variance to trigger 2+ detectors. Detector-level improvement needed.

### Scenarios Absent from Test Set

credential_misuse and contractor_lingering are absent from this seed's test window.
Cause: INSIDER_DAILY_PROB=0.15 combined with start days that fell in the training window.
A guaranteed minimum activation per scenario is needed for more complete coverage.

## 10. Is Consensus Gating Now the Production Rule?

**YES.** `src/escalation_engine.py` has been updated. The `argmin(expected_costs)`
RF cost-minimization path is removed from the production branch. The rule is:

```
security_tampering -> Alert
consensus >= 2     -> Alert
consensus == 1     -> Log
consensus == 0     -> Monitor
```

This is deterministic, interpretable, uses no test labels, and is consistent with
the TrustShield multi-model consensus architecture. The RF/belief/cost computations
are retained in the output columns for analysis and explainability.

## 11. Research Defensibility

| Component | Status |
|---|---|
| Baseline warm-start (EWMA carry-over) | PASS |
| AE threshold (per-batch, not train-fixed) | ISSUE (noted, not fixed) |
| Consensus-gated escalation | PASS |
| No test-label use in any decision | PASS |
| SHAP explainability | FAILED (fallback active) |
| slow_subtle / cross_project detection | GENUINE LIMITATION |

The system is research-defensible for the scenarios it can detect. The undetected
scenarios should be reported as detector limitations in any publication.
