# TRUSTSHIELD V2 FINAL EVALUATION REPORT

## 1. Dataset Design
- 60 users, 90 days (2026-04-01 to 2026-06-29)
- Total rows: 5,400 | Insider rows: 49 | Benign rows: 5,351
- Insider scenarios: standard_exfiltration (21), contractor_lingering (12),
  cross_project_exploration (13), slow_subtle_insider (3), credential_misuse (0 triggered)
- 4 benign anomaly users included (new_onboarding, cross_project_collab, emergency_access, deadline_crunch)

## 2. Why V1 Failed
V1 assigned all 6 insider users start windows of day 50-80, overlapping perfectly with
the chronological test period (20% = days 54-89). The pipeline also ran shrinkage on
test_input.csv alone, so EWMA started cold at 0 for all users, producing massive z-scores
(mean benign z: 15.4sigma, max: 86.9sigma). All three detectors fired simultaneously on
98% of test rows. Reported metrics: Precision=0.090, Recall=0.989, F1=0.164.

## 3. V2 Corrections
- Replaced INSIDER_WINDOWS with INSIDER_START_DAY (random 0-89) + INSIDER_DAILY_PROB=0.15
- Fixed label bug: insider_scenario='none' now always produces is_insider_ground_truth=0
- Scenarios now distributed across full 90-day timeline
- No TrustShield ML model, threshold, risk weight, or escalation logic was modified

## 4. Train / Val / Test Split (chronological 60/20/20)
| Split | Rows  | Date Range              | Insiders | Benign |
|-------|-------|-------------------------|----------|--------|
| Train | 3,240 | 2026-04-01 to 2026-05-24 | ~29     | ~3,211 |
| Val   | 1,080 | 2026-05-25 to 2026-06-11 | ~9      | ~1,071 |
| Test  | 1,080 | 2026-06-12 to 2026-06-29 | 11      | 1,069  |

## 5. Leakage Verification
- PASS: test ground truth withheld from test_input.csv
- PASS: strict chronological date-based split, no overlap
- PASS: shrinkage baseline updates EWMA AFTER scoring each row

## 6. AE Threshold Verification
**STATUS: ISSUE**

| Measurement | Value |
|---|---|
| TRAIN AE threshold (95th pct of train reconstruction errors) | 0.870013 |
| VAL alert rate using fixed train threshold | 81.57% |
| TEST alert rate using fixed train threshold | 78.89% |
| Production AE alert rate (per-batch 95th pct) | 5.00% |
| Difference | 73.89% |

The production consensus_baseline.py computes np.percentile(recon_err, 95) on the
CURRENT batch being scored. This keeps alert rate fixed at exactly 5% per batch.
It is not label leakage, but it is a calibration issue: the AE cannot signal that
an entire batch is more anomalous than training. The correct approach is to persist
the train-derived threshold and apply it unchanged to val/test.

## 7. Baseline Causality Verification
**STATUS: SAFE**

The EWMA is updated AFTER scoring. Example trace for U1000 (files_downloaded):

| Date       | Baseline (shrunk_mean) | Observed | Z-score |
|------------|------------------------|----------|---------|
| 2026-06-12 | 0.0000                 | 9.00     | +9.000  |
| 2026-06-13 | 0.0000                 | 1.00     | +1.000  |
| 2026-06-15 | 0.3800                 | 4.00     | +2.973  |
| 2026-06-16 | 0.7540                 | 7.00     | +4.576  |

SECONDARY ISSUE (cold-start): The pipeline runs shrinkage on test_input.csv alone.
Every user EWMA resets to 0 on test day 1 -> inflated z-scores for normal behavior.
This is the PRIMARY cause of the 98.5% FPR.

## 8. Overall Test Metrics
| Metric    | Value |
|-----------|-------|
| Rows      | 1,080 |
| Insider   | 11    |
| Benign    | 1,069 |
| TP        | 11    |
| FN        | 0     |
| FP        | 1,053 |
| TN        | 16    |
| Precision | 0.010 |
| Recall    | 1.000 |
| F1        | 0.020 |
| FPR       | 0.985 |
| FNR       | 0.000 |

NOTE: These metrics are NOT credible due to the cold-start baseline problem.

## 9. Scenario-Wise Test Metrics
| Scenario                  | Rows | Users | TP | FN | Recall | Notes |
|---------------------------|------|-------|----|----|--------|-------|
| standard_exfiltration     | 6    | 2     | 6  | 0  | 1.000  |       |
| slow_subtle_insider       | 2    | 1     | 2  | 0  | 1.000  | Latency: 0 days |
| credential_misuse         | 0    | 0     | -  | -  | N/A    | ABSENT - seed-dependent |
| cross_project_exploration | 3    | 1     | 3  | 0  | 1.000  |       |
| contractor_lingering      | 0    | 0     | -  | -  | N/A    | ABSENT - seed-dependent |
| Benign (all)              | 1069 | 60    | -  | -  | -      | FP=1,053, TN=16 |

SLOW_SUBTLE_INSIDER: Recall=1.000, F1=N/A (no benign rows in subset), Latency=0 days

## 10. Consensus Analysis
| Votes | Total | Insider | Benign | Alerts | FP  | Recall | FPR   |
|-------|-------|---------|--------|--------|-----|--------|-------|
| 0/3   | 46    | 0       | 46     | 45     | 45  | 0.000  | 0.978 |
| 1/3   | 954   | 2       | 952    | 939    | 937 | 1.000  | 0.984 |
| 2/3   | 57    | 1       | 56     | 57     | 56  | 1.000  | 1.000 |
| 3/3   | 23    | 8       | 15     | 23     | 15  | 1.000  | 1.000 |

Individual detectors:
| Detector          | Precision | Recall | FPR   | Alerts |
|-------------------|-----------|--------|-------|--------|
| Isolation Forest  | 0.148     | 0.727  | 0.043 | 54     |
| Autoencoder       | 0.167     | 0.818  | 0.042 | 54     |
| Statistical (z>3) | 0.011     | 1.000  | 0.952 | 1,029  |
| Consensus (2/3)   | 0.113     | 0.818  | 0.066 | 80     |

The consensus mechanism does NOT improve precision here.
The statistical detector dominates due to cold-start z-score inflation.
The escalation engine escalates 1/3-vote rows via cumulative risk threshold,
effectively bypassing the 2/3 consensus gate.

## 11. Ablation Study
| Configuration              | Precision | Recall | F1    | FPR   | FNR   |
|----------------------------|-----------|--------|-------|-------|-------|
| A. Full TrustShield        | 0.010     | 1.000  | 0.020 | 0.985 | 0.000 |
| B. No Adaptive Baseline    | 0.011     | 1.000  | 0.021 | 0.952 | 0.000 |
| C. No Consensus (IF only)  | 0.148     | 0.727  | 0.246 | 0.043 | 0.273 |
| D. No Context Escalation   | 0.113     | 0.818  | 0.198 | 0.066 | 0.182 |

KEY FINDING: Isolation Forest alone (C) achieves 7x better precision and 22x lower FPR
than the full pipeline. The statistical detector + escalation engine are the dominant
sources of FPR in the current configuration.

## 12. False Positive Analysis
Total FP: 1,053
- 100% have insider_scenario='none' (normal, legitimate behavior)
- 0/3 vote FPs: 45 (escalated despite all models agreeing 'normal')
- 1/3 vote FPs: 937 (statistical detector alone trips escalation via cumulative risk)
- 2/3 vote FPs: 56 (stat + AE, both inflated by cold-start)
- 3/3 vote FPs: 15 (all three trigger on normal users on cold-start day 1)
Primary cause: EWMA cold-start -> z-scores of +5 to +87 for completely normal behavior

## 13. False Negative Analysis
Total FN: 0. All 11 insider test rows correctly escalated.
NOTE: credential_misuse and contractor_lingering absent from test set this seed.
This is a COVERAGE GAP, not a true negative result.

## 14. XAI Status
**SHAP STATUS: FAILED**

| Item | Status |
|------|--------|
| SHAP executed | No - IndexError: index 1 is out of bounds for axis 1 with size 1 |
| SHAP values generated | No |
| Evidence-chain explanations | No (fallback activated) |
| Real SHAP coverage | 0 / 1,064 alerts |

Root cause: surrogate.predict_proba() returned only a single class column (all
predictions were Alert, no non-alerts in batch). predict_proba(X)[:, 1] fails.

## 15. Limitations
1. CRITICAL: Cold-start baseline - pipeline runs shrinkage on test alone with no
   carry-over of EWMA state from training. Primary driver of 98.5% FPR.
2. HIGH: AE threshold calibration - per-batch 95th percentile (not train-derived fixed).
3. HIGH: SHAP explainability broken (single-class predict_proba crash).
4. MEDIUM: 2 of 5 insider scenarios absent from test set (seed-dependent gap).
5. MEDIUM: Escalation engine bypasses 2/3 consensus gate via cumulative risk score.
6. MEDIUM: slow_subtle_insider represented by only 2 test rows.

## V1 vs V2 Comparison
| Aspect | V1 | V2 |
|---|---|---|
| Label bug fixed | No | Yes |
| Abrupt shift removed | No | Yes |
| Test insider count | 97 | 11 |
| FP count | 963 | 1,053 |
| Precision | 0.090 | 0.010 |
| Recall | 0.989 | 1.000 |
| F1 | 0.164 | 0.020 |
| FPR | 0.980 | 0.985 |
| Cold-start resolved | No | No |
| AE threshold TRAIN-only | No | No |
| SHAP working | No | No |

## Conclusion
The V2 dataset is scientifically correct: no label bug, no abrupt shift, scenarios
distributed across the full timeline. However, the pipeline integration has a critical
unresolved cold-start problem. The next required fix is NOT the dataset - it is carrying
over trained EWMA baseline state to the test evaluation window.
IS V2 READY AS A RESEARCH BASELINE: NO
RECOMMENDED NEXT STEP: Carry over per-user EWMA state from training when running the
pipeline on the test set. This removes the cold-start z-score inflation.
