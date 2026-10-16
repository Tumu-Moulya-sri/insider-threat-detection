# TRUSTSHIELD WARM-START EVALUATION

## 1. Cold-Start Problem

When the TrustShield pipeline ran on `test_input.csv`, `shrinkage_baseline.py` initialized
each user's EWMA history from **zero**. On the first test day, a user with a perfectly normal
`files_downloaded = 5` produced z = +5.0 (baseline = 0, variance = 0.001). The statistical
detector threshold is z > 3, so virtually every benign user tripped it on day 1.

**Result:** Statistical detector flagged 95.28% of all test rows, producing 1,053 false positives
out of 1,069 benign rows (FPR = 98.5%).

## 2. Training Baseline State

The `user_history` dictionary at the end of the training pass contains, for each user and
each of the 15 metrics:

| Field | Type | Description |
|-------|------|-------------|
| `ewma_mean` | float | Exponentially weighted moving average of the metric |
| `ewma_var` | float | EWMA variance estimate |
| `n` | int | Number of observations seen (54 days for a full-train user) |
| `prev_W` | float | Shrinkage weight from the last training observation |
| `current_project` | str | User's current project assignment |
| `department` | str | User's department |

Additionally saved:
- `priors_lookup`: cohort prior means/variances keyed by (role, project_id, clearance_level)
- `metrics`: ordered list of 15 metric column names

## 3. State Persistence

Training state is persisted via `joblib.dump()` to:

```
models/training_baseline_state.pkl
```

Contains 60 users x 15 metrics of EWMA state plus cohort priors. File is written at the
end of the training shrinkage pass using the new `save_state_path` parameter. No test
information is included.

## 4. Test Initialization

When `baseline_state_path` is provided, `apply_shrinkage_baseline()`:

1. Loads the saved state from `models/training_baseline_state.pkl`
2. Restores `user_history`, `priors_lookup`, and `metrics` from training
3. For each test user in the saved state, continues EWMA from the training endpoint
4. For any new user not in training, falls back to the existing cohort-prior mechanism
5. All 60 test users had training history (0 required cohort-only cold start)

## 5. Causal Scoring Verification

User U1000, metric `files_downloaded`:

| Date | Baseline (before scoring) | Observed | Z-score |
|------|---------------------------|----------|---------|
| 2026-06-12 | **3.4813** | 9.00 | +1.912 |
| 2026-06-13 | 3.2411 | 1.00 | -0.789 |
| 2026-06-14 | 3.0241 | 1.00 | -0.729 |
| 2026-06-15 | 3.1188 | 4.00 | +0.332 |
| 2026-06-16 | 3.4933 | 7.00 | +1.267 |
| 2026-06-17 | 3.5427 | 4.00 | +0.174 |
| 2026-06-18 | 4.0201 | 9.00 | +1.678 |
| 2026-06-19 | 3.9216 | 3.00 | -0.325 |

**Key observation:** Day 1 baseline is **3.48** (carried from training), not 0.0.
`files_downloaded = 9` now produces z = +1.91 instead of +9.0.

**CURRENT-ROW CAUSALITY: SAFE** - Each observation is scored against the prior baseline,
then the baseline is updated afterward.

## 6. Before / After Metrics

| Metric | V2 Cold-Start | V2 Warm-Start | Change |
|--------|---------------|---------------|--------|
| IF alert rate | 5.00% | 5.00% | - |
| AE alert rate | 5.00% | 5.00% | - |
| **Stat alert rate** | **95.28%** | **4.17%** | **-91.11pp** |
| Consensus rate | 7.41% | 3.15% | -4.26pp |
| Precision | 0.010 | 0.014 | +0.004 |
| Recall | 1.000 | 0.182 | -0.818 |
| F1 | 0.020 | 0.025 | +0.005 |
| **FPR** | **0.985** | **0.135** | **-0.850** |
| FNR | 0.000 | 0.818 | +0.818 |
| TP | 11 | 2 | -9 |
| **FP** | **1,053** | **144** | **-909** |
| FN | 0 | 9 | +9 |
| TN | 16 | 925 | +909 |

## 7. False Positive Analysis

### Cold-start FPs removed: 909

The warm-start eliminated the statistical detector's massive over-triggering. The stat
alert rate dropped from 95.28% to 4.17%.

### Remaining FPs: 144

| Consensus votes | FP count |
|-----------------|----------|
| 0/3 | 128 |
| 1/3 | 10 |
| 2/3 | 4 |
| 3/3 | 2 |

**Key finding:** 128 of 144 remaining FPs (89%) have **0/3 consensus votes** and come
from neither the statistical detector, nor IF, nor AE. These are rows escalated purely
by the **risk scoring / escalation engine** based on cumulative risk score, context flags
(resignation notice, disciplinary case), or other non-consensus features.

The remaining 16 FPs (11%) come from individual detector triggers:
- IF flagged 10
- AE flagged 9
- Statistical flagged 5

These represent genuine benign anomaly patterns (unusual but legitimate behaviour).

## 8. False Negative Analysis

### New False Negatives: 9

The warm-start introduced 9 FNs that were previously caught only because the cold-start
inflated z-scores:

| User | Date | Scenario | IF | AE | Stat | Consensus | Risk | Decision |
|------|------|----------|----|----|------|-----------|------|----------|
| U1040 | 2026-06-13 | standard_exfiltration | 1 | 1 | 1 | 3/3 | 65.67 | Monitor |
| U1015 | 2026-06-16 | cross_project_exploration | 0 | 0 | 0 | 0/3 | 17.75 | Log |
| U1007 | 2026-06-20 | standard_exfiltration | 1 | 1 | 0 | 2/3 | 46.64 | Log |
| U1015 | 2026-06-20 | cross_project_exploration | 0 | 0 | 0 | 0/3 | 20.37 | Monitor |
| U1040 | 2026-06-22 | standard_exfiltration | 1 | 1 | 1 | 3/3 | 53.69 | Monitor |
| U1007 | 2026-06-23 | standard_exfiltration | 1 | 1 | 0 | 2/3 | 63.61 | Monitor |
| U1007 | 2026-06-27 | standard_exfiltration | 1 | 1 | 0 | 2/3 | 48.49 | Monitor |
| U1017 | 2026-06-27 | slow_subtle_insider | 0 | 0 | 0 | 0/3 | 19.31 | Monitor |
| U1015 | 2026-06-28 | cross_project_exploration | 0 | 0 | 0 | 0/3 | 11.49 | Monitor |

**Critical observation:** Several insiders (U1040, U1007) have consensus votes of 2/3 or
even 3/3 with risk scores of 46-66, but the escalation engine assigns them `Monitor` or
`Log` instead of `Alert`. This indicates the **escalation engine's threshold for Alert**
is calibrated too high in the warm-start scenario — it was previously overwhelmed by
cold-start noise and its thresholds adapted to that inflated baseline.

**Slow/subtle insider U1017** is completely invisible to all three detectors (0/0/0 votes,
risk = 19.31). This is expected: the slow/subtle scenario adds only 2-6 extra file
downloads and 20-50 MB transfer, which falls within the user's training baseline range.

## 9. Remaining Issues

1. **Escalation engine calibration:** The escalation engine's cost/threshold model was
   fitted on the cold-start-inflated feature distribution. With warm-start, the risk
   scores are much lower for genuine insiders, but the escalation thresholds have not
   been recalibrated. This causes 4 insiders with 2/3 or 3/3 consensus and risk > 45
   to receive Monitor instead of Alert.

2. **AE threshold per-batch:** The autoencoder still uses per-batch 95th percentile
   (always exactly 5% alert rate). A train-derived fixed threshold would be more
   principled.

3. **Slow/subtle detection:** The slow_subtle_insider scenario is undetectable by all
   three models when baseline carry-forward is correct. The added signal (2-6 extra
   downloads, 20-50 MB transfer) is too small vs the user's 54-day training variance.

4. **128 non-consensus FPs:** The escalation engine escalates 128 rows with 0/3
   consensus votes purely from cumulative risk score. The risk engine contributes
   false positives independently of the consensus mechanism.
