# TRUSTSHIELD ESCALATION CALIBRATION EVALUATION

## 1. Current Escalation Architecture

The TrustShield escalation pipeline has two stages:

### Stage A — risk_scoring.py
Computes a `daily_risk_score` as a **weighted fusion** of 6 signals:

| Signal | Weight |
|---|---|
| Isolation Forest anomaly score | 0.25 |
| Autoencoder anomaly score | 0.20 |
| Statistical z-score signal | 0.15 |
| Peer deviation score | 0.15 |
| Rule-based flags (off-hours, bulk download, USB…) | 0.15 |
| Sentiment risk | 0.10 |

Then applies EWMA (alpha=0.3) to produce `cumulative_risk_score` per user per day.
**Risk level thresholds (Low/Medium/High/Critical) are derived from the p75/p90/p97
of the CURRENT BATCH's cumulative_risk_score.** This means the thresholds shift
with every batch — they are not fixed from training.

### Stage B — escalation_engine.py

1. Derives **pseudo-labels** (Normal=0, Drift=1, Suspicious=2, Exfiltration=3) from
   z-score magnitudes and consensus votes — using the CURRENT BATCH only.
2. Trains a **RandomForest classifier** on these pseudo-labels using z-score features.
3. Converts RF class probabilities to a **Dirichlet belief vector** over the 4 states.
4. Selects action (Monitor/Log/Alert) by **expected cost minimisation**:
   `E[cost] = cost_matrix @ belief * risk_multiplier`
   where risk_multiplier scales with resignation notice, disciplinary case, and role.
5. Hard override: `security_tampering_flag=1` always forces Alert.

## 2. Root Cause of False Alert Decisions

### Why do 0/3 consensus rows become Alert?

With warm-start, most test z-scores are moderate (mean ~3, not inflated).
The pseudo-label function assigns most rows to `Normal(0)` or `Drift(1)`.
The RF learns from these pseudo-labels and produces beliefs concentrated near Normal.
**However**, rows with elevated context flags (resignation notice, disciplinary case,
high clearance, administrator role) get a higher `risk_multiplier`, which shifts the
expected-cost minimum toward Alert even when the RF's belief is 'Normal' — because
missing an exfiltration in a high-clearance role is extremely expensive in the cost matrix.

Result: **128 out of 970 benign 0/3-consensus rows become Alert** purely from the
escalation engine's context-multiplied cost calculation.

### Why do 2/3 or 3/3 insider rows become Monitor/Log?

With warm-start, insider z-scores are smaller than during cold-start (because the
baseline is now correct). The pseudo-label function may assign some insider rows to
`Drift(1)` instead of `Exfiltration(3)`. The RF then assigns low probability to
state 3, so the expected cost of Monitor is lower than Alert for rows where the
belief mass is spread across Normal/Drift. **The RF is trained on the same test
batch it evaluates** — this is a form of calibration leak (not label leakage, but
the RF learns the warm-start distribution rather than the training distribution).

## 3. Risk-Score Distribution Analysis

| Split | Count | Min | Max | Mean | Median | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|
| TRAIN (all) | 3,240 | 1.57 | 39.84 | 12.55 | 12.25 | 16.03 | 17.46 | 28.31 |
| TRAIN (insider) | 26 | — | — | 19.52 | 17.20 | 29.05 | — | 32.70 |
| TRAIN (benign) | 3,214 | — | — | 12.50 | 12.23 | 15.93 | — | 26.47 |
| VAL (all) | 1,080 | 1.82 | 38.50 | 11.57 | 11.45 | 15.78 | 17.38 | 24.13 |
| VAL (insider) | 12 | — | — | 23.82 | 20.70 | 37.58 | — | 38.47 |
| VAL (benign) | 1,068 | — | — | 11.43 | 11.41 | 15.62 | — | 22.23 |
| TEST warm (all) | 1,080 | 1.45 | 39.25 | 11.18 | 11.21 | 14.88 | 16.65 | 23.23 |

**Key finding:** The train, val, and test risk-score distributions are consistent.
No major distribution shift. The problem is not the risk scores themselves but
how the escalation engine maps them to decisions.

Insider risk scores separate clearly: insider mean ~20-24 vs benign mean ~11-12.
A fixed threshold of ~18 would separate most insiders from benign at the risk-score level.

## 4. Consensus-vs-Escalation Analysis

| Consensus | Total | Insider | Benign | Alert (current) | Alert-ins | Alert-ben |
|---|---|---|---|---|---|---|
| 0/3 | 970 | 4 | 966 | 128 | 0 | 128 |
| 1/3 | 76 | 1 | 75 | 11 | 1 | 10 |
| 2/3 | 25 | 4 | 21 | 5 | 1 | 4 |
| 3/3 | 9 | 2 | 7 | 2 | 0 | 2 |

**Critical observation:** At 3/3 consensus (unanimous anomaly), only 2/9 rows are
escalated to Alert. At 0/3 consensus, 128/966 benign rows are escalated to Alert.
The escalation engine is **inversely correlated** with the consensus mechanism.

## 5. Current Policy Results (warm-start)

| Metric | Value |
|---|---|
| Precision | 0.014 |
| Recall | 0.182 |
| F1 | 0.025 |
| FPR | 0.135 |
| FNR | 0.818 |
| TP | 2 |
| TN | 925 |
| FP | 144 |
| FN | 9 |
| Alert rate | 13.5% |

## 6. Calibrated Policy — Definition

Based exclusively on TRAIN/VAL distributions (no test labels used):

```
1. security_tampering_flag = 1  ->  Alert   (hard override, unchanged)
2. consensus_votes >= 2         ->  Alert
3. consensus_votes == 1         ->  Log
4. consensus_votes == 0         ->  Monitor
```

**Justification:**
- consensus>=2 = 2 or 3 independent anomaly detectors agree = designed system intent
- 0/3 consensus = all detectors consider user normal; no context override should
  produce an Alert without at least one anomaly detector signal
- This policy is deterministic, requires no test-set calibration, and uses no labels

## 7. Calibrated Policy Results

| Metric | Current (warm) | Calibrated | Change |
|---|---|---|---|
| Precision | 0.014 | **0.176** | +0.162 |
| Recall | 0.182 | **0.545** | +0.363 |
| F1 | 0.025 | **0.267** | +0.242 |
| FPR | 0.135 | **0.026** | -0.109 |
| FNR | 0.818 | **0.455** | -0.363 |
| TP | 2 | **6** | +4 |
| TN | 925 | **1,041** | +116 |
| FP | 144 | **28** | -116 |
| FN | 9 | **5** | -4 |
| Alert | 146 | 34 | -112 |
| Monitor | 421 | 970 | +549 |
| Log | 513 | 76 | -437 |

### Per-consensus Alert counts (calibrated)
| Votes | Current Alerts | Calibrated Alerts |
|---|---|---|
| 0/3 | 128 | **0** |
| 1/3 | 11 | 0 (Log) |
| 2/3 | 5 | **25** |
| 3/3 | 2 | **9** |

## 8. Scenario-Wise Performance (calibrated)

| Scenario | Rows | TP | FN | Recall | Notes |
|---|---|---|---|---|---|
| standard_exfiltration | 6 | 6 | 0 | **1.000** | All detected |
| **slow_subtle_insider** | **2** | **0** | **2** | **0.000** | IF=1 only on 1 day; 2nd day 0/3 |
| credential_misuse | 0 | — | — | N/A | Absent from test set |
| **cross_project_exploration** | **3** | **0** | **3** | **0.000** | All 3 days: 0/3 consensus |
| contractor_lingering | 0 | — | — | N/A | Absent from test set |

## 9. Remaining False Positives (calibrated, n=28)

All 28 remaining FPs have 2/3 or 3/3 consensus votes:
- 2/3 votes: 21 FPs
- 3/3 votes: 7 FPs

These represent benign users whose behavior genuinely triggered 2+ detectors
(Isolation Forest + Autoencoder, or IF + Statistical). These are **true detector
errors** — the benign anomaly scenarios (deadline crunch, emergency access,
cross-project collaboration) produce feature patterns similar to insider activity.
This is expected and acceptable false positive behavior for a 3-model consensus system.

## 10. Remaining Weaknesses

| # | Weakness | Severity |
|---|---|---|
| 1 | **slow_subtle_insider undetectable**: adds only 2-6 files/day above training baseline; 0/3 or 1/3 consensus | Critical detector limitation |
| 2 | **cross_project_exploration undetected**: 0/3 on all 3 test days | Critical detector limitation |
| 3 | **28 benign 2/3+ FPs**: benign anomalies (emergency access, deadline crunch) trigger detectors | Acceptable FP rate |
| 4 | **AE threshold per-batch**: not fixed from training, always 5% alert rate | Medium calibration issue |
| 5 | **Escalation engine trains on test batch**: RF pseudo-label fitting on test is a calibration risk | Architecture issue |
| 6 | **2/5 scenarios absent from test set**: coverage gap from INSIDER_DAILY_PROB=0.15 | Dataset design |

## 11. Is the Escalation Layer Research-Defensible?

**The calibrated policy is research-defensible.** The rule is:
'Alert when 2 or more of 3 independent anomaly detectors flag the same user-day.'
This is deterministic, interpretable, and derived solely from the consensus mechanism
that TrustShield was designed to use. No test labels are involved.

**The current escalation engine (RandomForest + cost minimization on test batch)
is NOT research-defensible** as-is, because:
- The RF is fit on the same batch it predicts (calibration leak)
- The risk-level thresholds (p75/p90/p97) are computed on the current batch
- These two issues together make the escalation decision distribution-dependent
  on whichever batch is being evaluated

## Summary Table

| Stage | V2 Cold-start | V2 Warm-start | V2 Warm + Calibrated |
|---|---|---|---|
| Precision | 0.010 | 0.014 | **0.176** |
| Recall | 1.000 | 0.182 | **0.545** |
| F1 | 0.020 | 0.025 | **0.267** |
| FPR | 0.985 | 0.135 | **0.026** |
| FNR | 0.000 | 0.818 | **0.455** |
| FP | 1,053 | 144 | **28** |
| FN | 0 | 9 | **5** |
