# TRUSTSHIELD XAI / SHAP EVALUATION

## 1. Audit Summary & Required Metrics

| Metric / Parameter | Result |
|---|---|
| **SHAP STATUS** | **OPERATIONAL & LEAKAGE-SAFE** |
| **TOTAL ALERTS** | **34** (Alert stage) / **110** (Total Escalated Alert + Log) |
| **REAL SHAP EXPLANATIONS** | **107** (Full Test Set) / **0** (Single-Class Subset) |
| **FALLBACK EXPLANATIONS** | **3** (Full Test Set) / **34** (Single-Class Subset) |
| **UNEXPLAINED ALERTS** | **0** |
| **TEST LABELS USED** | **NO** |
| **EXACT FILES CHANGED** | [`src/explainability.py`](file:///c:/Users/hp/Documents/insider-threat-detection/src/explainability.py) |

---

## 2. Root Cause Analysis & Fix Verification

1. **Defect Identified**: When the surrogate model was trained on a dataset or subset where `y_target` contained only a single class (e.g. only `Alert` rows or only `Monitor` rows), `surrogate.predict_proba(X)` returned a single-column matrix of shape `(N, 1)`. Calling `surrogate.predict_proba(X)[:, 1]` raised `IndexError: index 1 is out of bounds for axis 1 with size 1`.
2. **Leakage-Safe Resolution**:
   - [`src/explainability.py`](file:///c:/Users/hp/Documents/insider-threat-detection/src/explainability.py) inspects `len(np.unique(y_target))` prior to surrogate fitting.
   - When 2 classes are present (`Alert`/`Log` vs `Monitor`), a two-class `RandomForestClassifier` and `shap.TreeExplainer` compute genuine SHAP feature impacts (`REAL SHAP EXPLANATIONS`).
   - When only 1 class is present, genuine SHAP is bypassed to avoid fabricating values. The system falls back to top baseline z-score deviations (`FALLBACK EXPLANATIONS`).
   - All escalated records receive complete, human-understandable evidence chains without crashing.
   - Zero test ground-truth labels (`is_insider_ground_truth` / `insider_scenario`) are accessed at any stage.

---

## 3. Evaluation Across Operational Modes

| Operational Context | Total Escalated Records | Real SHAP Explanations | Fallback Explanations | Unexplained Alerts | Execution Result |
|---|---|---|---|---|---|
| **Full Test Set Inference (2 classes)** | 110 (34 Alert, 76 Log) | **107** | **3** | **0** | **PASSED** (Genuine SHAP explanations) |
| **Single-Class Subset (34 Alert rows)** | 34 | **0** | **34** | **0** | **PASSED** (Clean Evidence Fallback, No Crash) |
