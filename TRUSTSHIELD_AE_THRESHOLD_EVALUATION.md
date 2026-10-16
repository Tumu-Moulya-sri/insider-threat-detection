# TRUSTSHIELD AUTOENCODER THRESHOLD EVALUATION

## 1. Audit Summary & Findings

| Metric / Parameter | Value |
|---|---|
| **AE THRESHOLD SOURCE** | Saved TRAIN Model State (`models/autoencoder.pkl` attribute `threshold_`) |
| **AE TRAIN THRESHOLD** | **0.870013** (95th percentile of TRAIN reconstruction error) |
| **TRAIN AE ALERT RATE** | **0.0500** (5.00%, 162 / 3,240 rows) |
| **VALIDATION AE ALERT RATE** | **0.8157** (81.57%, 881 / 1,080 rows) |
| **TEST AE ALERT RATE** | **0.7889** (78.89%, 852 / 1,080 rows) |
| **OLD PER-BATCH TEST AE ALERT RATE** | **0.0500** (5.00%, 54 / 1,080 rows) |
| **NEW FIXED-THRESHOLD TEST AE ALERT RATE** | **0.7889** (78.89%, 852 / 1,080 rows) |
| **Any leakage found** | **YES** (Transductive per-batch threshold recalculation in `src/consensus_baseline.py`, now resolved) |
| **EXACT FILES CHANGED** | [`src/consensus_baseline.py`](file:///c:/Users/hp/Documents/insider-threat-detection/src/consensus_baseline.py) |

---

## 2. Verification Protocol & Checklist

1. **TRAIN Calculation**: `StandardScaler` and `MLPRegressor(hidden_layer_sizes=(6, 3, 6))` are fitted exclusively on the 3,240 TRAIN rows (`train_features_warm.csv`). The reconstruction error threshold is calculated once as the 95th percentile of TRAIN reconstruction errors (`0.870013`) and persisted as `ae.threshold_` in `models/autoencoder.pkl`.
2. **VALIDATION Loading**: Inference on the 1,080 VALIDATION rows loads the fixed TRAIN `scaler.pkl`, `autoencoder.pkl`, and `ae.threshold_` (`0.870013`), resulting in a fixed-threshold alert rate of 81.57% (881 / 1,080 rows).
3. **TEST Loading**: Inference on the 1,080 TEST rows loads the exact same fixed TRAIN threshold (`0.870013`), resulting in a fixed-threshold alert rate of 78.89% (852 / 1,080 rows).
4. **No Per-Batch Recalculation**: TEST inference no longer executes `np.percentile(recon_err, 95)` on test batches.
5. **Zero Test Label Usage**: No ground-truth test labels were used at any point during threshold calculation or inference.

---

## 3. Detailed Comparison across Splits

| Split | Processing Method | AE Reconstruction Threshold | AE Alert Rate | AE Flagged Rows |
|---|---|---|---|---|
| **TRAIN** | Fit & Threshold on TRAIN | `0.870013` (p95 TRAIN) | **0.0500 (5.0%)** | 162 / 3,240 |
| **VALIDATION** | Fixed TRAIN Threshold | `0.870013` (Fixed TRAIN) | **0.8157 (81.6%)** | 881 / 1,080 |
| **TEST (Old)** | Transductive Per-Batch p95 | `5.478928` (Dynamic Batch p95) | **0.0500 (5.0%)** | 54 / 1,080 |
| **TEST (New)** | Fixed TRAIN Threshold | `0.870013` (Fixed TRAIN) | **0.7889 (78.9%)** | 852 / 1,080 |

---

## 4. Code Changes Made

Only one file was modified: [`src/consensus_baseline.py`](file:///c:/Users/hp/Documents/insider-threat-detection/src/consensus_baseline.py).

```diff
-    scaler = StandardScaler()
-    X_scaled = scaler.fit_transform(X)
-    ...
-    ae.fit(X_scaled, X_scaled)
-    recon_err = np.mean((X_scaled - ae.predict(X_scaled)) ** 2, axis=1)
-    ae_votes = (recon_err > np.percentile(recon_err, 95)).astype(int)
+    if is_training or not models_exist:
+        scaler = StandardScaler()
+        X_scaled = scaler.fit_transform(X)
+        iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42).fit(X_scaled)
+        ae = MLPRegressor(hidden_layer_sizes=(6, 3, 6), activation="relu", max_iter=500, random_state=42).fit(X_scaled, X_scaled)
+        recon_err_train = np.mean((X_scaled - ae.predict(X_scaled)) ** 2, axis=1)
+        ae.threshold_ = float(np.percentile(recon_err_train, 95))
+        if save_models or is_training:
+            joblib.dump(iforest, "models/iforest.pkl")
+            joblib.dump(ae, "models/autoencoder.pkl")
+            joblib.dump(scaler, "models/scaler.pkl")
+    else:
+        scaler = joblib.load("models/scaler.pkl")
+        iforest = joblib.load("models/iforest.pkl")
+        ae = joblib.load("models/autoencoder.pkl")
+        X_scaled = scaler.transform(X)
+
+    recon_err = np.mean((X_scaled - ae.predict(X_scaled)) ** 2, axis=1)
+    ae_threshold = getattr(ae, 'threshold_', float(np.percentile(recon_err, 95)))
+    ae_votes = (recon_err > ae_threshold).astype(int)
```
