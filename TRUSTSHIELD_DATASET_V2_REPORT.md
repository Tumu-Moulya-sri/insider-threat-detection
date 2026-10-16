# TRUSTSHIELD DATASET V2 REPORT

## Overview
Comparison between the original synthetic dataset (V1) and the corrected dataset (V2) after fixing label bugs and removing the abrupt distribution shift. All TrustShield pipeline components remain unchanged.

## V1 Summary (original)
- Total rows: 1080
- Insider rows: 97
- Benign rows: 983
- Test date range: 2026-06-12 to 2026-06-17
- Confusion matrix (TP, FN, FP, TN): 96, 1, 963, 20

## V2 Summary (after fixes)
- Total rows: 1080
- Insider rows: 11
- Benign rows: 1069
- Test date range: 2026-06-12 to 2026-06-29
- Confusion matrix (TP, FN, FP, TN): 0, 11, 149, 920

### Performance Metrics (V2)
- Precision: 0.000
- Recall: 0.000
- F1 Score: 0.000
- False Positive Rate: 0.139
- False Negative Rate: 1.000

### Model Alert Rates (V2 test set)
