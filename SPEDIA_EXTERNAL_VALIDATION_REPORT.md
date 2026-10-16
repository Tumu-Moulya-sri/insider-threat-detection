# SPEDIA External Validation Report

## 1. Executive Summary

This report documents the external validation of the TrustShield UEBA feature extraction framework on the **SPEDIA** benchmark dataset.

SPEDIA contains malicious activity performed by real participants in a controlled cybersecurity exercise, combined with simulated benign enterprise activity and CERT-derived synthetic events.

In accordance with strict research integrity:
- **No data contamination**: SPEDIA was **not** mixed into the TrustShield V2 training or test split.
- **No retraining**: V2 production models were **not** retrained or tuned using SPEDIA ground truth.
- **Independent external validation**: SPEDIA was processed independently via [`src/spedia_adapter.py`](file:///c:/Users/hp/Documents/insider-threat-detection/src/spedia_adapter.py) to assess feature transferability and domain generalization.

---

## 2. Dataset Profile & Ground Truth Statistics

| Dataset Dimension | Metric / Value | Details |
|---|---|---|
| **Raw Source File** | `data/external/spedia/logs_SPEDIA_annotated_en.csv` | Multi-source endpoint & network telemetry |
| **Total Raw Events** | **72,250** entries | 70,878 events with complete timestamps & actors |
| **Unique Actors / Users** | **17** accounts | `camilo`, `humberto`, `irene`, `olaya`, `luis`, `ubuntu`, `root`, `impresora` |
| **Date Coverage** | **2025-03-06** to **2025-03-30** | 25 continuous calendar days |
| **Aggregated Daily Vectors** | **231** user-days | Generated via [`src/spedia_adapter.py`](file:///c:/Users/hp/Documents/insider-threat-detection/src/spedia_adapter.py) |
| **Raw Event Ground Truth** | **41,541** Malicious (57.5%) \| **30,709** Benign (42.5%) | Annotated red-team exercise attacks |
| **Daily User Ground Truth** | **144** Malicious Days (62.3%) \| **87** Benign Days (37.7%) | Days with $\ge 1$ red-team attack event |

---

## 3. Telemetry Composition & Activity Breakdown

Raw events in SPEDIA span 6 operational activity categories:
- **HTTP / Web Activity**: 21,723 events (URL access, browsing)
- **Email Activity**: 18,203 events (outbound communications, attachments)
- **File Activity**: 12,405 events (access, modification, creation, deletion via Syscheck)
- **Command Execution**: 10,114 events (system process calls via Auditd)
- **Session / Auth**: 5,399 events (PAM authentication, logins, logoffs)
- **Device Activity**: 4,406 events (USB/hardware connect and disconnect)

---

## 4. Feature Extraction & Schema Compatibility

[`src/spedia_adapter.py`](file:///c:/Users/hp/Documents/insider-threat-detection/src/spedia_adapter.py) extracts daily behavioral vectors without fabricating unsupported schema fields:

| Category | Compatible Features Extracted | Compatibility Status |
|---|---|---|
| **Authentication** | `num_logins`, `login_failed`, `num_logoffs`, `session_events` | **Directly Compatible** |
| **Off-Hours Activity** | `off_hours_events` (Timestamp $< 08:00$ or $\ge 18:00$ or weekend) | **Directly Compatible** |
| **Email & Exfiltration** | `emails_sent`, `external_emails`, `total_attachments`, `total_email_size_kb` | **Directly Compatible** |
| **File Access** | `files_accessed`, `files_modified`, `files_added`, `files_deleted` | **Partially Compatible** |
| **Device Connection** | `usb_connects`, `usb_disconnects`, `device_events` | **Directly Compatible** |
| **Host System Calls** | `commands_executed`, `unique_commands`, `high_level_commands` | **SPEDIA-Specific** (Linux Auditd) |

Full mapping details are documented in [`data/external/spedia/SPEDIA_FEATURE_MAPPING.md`](file:///c:/Users/hp/Documents/insider-threat-detection/data/external/spedia/SPEDIA_FEATURE_MAPPING.md).

---

## 5. External Validation Findings & Transferability

Applying behavioral statistical anomaly detection ($|Z| > 2.5$) across compatible behavioral features on SPEDIA yields:
- **True Positives (TP)**: 7
- **False Negatives (FN)**: 137
- **False Positives (FP)**: 0
- **True Negatives (TN)**: 87
- **Precision**: **1.0000** (100.0%)
- **Recall**: **0.0486** (4.86%)
- **F1 Score**: **0.0927** (9.27%)

### Why Direct Comparison with V2 is NOT Scientifically Valid
1. **Sensor & OS Discrepancy**: SPEDIA contains Linux kernel syscalls (`auditd`) and file integrity monitoring (`syscheck`), whereas TrustShield V2 models enterprise Windows Active Directory, email servers, and network proxies.
2. **Threat Density Asymmetry**: Over 57% of SPEDIA events are malicious red-team exercise attacks occurring in bursts, whereas real enterprise insider threats represent $< 0.1\%$ of daily telemetry.
3. **Threshold Calibration**: Models calibrated on enterprise baseline distributions require domain adaptation before achieving optimal recall on endpoint syscall logs.

---

## 6. Conclusions & Research Recommendations

1. **Successful Feature Adaptation**: The TrustShield adapter successfully transformed 72,250 raw multi-format events into structured daily behavioral feature vectors (`data/external/spedia/spedia_features.csv`).
2. **Zero False Alarms**: The behavioral anomaly baseline maintained 0 False Positives ($\text{Precision}=1.00$) on SPEDIA benign days.
3. **Future Extension**: Future iterations should introduce endpoint syscall anomaly detectors (e.g. Auditd process tree encoders) alongside the existing network/log consensus models.
