# SPEDIA Feature Mapping & Schema Compatibility Documentation

## 1. Dataset Overview

- **Source File**: `data/external/spedia/logs_SPEDIA_annotated_en.csv`
- **Total Raw Events**: 72,250 log entries (70,878 with valid non-null User and Timestamp)
- **Actors / Users**: 17 unique accounts (`camilo`, `humberto`, `irene`, `olaya`, `luis`, `ubuntu`, `root`, etc.)
- **Date Range**: 2025-03-06 to 2025-03-30 (25 calendar days)
- **Aggregated Daily Records**: 231 user-day vectors (`data/external/spedia/spedia_features.csv`)

> [!NOTE]
> **Dataset Nature**: SPEDIA contains malicious activity performed by real participants in a controlled cybersecurity exercise, combined with simulated benign enterprise activity and CERT-derived synthetic events. It is NOT purely uniform enterprise telemetry.

---

## 2. Feature-by-Feature Mapping Table

The table below documents how raw SPEDIA log fields map to TrustShield daily behavioral feature vectors:

| TrustShield Daily Feature | Raw SPEDIA Source Fields | Extraction / Aggregation Logic | Semantic Compatibility with TrustShield V2 |
|---|---|---|---|
| `num_logins` | `Action == 'Login'` | Daily count of successful authentication events | **Directly Compatible** |
| `login_failed` | `Action in ['Login Failed', 'Multiple failed logins']` | Daily count of failed authentication attempts | **Directly Compatible** |
| `off_hours_events` | `Timestamp` (Hour < 8, Hour $\ge$ 18, or Weekend) | Total actions occurring outside standard 08:00–18:00 weekday hours | **Directly Compatible** (Corresponds to `off_hours_logins`) |
| `commands_executed` | `Action == 'Command executed'`, `Activity == 'command'` | System commands recorded by `auditd` | **SPEDIA Specific** (Not modeled in V2 synthetic logs) |
| `high_level_commands` | `Level >= 8.0` | Suspicious/privileged command execution level | **SPEDIA Specific** |
| `files_accessed` | `Action == 'File accessed'` | Count of files read/opened | **Directly Compatible** (Corresponds to `files_accessed`) |
| `files_modified` | `Action == 'File modified'` | File integrity change events (`syscheck_integrity_changed`) | **Partially Compatible** |
| `files_added` / `deleted` | `Action in ['File added', 'File deleted']` | File creation and deletion counts | **Partially Compatible** |
| `emails_sent` | `Action == 'Email sent'`, `Activity == 'email'` | Daily outgoing email volume | **Directly Compatible** (Corresponds to `emails_sent`) |
| `external_emails` | `To` field domain check | Count of emails sent to non-corporate domains | **Directly Compatible** (Corresponds to `emails_external`) |
| `total_email_size_kb` | `Size` field summation | Total email payload volume in KB | **Partially Compatible** (Corresponds to `data_transferred_mb`) |
| `http_requests` / `unique_urls` | `Activity == 'http'`, `Url` | Web browsing volume and unique destination URLs | **Partially Compatible** |
| `usb_connects` | `Action == 'Connect'`, `Activity == 'device'` | Removable media / USB connections | **Directly Compatible** (Corresponds to `usb_connections`) |
| `spedia_is_malicious_day` | `Anomaly == 1` | 1 if any event on that day was annotated as exercise attack | **Ground Truth Label** (Held out for evaluation only) |

---

## 3. Schema Incompatibilities & Domain Shift

1. **Host-Level Linux Telemetry**: SPEDIA logs include `auditd` command executions and `syscheck` OS file integrity events, which are typical of endpoint Linux sensors but absent in TrustShield V2's Windows-style Active Directory and file server log schema.
2. **Event Distribution Asymmetry**: 57.5% of raw events in SPEDIA are labeled as anomalous/exercise attack, reflecting intensive red-team exercises, whereas enterprise insider threats represent $< 1\%$ of baseline activity.
3. **Zero-Shot Transfer Caution**: Direct application of V2 anomaly thresholds without recalibration reflects domain shift rather than model defect.
