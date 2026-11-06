# ATLAS Data Overlap & Training Contamination Audit
**Document**: `reports/atlas_data_overlap_audit.md`  
**Date**: 2026-11-06  
**Auditor**: Antigravity Research Engineer  
**Audit Target**: Overlap between ATLAS (`ntl-atlas-qwen3-8b-v1`) training data and `data/train.csv`

---

## 1. Overlap Metrics Summary

| Overlap Metric | Count | Total Dataset Size | Overlap Percentage | Status |
| :--- | :---: | :---: | :---: | :--- |
| **exact_overlap_count** | **2,699** | 2,699 | **100.0%** | **CONTAMINATED** |
| **normalized_overlap_count** | **2,699** | 2,699 | **100.0%** | **CONTAMINATED** |
| **near_duplicate_count** | **0** | 2,699 | 0.0% | *(all pairs are exact duplicates)* |
| **prompt_overlap_count** | **150** | 150 | **100.0%** | **CONTAMINATED** |
| **pair_overlap_count** | **2,699** | 2,699 | **100.0%** | **CONTAMINATED** |

---

## 2. Cryptographic Proof of Data Identity

- **Local Dataset SHA-256 (`data/train.csv`)**:
  `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`
- **ATLAS Training Manifest SHA-256 (`data/competition/train.csv`)**:
  `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`
- **Bundle Training Configuration (`atlas_bundle.json`)**:
  `"data_sha256": "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"`

---

## 3. Sample Overlap IDs (Top 10 Examples)

Every single ID in `data/train.csv` was an explicit training example for the ATLAS adapter:

| Sample ID | Prompt ID | Model Label | Text Snippet | Training Target in ATLAS |
| :---: | :---: | :---: | :--- | :---: |
| `0` | `p000` | `claude-fable-5` | `"Certainly! Let's explore..."` | Adapter Source Index 0 |
| `1` | `p000` | `claude-opus-4-6` | `"To understand this concept..."` | Adapter Source Index 1 |
| `2` | `p000` | `claude-opus-4-8` | `"Here is an overview..."` | Adapter Source Index 2 |
| `3` | `p000` | `claude-opus-5` | `"We can break down..."` | Adapter Source Index 3 |
| `4` | `p000` | `claude-sonnet-5` | `"When analyzing this..."` | Adapter Source Index 4 |
| `5` | `p000` | `gemini-3.1-pro-preview`| `"Below is the detailed..."` | Adapter Source Index 5 |
| `6` | `p000` | `gemini-3.6-flash` | `"The following points..."` | Adapter Source Index 6 |
| `7` | `p000` | `gemini-3.7-flash` | `"Let's look at how..."` | Adapter Source Index 7 |
| `8` | `p000` | `gemini-3.8-flash` | `"In order to address..."` | Adapter Source Index 8 |
| `9` | `p000` | `glm-5.3` | `"Consider the problem..."` | Adapter Source Index 9 |

---

## 4. Formal Contamination Classification

> [!CAUTION]
> **CONTAMINATION VERDICT**: The public ATLAS adapter `josephadamski91/ntl-atlas-qwen3-8b-v1` was trained on all 2,699 samples of the official competition training set. Any evaluation of this adapter against `data/train.csv` evaluates the model on its own training data, measuring training memorization rather than generalization.
