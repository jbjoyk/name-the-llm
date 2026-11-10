# EXP66 Open Pool Forensic Audit

## 1. Executive Summary
- **Artifact Audited**: `data/external/open_pool_clean.parquet` (212,623 samples, 103,375,180 bytes, SHA256: `0c5b289e9f5f7c3f...3982d5c9`).
- **Research Question**: Does the Open Pool introduce contamination, prompt leakage, or an artificial local validation advantage that fails to transfer to Kaggle?
- **Finding**: **REJECTED AS CAUSE OF THE GAP**.
  1. Exact text passage overlap with `train.csv`: **0 passages (0.0%)**.
  2. Exact text passage overlap with `test.csv`: **0 passages (0.0%)**.
  3. EXP52 runtime kernel (`notebooks/52_dual_lora_kernel.py`) does **NOT load, query, or reference the Open Pool** in any way. EXP52 is a pure dual-backbone transformer ensemble.
  4. While earlier experiments (EXP28, EXP33, EXP54) tested LinearSVC auxiliary heads fitted on the Open Pool, those were discarded because nested CV proved they added zero statistical lift over EXP52.

---

## 2. Dataset Properties & Provenance
| Property | Value |
|---|:---:|
| **Total Rows** | 212,623 |
| **Columns** | `id`, `text`, `model`, `lab`, `prompt_group`, `source_dataset`, `n_words` |
| **Model Classes** | Exactly 18 classes (all 18 competition targets represented) |
| **Class Min / Max Count** | `gemini-3.7-flash` (3,541) to `gemini-3.8-flash` (22,508) |
| **Exact Overlap with train.csv** | **0** |
| **Exact Overlap with test.csv** | **0** |

---

## 3. Potential Leakage & Contamination Assessment
- **Model Name Residues**: Regex scan for model name strings (e.g. "I am Claude", "As GPT-5.5") reveals that the open pool was cleaned of overt conversational greetings and system prompt self-identifications.
- **Local Validation Impact**: Because EXP52's frozen OOF predictions (`oof_exp47_qwen_centered_twin.npy` and `oof_q14v5_antopil_lora.csv`) were derived from models trained strictly on `train.csv` (2,699 samples across 150 prompt groups) and do not incorporate Open Pool features, the Open Pool is completely non-causal for EXP52's OOF performance or its Kaggle LB score.

---

## 4. Verdict
- **Audit Result**: PASS (No contamination, no leakage, zero involvement in production EXP52 kernel).
- **Explanation Power for 0.1553 Gap**: **0% (UNSUPPORTED / NON-FACTOR)**.
