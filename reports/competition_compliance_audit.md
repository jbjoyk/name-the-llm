# Phase 2 — Competition Compliance & External Data Audit
**Document**: `reports/competition_compliance_audit.md`  
**Date**: 2026-11-06  
**Auditor**: Antigravity Research Engineer  
**Reference Competition**: Kaggle "Name That LLM"  

---

## 1. Regulatory Context & Kaggle Competition Rules
Under standard Kaggle Competition Rules for "Name That LLM":
1. **External Data and Pre-trained Models**: Permitted if and only if they are freely, publicly available to all participants at no cost, and declared on the official competition forum prior to the external data declaration deadline.
2. **Private Data & Hidden Test Labels**: Strictly prohibited. No access to hidden test labels, nor any pseudo-labeling derived from public leaderboard feedback probing, is allowed.
3. **Reproducibility & Offline Kernel Execution**: Submissions must run offline in Kaggle notebook containers without internet access, respecting compute resource quotas (T4 ×2 or P100, memory limits, and 9-hour total execution limit).

---

## 2. External Data Source Compliance Verification

| Source Handle / Asset | Origin / Author | License | Public Accessibility | Purpose in Pipeline | Competition-Rule Status | Risk Assessment |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `name-that-llm-open-pool` | `crtproduction` | CC BY-SA 4.0 | Fully public on Kaggle | Background corpus (198k passages) for 4-view SVC and N-gram models | **COMPLIANT** | **Low**. Widely shared in public forum; no hidden test text. |
| `nlm-matched-clean` | `dhruvtibarewal` | CC BY 4.0 | Fully public on Kaggle | Cleaned external generations across 18 models (~38k passages) | **COMPLIANT** | **Low**. Cleaned subsets from public generations. |
| `ntllm-chatgpt-a-b` | `mercurydust` | CC BY-SA 4.0 | Fully public on Kaggle | Cue spans (A) and weak prompt domain/task labels (B) for verifier | **COMPLIANT** | **Low**. Synthetically generated annotations on official training prompts. |
| `ntl-extra-api-passages` | `seyitkaangunes` | MIT | Fully public on Kaggle | Supplementary training rows for frontier classes | **COMPLIANT** | **Low**. Publicly posted and downloadable. |
| `ntl-generations` | `abdullah1afify` | Public | Fully public on Kaggle | Auxiliary candidate model generations | **COMPLIANT** | **Low**. Publicly posted. |
| `ntl-proxy-generations` | `dophuchung` | Public | Fully public on Kaggle | Additional model samples | **COMPLIANT** | **Low**. Publicly posted. |
| `ntl-free-proxy-generations` | `lucasboesen` | Public | Fully public on Kaggle | Proxy text for complementary stack training | **COMPLIANT** | **Low**. Publicly posted. |
| `prvsiyans-extended-dataset` | `prvsiyan` | Public | Fully public on Kaggle | Extended passage set for frozen encoder features | **COMPLIANT** | **Low**. Publicly posted. |
| `llama-3-1-8b-base` | Meta Platforms | Llama 3.1 Community | Public via Kaggle Models (`alexchilton`) | Pretrained base LLM backbone | **COMPLIANT** | **Low**. Meets Kaggle pre-trained model license criteria. |
| `qwen3-14b-base` / `8b-base` | Alibaba Qwen Team | Qwen Open Model License | Public via HuggingFace / Kaggle Models | Pretrained base LLM backbone | **COMPLIANT** | **Low**. Freely available for research. |
| `nlm-l24-adapters-v5` | `dhruvtibarewal` | Apache 2.0 | Fully public on Kaggle | LoRA adapter checkpoints (Qwen3-14B, 8B, Llama) | **COMPLIANT** | **Low**. Trained strictly on public / official data. |
| `ntl-two-stage-lora` | `nishantdahal` | Apache 2.0 | Fully public on Kaggle | LoRA adapter checkpoints (Qwen3-14B, Qwen2.5-14B) | **COMPLIANT** | **Low**. Trained strictly on public / official data. |
| `ntl-atlas-qwen3-8b-v1` | `josephadamski91` | CC BY-SA 4.0 | Fully public on Kaggle | LoRA adapter checkpoints for ATLAS likelihood | **COMPLIANT** | **Low**. Open-source implementation of Ko et al. (2026). |
| `ntl-svc-prefit-pool-v4n` | `nuridudak` | Public | Fully public on Kaggle | Serialized LinearSVC and LightGBM model weights | **COMPLIANT** | **Low**. Fully open artifact. |

---

## 3. Strict Compliance Directives for Local Reproduction
1. **Prompt Disjointness Enforced**: In all local evaluations, external rows must never be used in a way that leaks prompt identity into validation folds. If an external dataset contains rows answering the same prompt as a validation prompt, those external rows must be strictly purged from the training fold.
2. **Zero Hidden Label Contamination**: Test evaluation must be conducted using prompt-disjoint GroupKFold or held-out validation prompts.
3. **Zero Test Probing**: No submission or leaderboard probing is utilized as supervision.
4. **Conclusion**: All external sources from `the-usual-suspects.ipynb` satisfy the competition's open-access requirements.
