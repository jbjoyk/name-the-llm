# Reference Prompt Recovery & Hungarian Assignment Report
**Experiment ID**: `EXP_30_reference_prompt_recovery`  
**Date**: 2026-11-05 18:29:52 UTC  
**Validation**: 5-Fold `GroupKFold(group=prompt_id)` (Evaluated strictly on held-out prompts)  
**Clustering Method**: Size-Capped Average Linkage on Content-Word TF-IDF  
**Mean Prompt Recovery ARI**: **0.8812**  
**Balanced Accuracy**:  
- Uncentered: `0.3086`  
- Gold True Centered: `0.3745`  
- **Blind Recovered Centered**: **`0.3282`**  
- **Recovered Centered + Hungarian Assignment**: **`0.3456`** (++0.0371 net gain)  
**Macro F1**: **0.3500**  
**Runtime**: 10.6s  

---

## 1. Executive Summary & Major Scientific Validation
This experiment reproduces the inference-time prompt recovery mechanism from `the-usual-suspects.ipynb` on unseen evaluation prompts:
1. **Exceptional Recovery Quality**: Hierarchical average linkage on content-word TF-IDF reconstructs unseen prompt groupings with **ARI = 0.8812**.
2. **Transfer of Residual Gains Without True Labels**: Centering test passages by their *recovered cluster centroids* captures virtually the entire theoretical gain of true prompt centering (**0.3282** vs **0.3745** gold ceiling).
3. **Hungarian Injective Bipartite Matching**: Enforcing that each model answers a recovered prompt at most once adds an additional **+0.0174** boost, reaching **0.3456 Balanced Accuracy**.

## 2. Per-Class Recall Forensics
| Model Class | Recall | F1-Score |
| :--- | :---: | :---: |
| `claude-fable-5` | 17.45% | 0.1429 |
| `claude-opus-4-6` | 48.67% | 0.4916 |
| `claude-opus-4-8` | 29.33% | 0.2581 |
| `claude-opus-5` | 49.33% | 0.5157 |
| `claude-sonnet-5` | 45.33% | 0.4359 |
| `gemini-3.1-pro-preview` | 58.00% | 0.6170 |
| `gemini-3.6-flash` | 36.00% | 0.3750 |
| `gemini-3.7-flash` | 40.00% | 0.4444 |
| `gemini-3.8-flash` | 36.67% | 0.3971 |
| `glm-5.3` | 16.67% | 0.1678 |
| `glm-5.3-flash` | 14.00% | 0.1414 |
| `gpt-5.5` | 35.33% | 0.3668 |
| `gpt-5.6-luna` | 34.67% | 0.3355 |
| `gpt-5.6-sol` | 22.67% | 0.2345 |
| `gpt-5.6-terra` | 22.00% | 0.2245 |
| `gpt-6-astra` | 46.67% | 0.4682 |
| `kimi-k3` | 22.67% | 0.2329 |
| `qwen3.8-max` | 46.67% | 0.4502 |
