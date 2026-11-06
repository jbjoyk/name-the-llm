# Phase 8: ATLAS Author-Conditioned Likelihood & Confidence Gating Report
**Experiment ID**: `EXP_37_atlas`  
**Date**: 2026-11-05 19:29:07 UTC  
**Validation Strategy**: 5-Fold `GroupKFold(group=prompt_id)`  
**Base Model Architecture**: `Qwen/Qwen3-8B-Base` (16.39 GB fp16, 36 layers, hidden_size 4096)  
**Adapters**: `josephadamski91/ntl-atlas-qwen3-8b-v1` (18 source LoRA adapters, rank 16, alpha 32, 442 MB)  
**Readout**: Argmax-prod across temperatures $T \in [1, 2, 4, 8]$  
**Hardware Profile**: Dual NVIDIA RTX 3060 (12 GB VRAM each, 24 GB total, ~8.5 GB allocated per GPU)  

---

## 1. Executive Benchmark Summary

| Regime | Description | Routed Samples | GPU-Hours | Balanced Accuracy | Macro F1 | BalAcc / GPU-Hr |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regime A** | Current Cheap Pipeline (EXP_36) | 0 (0.0%) | 0.00 h | `0.6034` | `0.6014` | *baseline* |
| **Regime B** | ATLAS Standalone | 2,699 (100%) | 1.31 h | `1.0000` | - | - |
| **Regime C** | Full ATLAS on All Samples + Fused | 2,699 (100%) | 1.31 h | `0.9766` | - | `0.2845` |
| **Regime D** | ATLAS Only on Uncertain Samples | 1632 (60.5%) | 0.79 h | *routing only* | - | - |
| **Regime E** | **Confidence-Gated ATLAS + Fused** | **1632 (60.5%)** | **0.79 h** | **`0.9374 ± 0.0139`** | **`0.9374`** | **`0.4209`** |

> **Key Discovery**: Confidence gating provides a **39.5% reduction in GPU compute** (0.79 h vs 1.31 h) while selectively targeting the ambiguous sibling pairs (`glm-5.3` vs `flash`, `gpt-5.6` variants, `claude-fable-5`), producing **0.4209 Balanced Accuracy gain per GPU-hour**.

---

## 2. Confidence Gating Analysis

Confidence score is computed as the margin between top-1 and top-2 probabilities:
$$\Delta_i = p_(1, np.int64(2697)) - p_(2, np.int64(2697))$$
- **Mean Margin for Correct Predictions**: `0.4281`
- **Mean Margin for Erroneous Predictions**: `0.2249`
- **Separation**: Correct predictions have **1.90x** larger confidence margin than incorrect predictions.

### Gating Profile Across Thresholds

| Margin Threshold ($\Delta < \theta$) | Samples Routed to ATLAS | Percentage of Dataset | Estimated GPU Runtime |
| :---: | :---: | :---: | :---: |
| `Delta < 0.15` | 918 | 34.0% | 26.8 min |
| `Delta < 0.20` | 1123 | 41.6% | 32.8 min |
| `Delta < 0.25` | 1325 | 49.1% | 38.6 min |
| `Delta < 0.30` | 1466 | 54.3% | 42.8 min |
| `Delta < 0.36` | 1632 | 60.5% | 47.6 min |
| `Delta < 0.45` | 1838 | 68.1% | 53.6 min |

---

## 3. Fold-by-Fold Performance Progression (Regime E)

| Fold | Fold Balanced Accuracy |
| :---: | :---: |
| Fold 1 | `0.9278` |
| Fold 2 | `0.9352` |
| Fold 3 | `0.9648` |
| Fold 4 | `0.9296` |
| Fold 5 | `0.9295` |
| **Mean ± Std** | **`0.9374 ± 0.0139`** |

---

## 4. Per-Class Attribution Forensics (Regime E)

| Model Class | Recall | Precision | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `claude-opus-5` | 99.3% | 96.8% | 0.9803 | 150 |
| `gpt-6-astra` | 98.7% | 97.4% | 0.9801 | 150 |
| `gemini-3.1-pro-preview` | 98.0% | 96.7% | 0.9735 | 150 |
| `claude-opus-4-6` | 97.3% | 96.7% | 0.9701 | 150 |
| `claude-opus-4-8` | 95.3% | 96.6% | 0.9597 | 150 |
| `glm-5.3` | 94.7% | 95.9% | 0.9530 | 150 |
| `glm-5.3-flash` | 94.0% | 95.9% | 0.9495 | 150 |
| `gpt-5.6-sol` | 94.0% | 94.6% | 0.9431 | 150 |
| `gpt-5.6-terra` | 94.0% | 96.6% | 0.9527 | 150 |
| `qwen3.8-max` | 93.3% | 90.9% | 0.9211 | 150 |
| `claude-sonnet-5` | 93.3% | 88.6% | 0.9091 | 150 |
| `gpt-5.6-luna` | 92.7% | 94.6% | 0.9360 | 150 |
| `claude-fable-5` | 92.6% | 93.9% | 0.9324 | 149 |
| `gemini-3.6-flash` | 92.0% | 93.9% | 0.9293 | 150 |
| `gpt-5.5` | 92.0% | 90.2% | 0.9109 | 150 |
| `kimi-k3` | 91.3% | 93.2% | 0.9226 | 150 |
| `gemini-3.8-flash` | 88.0% | 86.8% | 0.8742 | 150 |
| `gemini-3.7-flash` | 86.7% | 88.4% | 0.8754 | 150 |

---

## 5. Hardware Specifications & Optimization
- **Dual GPU Allocation**: Model sharded across GPU 0 and GPU 1 (`device_map='auto'`).
- **Memory Footprint**: ~8.5 GB VRAM per RTX 3060 (leaving ~3.5 GB headroom for KV-cache and chunked attention).
- **Inference Speed**: Batch size 4, source chunk 3, ~1.75 s per passage.
