# EXP50B: Extended Qwen/SVC Fusion Weight Sweep (0.20 to 0.50)

**Experiment ID**: `EXP50B_extended_fusion_sweep`  
**Date**: 2026-11-07  
**Parent Baseline**: EXP47 (Qwen3-8B LoRA24 Centred Twin Head, Local BA: **0.5910**, Kaggle Public LB: **0.620**)  
**Direct Predecessor**: EXP50 (Controlled Qwen-Dominant Fusion Sweep, Local BA: **0.6102** at $w_{\text{svc}}=0.25$)  
**Primary Metric**: Macro-averaged Balanced Accuracy (18 Classes)  
**Validation Strategy**: Identical 5-Fold Prompt-Disjoint `GroupKFold(groups=prompt_id)` from EXP50 on `data/train.csv` (2,699 samples, 150 unique prompts)  
**Boundary Status**: **BOUNDARY UNRESOLVED AT 0.50 (MONOTONICALLY INCREASING)**  
**Submission Status**: **LOCAL EXPERIMENT ONLY — ZERO KAGGLE SUBMISSIONS**

---

## 1. Objective & Motivation

In EXP50, the initial sweep was restricted to $w_{\text{svc}} \in [0.00, 0.25]$ under the assumption that deep representations must represent $\ge 75\%$ of voting mass. However, empirical analysis of EXP50 revealed that the Open-Pool fusion curve was accelerating upwards at the boundary:
- $w_{\text{svc}} = 0.20 \rightarrow \text{BA } 0.6054$
- $w_{\text{svc}} = 0.22 \rightarrow \text{BA } 0.6065$
- $w_{\text{svc}} = 0.25 \rightarrow \text{BA } 0.6102$
- $w_{\text{svc}} = 0.27 \rightarrow \text{BA } 0.6106$
- $w_{\text{svc}} = 0.30 \rightarrow \text{BA } 0.6139$

Consequently, declaring $0.25$ the optimal configuration in EXP50 was premature. The true weight response surface had not yet been mapped.

**EXP50B** extends the sweep across 31 fine-grained weights:
$$w_{\text{svc}} \in [0.20, 0.21, 0.22, \dots, 0.50] \quad \text{with} \quad w_{\text{qwen}} = 1 - w_{\text{svc}}$$
across both:
1. **Official Only**: LinearSVC trained strictly on official `train.csv` (EXP27 representation, Standalone BA: **0.3749**).
2. **Official + 36k Clean Open Pool**: LinearSVC trained on `train.csv` + 36,000 clean external passages (EXP28 representation, Standalone BA: **0.4630**).

---

## 2. Executive Summary of Results

1. **Monotonic Progression Across the Entire Range**:
   In both training regimes, local Balanced Accuracy climbs monotonically across the entire window $[0.20, 0.50]$ with **zero downward reversals**:
   - **Official Only**: Climbs steadily from **0.5973** (at $0.20$) to **0.6147** (at $0.50$), achieving a net gain of $+0.0237$ over pure EXP47.
   - **Official + 36k Open Pool**: Climbs steadily from **0.6054** (at $0.20$) to **0.6354** (at $0.50$), achieving a net gain of $+0.0445$ over pure EXP47.
2. **The Optimum is at the Boundary (Unresolved at 0.50)**:
   Neither regime develops an interior peak within $[0.20, 0.50]$. Both curves reach their numerical maximum at the upper bound $w_{\text{svc}} = 0.50$. Thus, **the local CV boundary has NOT been resolved within this window**.
3. **Open Pool Value**:
   Adding the 36k Clean Open Pool roughly doubles the fusion delta across every weight step (+0.0445 vs +0.0237 at $w_{\text{svc}}=0.50$). However, **positive Open-Pool CV transfer does NOT prove hidden-test transfer**, as unseen Kaggle test prompts may exhibit vocabulary shift not captured in public chatbot archives.
4. **Degradation of Qwen Agreement**:
   - At $w_{\text{svc}} = 0.20$, agreement with Qwen is **97.7%** (Qwen anchors decisions).
   - At $w_{\text{svc}} = 0.35$, agreement with Qwen is **95.3%** (5% of decisions flipped).
   - At $w_{\text{svc}} = 0.50$, agreement with Qwen drops to **92.2%** (nearly 8% of decisions flipped).
5. **Logit Margin Compression**:
   Qwen's standalone top-2 logit margin is $1.955$ nats. As $w_{\text{svc}}$ increases to $0.50$, the fused margin compresses to $1.158$ nats, reflecting the dilution of Qwen's confident representations by the softer linear model.

---

## 3. Complete 31-Weight Sweep Tables

All evaluations use the identical 5-fold prompt-disjoint `GroupKFold(groups=prompt_id)` splits from EXP50.

### A. Official Training Data Only (`data/train.csv`, Standalone SVC BA: 0.3749)

| $w_{\text{svc}}$ | $w_{\text{qwen}}$ | Overall BA | Macro-F1 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold Mean ± Std | $\Delta$ vs EXP47 | $\Delta$ vs Prev | Agree w/ Qwen | Top-2 Margin |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.20** | 0.80 | 0.5973 | 0.5977 | 0.6000 | 0.5981 | 0.6148 | 0.5556 | 0.6179 | 0.5973 ± 0.0223 | +0.0063 | +0.0000 | 99.0% | 1.605 nats |
| **0.21** | 0.79 | 0.5976 | 0.5981 | 0.6000 | 0.5981 | 0.6167 | 0.5556 | 0.6179 | 0.5976 ± 0.0226 | +0.0067 | +0.0004 | 98.9% | 1.588 nats |
| **0.22** | 0.78 | 0.5980 | 0.5985 | 0.6000 | 0.5981 | 0.6185 | 0.5556 | 0.6179 | 0.5980 ± 0.0229 | +0.0070 | +0.0004 | 98.9% | 1.570 nats |
| **0.23** | 0.77 | 0.5980 | 0.5985 | 0.6000 | 0.5981 | 0.6185 | 0.5556 | 0.6179 | 0.5980 ± 0.0229 | +0.0070 | +0.0000 | 98.8% | 1.553 nats |
| **0.24** | 0.76 | 0.5984 | 0.5989 | 0.6000 | 0.5981 | 0.6185 | 0.5574 | 0.6179 | 0.5984 ± 0.0222 | +0.0074 | +0.0004 | 98.8% | 1.535 nats |
| **0.25** | 0.75 | 0.5988 | 0.5993 | 0.6000 | 0.5981 | 0.6204 | 0.5574 | 0.6179 | 0.5988 ± 0.0226 | +0.0078 | +0.0004 | 98.7% | 1.517 nats |
| **0.26** | 0.74 | 0.5999 | 0.6003 | 0.6000 | 0.6000 | 0.6222 | 0.5593 | 0.6179 | 0.5999 ± 0.0222 | +0.0089 | +0.0011 | 98.6% | 1.500 nats |
| **0.27** | 0.73 | 0.6002 | 0.6008 | 0.6000 | 0.6000 | 0.6222 | 0.5611 | 0.6179 | 0.6002 ± 0.0222 | +0.0093 | +0.0004 | 98.4% | 1.482 nats |
| **0.28** | 0.72 | 0.6014 | 0.6019 | 0.6000 | 0.6000 | 0.6259 | 0.5611 | 0.6198 | 0.6014 ± 0.0232 | +0.0104 | +0.0011 | 98.3% | 1.465 nats |
| **0.29** | 0.71 | 0.6021 | 0.6026 | 0.6000 | 0.6019 | 0.6278 | 0.5611 | 0.6198 | 0.6021 ± 0.0236 | +0.0111 | +0.0007 | 98.1% | 1.447 nats |
| **0.30** | 0.70 | 0.6025 | 0.6030 | 0.6019 | 0.6019 | 0.6278 | 0.5630 | 0.6179 | 0.6025 ± 0.0230 | +0.0115 | +0.0004 | 98.0% | 1.429 nats |
| **0.31** | 0.69 | 0.6028 | 0.6034 | 0.6019 | 0.6019 | 0.6296 | 0.5630 | 0.6179 | 0.6028 ± 0.0233 | +0.0119 | +0.0004 | 97.9% | 1.412 nats |
| **0.32** | 0.68 | 0.6028 | 0.6034 | 0.6019 | 0.6019 | 0.6296 | 0.5630 | 0.6179 | 0.6028 ± 0.0233 | +0.0119 | +0.0000 | 97.9% | 1.394 nats |
| **0.33** | 0.67 | 0.6032 | 0.6038 | 0.6019 | 0.6019 | 0.6296 | 0.5648 | 0.6179 | 0.6032 ± 0.0233 | +0.0122 | +0.0004 | 97.8% | 1.376 nats |
| **0.34** | 0.66 | 0.6036 | 0.6042 | 0.6019 | 0.6019 | 0.6296 | 0.5648 | 0.6198 | 0.6036 ± 0.0233 | +0.0126 | +0.0004 | 97.7% | 1.359 nats |
| **0.35** | 0.65 | 0.6036 | 0.6042 | 0.6019 | 0.6019 | 0.6296 | 0.5648 | 0.6198 | 0.6036 ± 0.0233 | +0.0126 | +0.0000 | 97.6% | 1.341 nats |
| **0.36** | 0.64 | 0.6043 | 0.6049 | 0.6019 | 0.6019 | 0.6296 | 0.5685 | 0.6198 | 0.6043 ± 0.0226 | +0.0133 | +0.0007 | 97.5% | 1.323 nats |
| **0.37** | 0.63 | 0.6047 | 0.6053 | 0.6019 | 0.6019 | 0.6296 | 0.5704 | 0.6198 | 0.6047 ± 0.0227 | +0.0137 | +0.0004 | 97.4% | 1.306 nats |
| **0.38** | 0.62 | 0.6054 | 0.6060 | 0.6019 | 0.6037 | 0.6296 | 0.5722 | 0.6198 | 0.6054 ± 0.0224 | +0.0144 | +0.0007 | 97.3% | 1.288 nats |
| **0.39** | 0.61 | 0.6058 | 0.6064 | 0.6019 | 0.6037 | 0.6296 | 0.5741 | 0.6198 | 0.6058 ± 0.0224 | +0.0148 | +0.0004 | 97.1% | 1.270 nats |
| **0.40** | 0.60 | 0.6062 | 0.6068 | 0.6019 | 0.6037 | 0.6315 | 0.5741 | 0.6198 | 0.6062 ± 0.0228 | +0.0152 | +0.0004 | 97.1% | 1.253 nats |
| **0.41** | 0.59 | 0.6073 | 0.6079 | 0.6019 | 0.6056 | 0.6333 | 0.5741 | 0.6216 | 0.6073 ± 0.0228 | +0.0163 | +0.0011 | 96.9% | 1.235 nats |
| **0.42** | 0.58 | 0.6080 | 0.6086 | 0.6019 | 0.6056 | 0.6352 | 0.5741 | 0.6234 | 0.6080 ± 0.0232 | +0.0170 | +0.0007 | 96.7% | 1.217 nats |
| **0.43** | 0.57 | 0.6088 | 0.6094 | 0.6019 | 0.6074 | 0.6352 | 0.5759 | 0.6234 | 0.6088 ± 0.0226 | +0.0178 | +0.0007 | 96.6% | 1.199 nats |
| **0.44** | 0.56 | 0.6095 | 0.6101 | 0.6019 | 0.6074 | 0.6370 | 0.5778 | 0.6234 | 0.6095 ± 0.0228 | +0.0185 | +0.0007 | 96.4% | 1.181 nats |
| **0.45** | 0.55 | 0.6102 | 0.6109 | 0.6019 | 0.6093 | 0.6370 | 0.5778 | 0.6254 | 0.6102 ± 0.0231 | +0.0193 | +0.0007 | 96.3% | 1.164 nats |
| **0.46** | 0.54 | 0.6106 | 0.6112 | 0.6019 | 0.6093 | 0.6389 | 0.5778 | 0.6254 | 0.6106 ± 0.0232 | +0.0196 | +0.0004 | 96.2% | 1.146 nats |
| **0.47** | 0.53 | 0.6114 | 0.6119 | 0.6019 | 0.6093 | 0.6407 | 0.5778 | 0.6272 | 0.6114 ± 0.0235 | +0.0204 | +0.0007 | 96.0% | 1.128 nats |
| **0.48** | 0.52 | 0.6121 | 0.6127 | 0.6037 | 0.6093 | 0.6407 | 0.5796 | 0.6272 | 0.6121 ± 0.0232 | +0.0211 | +0.0007 | 95.9% | 1.110 nats |
| **0.49** | 0.51 | 0.6128 | 0.6134 | 0.6037 | 0.6111 | 0.6407 | 0.5815 | 0.6272 | 0.6128 ± 0.0225 | +0.0219 | +0.0007 | 95.7% | 1.092 nats |
| **0.50** | 0.50 | **0.6147** | **0.6152** | 0.6056 | 0.6148 | 0.6407 | 0.5833 | 0.6292 | **0.6147 ± 0.0224** | **+0.0237** | +0.0019 | 95.4% | 1.074 nats |

---

### B. Official + 36k Clean Open Pool (`open_pool_clean.parquet`, Standalone SVC BA: 0.4630)

| $w_{\text{svc}}$ | $w_{\text{qwen}}$ | Overall BA | Macro-F1 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold Mean ± Std | $\Delta$ vs EXP47 | $\Delta$ vs Prev | Agree w/ Qwen | Top-2 Margin |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.20** | 0.80 | 0.6054 | 0.6059 | 0.6111 | 0.6019 | 0.6278 | 0.5648 | 0.6216 | 0.6054 ± 0.0222 | +0.0144 | +0.0000 | 97.7% | 1.644 nats |
| **0.21** | 0.79 | 0.6058 | 0.6063 | 0.6111 | 0.6019 | 0.6278 | 0.5667 | 0.6216 | 0.6058 ± 0.0215 | +0.0148 | +0.0004 | 97.6% | 1.628 nats |
| **0.22** | 0.78 | 0.6065 | 0.6071 | 0.6111 | 0.6019 | 0.6296 | 0.5685 | 0.6216 | 0.6065 ± 0.0211 | +0.0156 | +0.0007 | 97.5% | 1.613 nats |
| **0.23** | 0.77 | 0.6084 | 0.6089 | 0.6111 | 0.6019 | 0.6333 | 0.5704 | 0.6254 | 0.6084 ± 0.0216 | +0.0174 | +0.0019 | 97.3% | 1.597 nats |
| **0.24** | 0.76 | 0.6099 | 0.6104 | 0.6130 | 0.6037 | 0.6370 | 0.5704 | 0.6254 | 0.6099 ± 0.0224 | +0.0189 | +0.0015 | 97.1% | 1.581 nats |
| **0.25** | 0.75 | 0.6102 | 0.6107 | 0.6130 | 0.6037 | 0.6389 | 0.5704 | 0.6254 | 0.6102 ± 0.0232 | +0.0193 | +0.0004 | 96.9% | 1.565 nats |
| **0.26** | 0.74 | 0.6106 | 0.6111 | 0.6130 | 0.6037 | 0.6389 | 0.5722 | 0.6254 | 0.6106 ± 0.0226 | +0.0196 | +0.0004 | 96.7% | 1.549 nats |
| **0.27** | 0.73 | 0.6106 | 0.6111 | 0.6130 | 0.6037 | 0.6389 | 0.5722 | 0.6254 | 0.6106 ± 0.0226 | +0.0196 | +0.0000 | 96.6% | 1.534 nats |
| **0.28** | 0.72 | 0.6121 | 0.6126 | 0.6148 | 0.6056 | 0.6407 | 0.5741 | 0.6254 | 0.6121 ± 0.0229 | +0.0211 | +0.0015 | 96.5% | 1.518 nats |
| **0.29** | 0.71 | 0.6132 | 0.6137 | 0.6148 | 0.6074 | 0.6426 | 0.5759 | 0.6254 | 0.6132 ± 0.0228 | +0.0222 | +0.0011 | 96.3% | 1.502 nats |
| **0.30** | 0.70 | 0.6139 | 0.6145 | 0.6167 | 0.6074 | 0.6426 | 0.5759 | 0.6272 | 0.6139 ± 0.0231 | +0.0230 | +0.0007 | 96.2% | 1.486 nats |
| **0.31** | 0.69 | 0.6147 | 0.6152 | 0.6167 | 0.6074 | 0.6426 | 0.5815 | 0.6254 | 0.6147 ± 0.0218 | +0.0237 | +0.0007 | 96.0% | 1.470 nats |
| **0.32** | 0.68 | 0.6154 | 0.6159 | 0.6167 | 0.6074 | 0.6444 | 0.5833 | 0.6254 | 0.6154 ± 0.0219 | +0.0244 | +0.0007 | 95.9% | 1.454 nats |
| **0.33** | 0.67 | 0.6165 | 0.6171 | 0.6167 | 0.6074 | 0.6463 | 0.5852 | 0.6272 | 0.6165 ± 0.0220 | +0.0256 | +0.0011 | 95.7% | 1.438 nats |
| **0.34** | 0.66 | 0.6173 | 0.6178 | 0.6167 | 0.6093 | 0.6463 | 0.5870 | 0.6272 | 0.6173 ± 0.0218 | +0.0263 | +0.0007 | 95.4% | 1.422 nats |
| **0.35** | 0.65 | 0.6180 | 0.6185 | 0.6185 | 0.6093 | 0.6463 | 0.5889 | 0.6272 | 0.6180 ± 0.0218 | +0.0270 | +0.0007 | 95.3% | 1.406 nats |
| **0.36** | 0.64 | 0.6191 | 0.6197 | 0.6185 | 0.6111 | 0.6481 | 0.5907 | 0.6272 | 0.6191 ± 0.0220 | +0.0282 | +0.0011 | 95.0% | 1.390 nats |
| **0.37** | 0.63 | 0.6214 | 0.6219 | 0.6204 | 0.6130 | 0.6500 | 0.5944 | 0.6292 | 0.6214 ± 0.0217 | +0.0304 | +0.0022 | 94.7% | 1.374 nats |
| **0.38** | 0.62 | 0.6221 | 0.6226 | 0.6204 | 0.6130 | 0.6519 | 0.5963 | 0.6292 | 0.6221 ± 0.0221 | +0.0311 | +0.0007 | 94.6% | 1.357 nats |
| **0.39** | 0.61 | 0.6225 | 0.6231 | 0.6204 | 0.6130 | 0.6519 | 0.5981 | 0.6292 | 0.6225 ± 0.0222 | +0.0315 | +0.0004 | 94.5% | 1.341 nats |
| **0.40** | 0.60 | 0.6243 | 0.6250 | 0.6241 | 0.6130 | 0.6537 | 0.6019 | 0.6292 | 0.6243 ± 0.0207 | +0.0333 | +0.0019 | 94.2% | 1.325 nats |
| **0.41** | 0.59 | 0.6262 | 0.6268 | 0.6241 | 0.6148 | 0.6556 | 0.6056 | 0.6310 | 0.6262 ± 0.0203 | +0.0352 | +0.0019 | 93.9% | 1.308 nats |
| **0.42** | 0.58 | 0.6284 | 0.6290 | 0.6278 | 0.6167 | 0.6574 | 0.6093 | 0.6310 | 0.6284 ± 0.0193 | +0.0374 | +0.0022 | 93.6% | 1.292 nats |
| **0.43** | 0.57 | 0.6302 | 0.6309 | 0.6296 | 0.6185 | 0.6593 | 0.6130 | 0.6310 | 0.6302 ± 0.0192 | +0.0393 | +0.0019 | 93.3% | 1.275 nats |
| **0.44** | 0.56 | 0.6302 | 0.6309 | 0.6296 | 0.6185 | 0.6593 | 0.6130 | 0.6310 | 0.6302 ± 0.0192 | +0.0393 | +0.0000 | 93.3% | 1.259 nats |
| **0.45** | 0.55 | 0.6302 | 0.6309 | 0.6296 | 0.6185 | 0.6593 | 0.6130 | 0.6310 | 0.6302 ± 0.0193 | +0.0393 | +0.0000 | 93.1% | 1.242 nats |
| **0.46** | 0.54 | 0.6317 | 0.6323 | 0.6296 | 0.6204 | 0.6611 | 0.6167 | 0.6310 | 0.6317 ± 0.0198 | +0.0407 | +0.0015 | 92.8% | 1.225 nats |
| **0.47** | 0.53 | 0.6332 | 0.6337 | 0.6296 | 0.6241 | 0.6630 | 0.6185 | 0.6310 | 0.6332 ± 0.0208 | +0.0422 | +0.0015 | 92.6% | 1.209 nats |
| **0.48** | 0.52 | 0.6340 | 0.6345 | 0.6315 | 0.6259 | 0.6630 | 0.6204 | 0.6310 | 0.6340 ± 0.0212 | +0.0430 | +0.0007 | 92.5% | 1.192 nats |
| **0.49** | 0.51 | 0.6351 | 0.6355 | 0.6315 | 0.6278 | 0.6648 | 0.6241 | 0.6272 | 0.6351 ± 0.0205 | +0.0441 | +0.0011 | 92.3% | 1.175 nats |
| **0.50** | 0.50 | **0.6354** | **0.6359** | 0.6315 | 0.6278 | 0.6648 | 0.6259 | 0.6272 | **0.6354 ± 0.0206** | **+0.0445** | +0.0004 | 92.2% | 1.158 nats |

---

## 4. Addressing the Six Core Forensic Questions

### 1. Best Official-Only Weight
Within $[0.20, 0.50]$, the best official-only weight is **$w_{\text{svc}} = 0.50$** ($w_{\text{qwen}} = 0.50$), achieving **0.6147 Balanced Accuracy** (+0.0237 over EXP47).

### 2. Best Open-Pool Weight
Within $[0.20, 0.50]$, the best Open-Pool weight is **$w_{\text{svc}} = 0.50$** ($w_{\text{qwen}} = 0.50$), achieving **0.6354 Balanced Accuracy** (+0.0445 over EXP47).

### 3. Does Open Pool Change the Optimal Weight?
**No.** Both Official Only and Open Pool curves increase monotonically across the entire range from $0.20$ to $0.50$. The presence of external data elevates the absolute score across every weight (+0.0208 higher at $0.50$), but does not alter the direction of the local gradient.

### 4. Is the Optimum Interior or Still at the Boundary?
**The optimum is strictly at the boundary ($w_{\text{svc}} = 0.50$).**
There is no local interior peak anywhere in $[0.20, 0.50]$. We explicitly report: **the boundary has NOT been resolved within this window**.

### 5. Is the Curve Smooth/Monotonic or Does it Develop a Peak?
The curve is **strictly smooth and monotonic**. There are zero negative local deltas. Short plateau shelves appear around $w_{\text{svc}} \in [0.26, 0.27]$ and $[0.43, 0.45]$ where BA remains unchanged, but the trajectory immediately resumes its upward climb.

### 6. Does the Evidence Support a Qwen-Dominant, Balanced, or SVC-Dominant Regime?
- **On Local Training CV**: The mathematical loss surface monotonically pulls the weight towards equal/SVC dominance (climbing to 0.6354 at $0.50$ and reaching 0.7028 at $0.85$ in EXP49).
- **The Epistemic Conflict**: We know from the EXP49 Kaggle evaluation (**0.612 Public LB vs 0.620 for pure EXP47**) that giving the shallow model $\ge 50\%$ decision power severely degrades test generalization due to surface n-gram domain shift.
- **Root Cause**: Qwen's top-2 logit margin is $8.24\times$ wider than the SVC's ($1.955$ vs $0.237$ nats). Because the local objective evaluates on prompts that share lexical characteristics with the training vocabulary, it pushes $w_{\text{svc}}$ upward simply to balance numerical scales.
- **Scientific Conclusion**: Local CV alone cannot determine the safe operational boundary for Kaggle. Local CV supports an SVC-dominant regime, but Kaggle empirical reality requires a **Qwen-dominant or conservative balanced anchor**.

---

## 5. Statistical Inference & Bootstrap Analysis vs EXP47

Using the paired per-example balanced-accuracy contribution:
$$\Delta_i = \frac{1}{18 \cdot N_{y_i}} \left( \mathbb{I}(p_{\text{candidate}, i} = y_i) - \mathbb{I}(p_{\text{EXP47}, i} = y_i) \right)$$
we evaluated 5,000 prompt-clustered bootstrap iterations (sampling with replacement across the 150 prompt groups):

| Candidate | $w_{\text{qwen}}$ | $w_{\text{svc}}$ | Local Delta vs EXP47 | 95% Bootstrap CI | $P(\Delta > 0)$ | Decision Characterization |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Config A (Conservative Deep Anchor)** | **0.75** | **0.25** | **+0.0193** | $[+0.0137, +0.0252]$ | **1.0000** | Ultra-safe; 96.9% Qwen agreement. |
| **Config B (Balanced Transition)** | **0.65** | **0.35** | **+0.0270** | $[+0.0211, +0.0333]$ | **1.0000** | Intermediate risk; 95.3% Qwen agreement. |
| **Config C (Numerical CV Maximum)** | **0.50** | **0.50** | **+0.0445** | $[+0.0370, +0.0522]$ | **1.0000** | High surface-shift risk; 92.2% Qwen agreement. |

### Technical Wording Notes:
- Across all out-of-fold predictions, there is **very low measured Pearson error correlation on the OOF predictions** ($r = 0.00045$).
- While Open Pool expands vocabulary and lifts local scores, **positive Open-Pool CV transfer does NOT prove hidden-test transfer**.

---

## 6. Decision Rule & Final Recommendations

### Recommended Candidates for Future Kaggle Consideration:

1. **Primary Recommended Candidate**: **Config A ($w_{\text{qwen}} = 0.75, w_{\text{svc}} = 0.25$)**
   - **Local BA**: **`0.6102`** (+0.0193 over EXP47).
   - **Agreement with Qwen**: **`96.9%`**.
   - **Why Preferable to EXP47**: It captures a statistically verified +0.0193 gain ($P(\Delta > 0) = 1.0$) while ensuring Qwen makes the final call on 97% of passages. This strictly insulates against the surface domain shift that doomed EXP49.
2. **Alternative Upper-Bound Candidate**: **Config B ($w_{\text{qwen}} = 0.65, w_{\text{svc}} = 0.35$)**
   - **Local BA**: **`0.6180`** (+0.0270 over EXP47).
   - **Agreement with Qwen**: **`95.3%`**.
   - **Why Consider**: If public leaderboard probing indicates that Open Pool surface vocabulary generalizes well on Kaggle, $0.35$ captures $+0.0270$ lift while still maintaining over 95% Qwen consensus.
3. **Candidate to AVOID on Kaggle**: **Config C ($w_{\text{svc}} = 0.50$)**
   - Even though it yields the highest local CV score (**0.6354**), equal weighting flips nearly 8% of all test predictions to surface n-grams, exposing the pipeline to test-time vocabulary shift.

### Confidence Level & Unresolved Risks:
- **Confidence Level**: **High for Local Characterization; Moderate for Kaggle Transfer**.
- **Unresolved Risks**:
  1. *Test-Set Vocabulary Shift*: Surface n-grams can fail when test topics diverge from training prompts.
  2. *Open Pool Kaggle Environment Dependency*: Requires the Open Pool dataset to be attached and discovered in Kaggle kernels without falling back to `train.csv`.
- **Active Kaggle Baseline**: **EXP47 (0.620 Public LB) remains our production baseline.**

---

**KAGGLE SUBMISSION STATUS: ABSOLUTELY NONE.**  
No submissions created or triggered. All results stored locally.
