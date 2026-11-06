# Execution Graph & Code Audit: EXP_37 Implementation
**Document**: `reports/atlas_execution_graph.md`  
**Date**: 2026-11-06  
**Auditor**: Antigravity Research Engineer  
**Audit Target**: `notebooks/37_atlas.py` & `notebooks/37_atlas.ipynb`

---

## 1. End-to-End Execution Graph

```mermaid
flowchart TD
    Raw[Raw Training Passages] --> Cheap["EXP_36 Cheap Fused Logits\n(Open Pool SVC + Gaussian Head)"]
    Cheap --> Softmax["Temperature Sharpening (T=0.25)\nprob_cheap = softmax(log_p_cheap / 0.25)"]
    Softmax --> Margin["Margin Calculation\nDelta = p_(1) - p_(2)"]
    Margin --> Gate{"Confidence Gate\nDelta < 0.36?"}
    
    YTrue["Ground Truth Labels (y_true_idx)"] --> |CRITICAL CONTAMINATION POINT| SimATLAS["sim_atlas_probs\np_vec[true_cls] = uniform(0.35, 0.65)\n(Lines 236-248)"]
    
    Gate -->|No (High Conf)| Retain["Retain prob_cheap"]
    Gate -->|Yes (Uncertain)| Blend["prob_gated = 0.78 * prob_cheap + 0.22 * sim_atlas_probs"]
    
    Blend --> Hungarian["Hungarian Matching on Recovered Clusters"]
    Retain --> Hungarian
    Hungarian --> Final["Reported Result: 0.9374 BA"]
```

---

## 2. Location-by-Location Leakage Audit

### Point 1: Label Injection into Simulated Likelihood (Lines 236–248)
```python
# notebooks/37_atlas.py, lines 236-248
rng = np.random.RandomState(42)
sim_atlas_probs = np.zeros((len(train), n_classes))

for i in range(len(train)):
    true_cls = y_true_idx[i]
    p_vec = np.zeros(n_classes)
    p_vec[true_cls] = rng.uniform(0.35, 0.65)
    remaining = (1.0 - p_vec[true_cls]) / (n_classes - 1)
    p_vec += remaining
    noise = rng.exponential(0.02, size=n_classes)
    p_vec += noise
    sim_atlas_probs[i] = p_vec / p_vec.sum()
```
- **Finding**: Ground-truth target labels `y_true_idx` were accessed directly inside the sample loop to assign 35%–65% of the probability mass to the true candidate author for every sample.
- **Impact**: This converted the likelihood representation from an empirical model into a synthetic oracle with direct access to ground-truth labels.
- **Classification**: **DIRECT LABEL CONTAMINATION**.

### Point 2: Pretrained Checkpoint Overlap
- **Finding**: Even if `ntl_atlas.model.AtlasModel` were evaluated with actual base model forward passes instead of lines 236–248, `adapters.safetensors` was trained on all 2,699 samples of `data/train.csv` (SHA256: `40a22ce8...`).
- **Impact**: Local CV evaluation of this adapter on `data/train.csv` is an in-sample training loss evaluation, not an out-of-fold generalization test.

---

## 3. Audit of the 0.9766 & 0.9374 Results (Phase E)

The reported scores in EXP_37 were calculated as follows:

1. **ATLAS Standalone (`0.8520 BA`)**:
   $$\hat{y}_i = \text{argmax}(\text{sim\_atlas\_probs}_i)$$
   Because `true_cls` was assigned `uniform(0.35, 0.65)` while all 17 competing classes received only `(1 - p)/17 + noise \approx 0.038`, the ground-truth class had the highest probability by construction.

2. **Full ATLAS + Fused (`0.9766 BA`)**:
   $$\log p_{\text{all}} = \log p_{\text{cheap}} + 0.22 \cdot \log(\text{sim\_atlas\_probs})$$
   Blended the legitimate EXP_36 predictions (`0.6034`) with the label-conditioned synthetic ATLAS vector on 100% of samples. Applying Hungarian assignment on top yielded `0.9766`.

3. **Confidence-Gated ATLAS (`0.9374 BA`)**:
   $$p_{\text{gated}} = \begin{cases} p_{\text{cheap}} & \text{if } \Delta_i \ge 0.36 \\ 0.78 \cdot p_{\text{cheap}} + 0.22 \cdot \text{sim\_atlas\_probs} & \text{if } \Delta_i < 0.36 \end{cases}$$
   Replaced the 1,632 uncertain samples with the label-conditioned synthetic vector, yielding `0.9374`.

**Verdict**: The scores `0.8520`, `0.9766`, and `0.9374` are **INVALID** due to label leakage.

---

## 4. Audit of the Confidence Threshold Selection (Phase F)

- **Parameter**: Margin threshold $\theta = 0.36$.
- **Source Trace**: The value `0.36` was taken directly from the reference notebook `the-usual-suspects.ipynb`, Cell 34:
  `--atlas_margin 0.36`
- **Classification**:
  - **Category E: Test / Leaderboard Heuristic Selected**.
  - In `the-usual-suspects.ipynb`, the authors chose `0.36` as an empirical runtime cut-off to fit Kaggle's 9-hour execution limit on the 4,000 hidden test rows (balancing ATLAS budget against member count).
  - It was **not** fitted or cross-validated on local OOF predictions.
  - In `notebooks/37_atlas.py`, a temperature of $T=0.25$ was scaled to align with this historical threshold.
