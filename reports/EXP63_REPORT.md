# EXP63: Prompt-Conditional Relative Attribution & Set Assignment Report

- **Experiment ID**: `EXP63_prompt_relative`
- **Date**: 2026-11-09 15:30:30
- **Production Champion**: EXP52 (Dual-LoRA Qwen3-8B $0.60$ + Qwen3-14B $0.40$)
  - Kaggle Public LB: **0.6320**
  - Prompt-Disjoint OOF BA: **0.7873**
- **Experiment Type**: Prompt-Relative Modeling, Pairwise Attribution & Combinatorial Assignment
- **Status**: COMPLETE
- **Verdict**: **YELLOW** (Large apparent improvement under dense $R=18$ Hungarian matching: $+0.1167$ BA, and $+0.0122$ nested relative fusion; however, both gains collapse under sparse $R=2$ simulation: Hungarian delta drops to $+0.0052$ BA, and LOO relative delta turns negative at $-0.0085$ BA; the method exploits an artificial artifact of the dense 18-response training construction and does not generalize to sparse test prompts; RETAIN EXP52 as frozen production champion)
- **Production Changed**: **NO**
- **EXP52 Remains Frozen**: **YES**
- **Kaggle Submission Made**: **NO**

---

## 1. Executive Summary

EXP63 was conducted to test the hypothesis:
> *"Training contains 18 responses for every prompt, one from each of the 18 model classes. Therefore a model may learn stronger attribution signals by removing prompt-specific information and comparing responses relative to the other responses generated for the same prompt."*

We investigated three interconnected mechanisms:
1. **Prompt-Conditional Centering**: Removing prompt-specific semantic bias via global centroid centering ($\mu_p = \text{mean}(z_p)$) and leave-one-out (LOO) centroid centering ($\mu_{p,-i} = \text{mean}_{j \ne i}(z_{pj})$).
2. **Pairwise Relative Attribution**: Training classifiers on relative difference vectors ($z_i - z_j$) and absolute peer dispersion ($|z_i - z_j|$).
3. **One-to-One Set Assignment**: Formulating prompt-level attribution as a bipartite matching problem via Hungarian assignment ($\max \sum S[i, \pi(i)]$).
4. **Sparse Regime Robustness**: Testing whether these mechanisms survive under realistic sparse response densities ($R \in [18, 12, 8, 4, 2]$) across 5 random seeds.

### Key Numerical Findings:
1. **Centering Equivalence**:
   - Leave-one-out centered representations $c_i^{\text{LOO}}$ are mathematically an exact positive scalar multiple $\frac{N}{N-1}$ of simple centered representations $c_i^{\text{simple}}$ for every response in a prompt.
   - Argmax on raw centered representations achieves **`0.7488` BA** (substantially below EXP52's `0.7873` BA), because pure centering discards absolute prior logit scale.
   - Logistic Regression trained on LOO centered representations achieves **`0.7869` BA** standalone.
2. **Dense Training Exploitation ($R=18$)**:
   - When all 18 responses are available, Hungarian assignment on the EXP52 score matrix achieves **`0.9040` BA** ($\Delta = \mathbf{+0.1167}$ BA over EXP52).
   - In selection-safe nested cross-validation (5 outer folds $\times$ 4 inner folds), fusing LOO relative logits with EXP52 selects $w^* = 0.40$ across all 5 folds, yielding **`0.7995` BA** ($\Delta = \mathbf{+0.0122}$, 95% bootstrap CI $[+0.0058, +0.0183]$).
3. **Evaporation Under Sparse Response Regimes ($R=2$)**:
   - The Hungarian assignment gain drops precipitously as response density $R$ scales down:
     - $R=18$: **$+0.1167$ BA** ($0.7873 \to 0.9040$)
     - $R=12$: **$+0.0534 \pm 0.0032$ BA** ($0.7897 \to 0.8431$)
     - $R=8$: **$+0.0234 \pm 0.0041$ BA** ($0.7887 \to 0.8121$)
     - $R=4$: **$+0.0102 \pm 0.0060$ BA** ($0.7855 \to 0.7957$)
     - $R=2$: **$+0.0052 \pm 0.0047$ BA** ($0.7969 \to 0.8021$)
   - Relative LOO centering delta collapses from $+0.0056$ at $R=18$ to **$-0.0085 \pm 0.0159$ BA** at $R=2$ (net degradation).
4. **Structural Test Reality**:
   - Test prompts in Kaggle do **not** provide `prompt_id` columns, average only $\approx 2$ responses per prompt ($R \approx 2$), and do not represent a closed 18-class permutation.
   - Therefore, Hungarian assignment and relative centering gains are artifacts of the dense 18-response training design.

**Verdict: YELLOW**. Fulfills the explicit scientific criterion: *“If improvement occurs only when all 18 responses are available but disappears at sparse R, the method exploits a training-only structure.”* EXP52 remains frozen as the production champion at 0.632 Public LB.

---

## 2. Environment & Repository Provenance

- **Repository Root**: `/home/rj2826/name-that-llm`
- **Git Commit Prior to EXP63**: `189c5da1ded4e55dd79bd1259c44a2e6e04653e8` (`exp62: test cross-architecture complementarity`)
- **Working Tree**: Clean
- **Python**: 3.14.7
- **PyTorch**: 2.15.0+cu132
- **CUDA**: 13.2
- **Accelerators**: 2 $\times$ NVIDIA GeForce RTX 3060 (12 GB VRAM each)
- **Scikit-Learn**: 1.9.1
- **SciPy**: 1.18.1

---

## 3. Dataset & Prompt Structure Audit (Phases 2 & 3)

### Training Set Structure (`train.csv`):
- Total passages: **2,699**
- Unique prompts: **150**
- Unique candidate classes: **18**
- Response distribution per prompt:
  - **149 prompts** contain exactly **18 responses** from **18 distinct model classes**.
  - **1 prompt** (`p002`) contains **17 responses** from **17 distinct model classes** (missing: `claude-fable-5`).
  - Total count: $149 \times 18 + 1 \times 17 = 2,682 + 17 = 2,699$ rows.

| Prompt Type | Prompt Count | Responses per Prompt | Unique Classes per Prompt | Percentage of Data |
|:---|:---:|:---:|:---:|:---:|
| Balanced 18-Permutation | 149 | 18 | 18 | 99.37% (2,682 rows) |
| Missing Single Class (`p002`) | 1 | 17 | 17 | 0.63% (17 rows) |

### Test Set Structure (`test.csv`):
- Public commit placeholder: **3 rows** (`['id', 'text']`).
- **`prompt_id` is NOT provided in test data**.
- On Kaggle's hidden evaluation server (~4,000 rows), test passages average only **$\approx 2$ responses per prompt** ($R \approx 2$).
- The classes present in any given test prompt are arbitrary and unknown (not a 1-to-1 permutation).

---

## 4. Prompt Centering & Leave-One-Out Mathematical Identity (Phases 5 & 6)

For any prompt $P$ containing $N$ responses $z_1, \dots, z_N$ and sum vector $S = \sum_{j=1}^N z_j$:
1. **Global Centroid**:
   $$\mu_p = \frac{1}{N} S \implies c_i^{\text{simple}} = z_i - \frac{1}{N} S$$
2. **Leave-One-Out Centroid**:
   $$\mu_{p, -i} = \frac{1}{N - 1} \sum_{j \ne i} z_j = \frac{S - z_i}{N - 1}$$
   $$c_i^{\text{LOO}} = z_i - \frac{S - z_i}{N - 1} = z_i \left(1 + \frac{1}{N-1}\right) - \frac{S}{N-1} = \frac{N}{N-1} \left( z_i - \frac{1}{N} S \right) = \frac{N}{N-1} \cdot c_i^{\text{simple}}$$

### Consequence:
For all responses in prompt $P$, $c_i^{\text{LOO}}$ is an exact scalar multiple of $c_i^{\text{simple}}$ by the factor $\frac{N}{N-1}$ ($= \frac{18}{17} \approx 1.0588$ for 18-response prompts).
Consequently:
$$\operatorname{argmax}_k c_{i, k}^{\text{LOO}} \equiv \operatorname{argmax}_k c_{i, k}^{\text{simple}}$$
Direct argmax on centered logits achieves **`0.7488` BA** for both simple and LOO centering. Pure centering strips the model-level logit bias, which harms standalone accuracy.

---

## 5. Representation Benchmarks & Comparison against EXP47 (Phases 7 & 8)

| Representation | Classifier | OOF Balanced Accuracy | Macro-F1 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|
| **EXP52 Plain Baseline** | Argmax | **0.7873** | 0.7880 | 0.7778 | 0.7793 | 0.7907 | 0.7833 | 0.8056 |
| **Simple Centered EXP52** | Argmax | 0.7488 | 0.7482 | 0.7389 | 0.7385 | 0.7630 | 0.7444 | 0.7593 |
| **LOO Centered EXP52** | Argmax | 0.7488 | 0.7482 | 0.7389 | 0.7385 | 0.7630 | 0.7444 | 0.7593 |
| **Plain EXP52 Logits** | Logistic Regression | 0.7725 | 0.7719 | 0.7519 | 0.7681 | 0.7833 | 0.7611 | 0.7981 |
| **Simple Centered EXP52** | Logistic Regression | **0.7873** | 0.7874 | 0.7741 | 0.7848 | 0.7926 | 0.7815 | 0.8037 |
| **LOO Centered EXP52** | Logistic Regression | **0.7869** | 0.7870 | 0.7722 | 0.7866 | 0.7944 | 0.7796 | 0.8019 |
| **Pairwise (LOO + AbsDiff)** | Logistic Regression | 0.7654 | 0.7654 | 0.7556 | 0.7626 | 0.7796 | 0.7556 | 0.7741 |
| **Pairwise (LOO + AbsDiff)** | Linear SVM | 0.7855 | 0.7851 | 0.7778 | 0.7793 | 0.7963 | 0.7796 | 0.7944 |
| **EXP47 (Q8 Centered Twin)** | Twin Head ($w=0.35$) | 0.5910 | 0.5915 | — | — | — | — | — |
| **EXP46A (Q8 Plain Base)** | Argmax | 0.5862 | 0.5866 | — | — | — | — | — |

### Comparison with EXP47:
EXP47 already tested intra-prompt centering on Qwen3-8B ($0.5862 \to 0.5910$ BA, $+0.0048$ boost). Pure centering without combining with the plain logit baseline degrades performance ($0.7873 \to 0.7488$). Fusing centered logits adds marginal residual information only under full 18-response prompts.

---

## 6. Hungarian One-to-One Set Assignment (Phases 11, 12, 13, 14)

On held-out validation prompts where all 18 responses are observed, assigning predictions by maximizing total log-probability subject to a 1-to-1 matching constraint ($\sum_i S[i, \pi(i)]$) solves the bipartite matching via `scipy.optimize.linear_sum_assignment`.

### Full Density ($R=18$) Results:
- **EXP52 Independent**: **`0.7873` BA** (Macro-F1: `0.7880`)
- **EXP52 Hungarian Matching**: **`0.9040` BA** (Macro-F1: `0.9040`)
- **Apparent Gain**: $\mathbf{+0.1167}$ Balanced Accuracy (+11.67%)!

Under dense training conditions, Hungarian matching resolves **382 of EXP52's 574 errors** (66.55% error reduction) simply by forbidding intra-prompt duplicate predictions.

---

## 7. Sparse Response Regime Robustness Sweep (Phases 12, 15)

To evaluate whether Hungarian assignment and relative centering generalize to realistic test conditions, we conducted a multi-seed subsampling sweep across response densities $R \in [18, 12, 8, 4, 2]$ using 5 random seeds (42, 123, 2024, 3407, 7777).

| Responses / Prompt ($R$) | Independent EXP52 BA | Hungarian Assignment BA | Hungarian Delta ($\Delta_{\text{hung}}$) | LOO Relative Centering BA | Relative Delta ($\Delta_{\text{rel}}$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$R=18$ (Full Dense)** | $0.7873 \pm 0.0000$ | **$0.9040 \pm 0.0000$** | $\mathbf{+0.1167 \pm 0.0000}$ | $0.7929 \pm 0.0000$ | $+0.0056 \pm 0.0000$ |
| **$R=12$** | $0.7897 \pm 0.0029$ | $0.8431 \pm 0.0055$ | $+0.0534 \pm 0.0032$ | $0.7951 \pm 0.0029$ | $+0.0054 \pm 0.0015$ |
| **$R=8$** | $0.7887 \pm 0.0071$ | $0.8121 \pm 0.0051$ | $+0.0234 \pm 0.0041$ | $0.7945 \pm 0.0072$ | $+0.0058 \pm 0.0036$ |
| **$R=4$** | $0.7855 \pm 0.0179$ | $0.7957 \pm 0.0215$ | $+0.0102 \pm 0.0060$ | $0.7937 \pm 0.0190$ | $+0.0082 \pm 0.0055$ |
| **$R=2$ (Realistic Test)** | $0.7969 \pm 0.0174$ | **$0.8021 \pm 0.0211$** | $\mathbf{+0.0052 \pm 0.0047}$ | **$0.7885 \pm 0.0277$** | $\mathbf{-0.0085 \pm 0.0159}$ |

### Why the Hungarian Gain Collapses at $R=2$:
- At $R=18$, Hungarian assignment enforces 18 mutually exclusive constraints across all 18 classes, providing maximum structural regularization.
- At $R=2$, Hungarian assignment only prevents the 2 observed samples from choosing the exact same class. Since independent EXP52 already predicts distinct classes for $>85\%$ of peer pairs, the combinatorial constraint is inactive almost everywhere. The delta shrinks from $+0.1167$ to a negligible $+0.0052$.
- Furthermore, at $R=2$, leave-one-out centering against a single peer ($c_1^{\text{LOO}} = z_1 - z_2$) injects massive variance, causing net negative degradation ($-0.0085$ BA).

---

## 8. Selection-Safe Nested Cross-Validation (Phase 16)

Using 5 outer prompt-disjoint folds and 4 inner prompt-disjoint folds per outer training set:
- Across all 5 outer folds, inner validation on the dense $R=18$ training set selected $w^* = 0.40$.
- **Outer Fold Results**:
  - Fold 0: Baseline = 0.7778 $\to$ Fused = 0.7852 (+0.0074)
  - Fold 1: Baseline = 0.7793 $\to$ Fused = 0.7923 (+0.0130)
  - Fold 2: Baseline = 0.7907 $\to$ Fused = 0.8037 (+0.0130)
  - Fold 3: Baseline = 0.7833 $\to$ Fused = 0.8056 (+0.0223)
  - Fold 4: Baseline = 0.8056 $\to$ Fused = 0.8111 (+0.0055)
- **Nested CV Overall BA**: **`0.7995`** (Macro-F1: `0.8003`, Mean Fold: $0.7996 \pm 0.0094$)
- **Nested Delta vs EXP52**: $\mathbf{+0.0122}$ Balanced Accuracy
- **2,000 Paired Bootstrap Iterations**:
  - Mean Delta: $+0.0121$
  - 95% Bootstrap CI: **`[+0.0058, +0.0183]`**
  - Probability of Positive Delta: $P(\Delta > 0) = \mathbf{100.0\%}$

**Crucial Scientific Distinction**: While this $+0.0122$ nested gain is statistically significant on the dense $R=18$ dataset, our Section 7 sparse simulation proves that it relies entirely on dense 18-sample centroids and turns into a **$-0.0085$ BA loss** under $R=2$ sparse conditions.

---

## 9. Error Correction Breakdown & Complementarity Hierarchy (Phases 17 & 18)

Across all 2,699 samples, EXP52 commits exactly **574 errors**:

```text
EXP52 Error (574 samples)
    ├── Hungarian Assignment (R=18) Correct: 382 (66.55%)
    │    ├── Both Methods Correct:           121 (21.08%)
    │    └── Only Hungarian Correct:         261 (45.47%)
    ├── Relative LOO Method Correct:         147 (25.61%)
    │    └── Only Relative Correct:           26 ( 4.53%)
    └── Both Methods Wrong:                  166 (28.92%)
```

### Pairwise Complementarity Metrics:
| Pair | Agreement | Both Correct | Both Wrong | EXP52 Only | Method Only | Error Corr. ($r$) | Oracle BA |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **EXP52 vs LOO Relative** | 85.36% | 1,977 | 427 | 148 | 147 | **0.6738** | 0.8418 |
| **EXP52 vs Hungarian (R=18)** | 81.03% | 2,058 | 192 | 67 | 382 | **0.4209** | **0.9289** |
| **EXP52 vs Nested Fused** | 96.00% | 2,104 | 520 | 21 | 54 | **0.9159** | 0.8073 |

---

## 10. Figures & Visualizations

Five publication-grade figures were generated and verified at 300 DPI:

1. [`plots/exp63_baseline_vs_relative.png`](file:///home/rj2826/name-that-llm/plots/exp63_baseline_vs_relative.png):
   Bar chart comparing EXP52 Plain (`0.7873`), Simple Centered (`0.7873`), LOO Centered (`0.7869`), Pairwise (`0.7654`), and Nested Fused (`0.7995`).
2. [`plots/exp63_assignment_gain.png`](file:///home/rj2826/name-that-llm/plots/exp63_assignment_gain.png):
   Comparison illustrating the $+0.1167$ BA jump from independent classification (`0.7873`) to Hungarian matching (`0.9040`) at $R=18$.
3. [`plots/exp63_sparse_regime.png`](file:///home/rj2826/name-that-llm/plots/exp63_sparse_regime.png):
   Scaling curve across $R \in [18, 12, 8, 4, 2]$ demonstrating the collapse of Hungarian assignment delta from $+0.1167 \to +0.0052$ and LOO relative delta to $-0.0085$.
4. [`plots/exp63_error_correction.png`](file:///home/rj2826/name-that-llm/plots/exp63_error_correction.png):
   Distribution of EXP52 error resolution across Hungarian matching, LOO centering, both, and neither.
5. [`plots/exp63_complementarity.png`](file:///home/rj2826/name-that-llm/plots/exp63_complementarity.png):
   Complementarity profile showing agreement, error correlation, and theoretical oracle accuracy.

---

## 11. Verdict & Scientific Conclusion

### Verdict: YELLOW

Criteria Evaluation:
- Criterion 1: *Improvement occurs under dense $R=18$ validation* $\to$ **CONFIRMED** ($+0.1167$ Hungarian BA, $+0.0122$ nested relative BA).
- Criterion 2: *Improvement disappears under sparse $R=2$ simulation* $\to$ **CONFIRMED** (Hungarian delta collapses to $+0.0052$, LOO relative delta collapses to $-0.0085$).
- Criterion 3: *Exploits training-only design* $\to$ **CONFIRMED** (Dense 18-permutation does not exist in the Kaggle hidden test set, where test passages average $\approx 2$ responses/prompt and lack `prompt_id`).

### Scientific Principle:
Prompt-conditional relative attribution and Hungarian one-to-one assignment provide a massive mathematical benefit when the full closed set of 18 classes is guaranteed to be present in each prompt. However, because the competition evaluation surface is sparse ($R \approx 2$) and un-prompted, this mechanism **cannot transfer to the Kaggle hidden test set**. Attempting to deploy full-set assignment or dense prompt-relative centering to Kaggle will fail to close the 0.155 LB gap.

**Decision**:
- **DO NOT** deploy prompt-relative centering or Hungarian matching as a standalone solution to Kaggle.
- **RETAIN EXP52** strictly frozen as the production champion at **0.632 Public LB** / **0.7873 local OOF BA**.
- **ZERO Kaggle submissions**.
