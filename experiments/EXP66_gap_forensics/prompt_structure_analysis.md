# EXP66 Prompt Structure & Task Distribution Analysis

## 1. Executive Summary
- **Research Question**: Does the hidden-test response set occupy a shifted prompt or task distribution compared to the competition training set?
- **Analysis Scope**: Unsupervised semantic comparison of 150 training prompt clusters vs test passages using TF-IDF content manifolds and nearest-centroid projection.
- **Core Finding**: **No catastrophic domain shift or out-of-vocabulary collapse was observed**. Test passages have a mean cosine similarity of **0.1310** (range: 0.0806 - 0.1618) to the nearest training prompt centroid.
- **However**: The test prompts represent **unseen, disjoint semantic queries**, meaning models cannot rely on memorized prompt-specific topic vocabulary.

---

## 2. Nearest Training Prompt Centroid Distances
| Metric | Cosine Similarity to Nearest Train Prompt |
|---|:---:|
| **Mean** | 0.1310 |
| **Median** | 0.1505 |
| **Standard Deviation** | 0.0359 |
| **Minimum** | 0.0806 |
| **Maximum** | 0.1618 |
| **25th Percentile** | 0.1156 |
| **75th Percentile** | 0.1562 |

---

## 3. Response Length Conditioned on Prompt
- **Training Data**: Within a single prompt, response lengths have an average standard deviation of **371.8 characters** around the prompt mean of **755.3 characters**.
- **Test Data**: Test passages have a mean length of **719.3 characters** (std: 379.6), which falls well within the training prompt length distribution (mean Cohen's d = -0.104, negligible).

---

## 4. Semantic Cluster & Formatting Distribution
1. **Instruction Types**:
   - Both train and test contain informative expository passages, technical/scientific explanations (physics, geology, economics, computer science), historical analysis, and philosophical arguments.
   - Zero test passages contain conversational chatbot artifacts (e.g., "Sure, I can help with that!"), confirming identical system-prompt stripping across train and test.
2. **Formatting**:
   - Neither train nor test contains heavy markdown bullet dumps or conversational pleasantries. Both consist of structured academic/professional expository text passages.

---

## 5. Diagnostic Implication for Kaggle Gap
- The OOF-to-LB gap is **NOT** explained by prompt topic drift (e.g., test is not suddenly code or foreign language).
- The gap is driven by the **unseen prompt condition**: when prompts are unseen, any residual prompt-style memorization fails, and models must rely strictly on true attribution invariance.
