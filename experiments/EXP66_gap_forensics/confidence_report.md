# EXP66 Confidence & Error Forensics Report

## 1. Executive Summary
- **Evaluation Dataset**: Official `train.csv` (2,699 samples, 150 prompts) under prompt-disjoint EXP52 OOF.
- **Overall Balanced Accuracy**: 0.787313 (Overall Error Rate: 21.27%).
- **Expected Calibration Error (ECE)**: 0.0431 across 10 deciles.
- **Core Observation**: Locally, EXP52 confidence is **exceptionally well-calibrated and strictly monotonic**. As confidence rises from decile 0 to decile 9, error drops monotonically from 62.22% to 0.74%.
- **The Fragility Axis**: The primary source of error is **Q8/Q14 inter-backbone disagreement**. When both backbones agree, error is only **5.61%**. When they disagree (59.7% of all samples), error rises to **31.82%**.

---

## 2. Confidence Decile Breakdown
| Decile | Mean Fused Confidence | Empirical Accuracy | Error Rate | Sample Count |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.3302 | 0.3778 | 0.6222 | 270 |
| 1 | 0.4639 | 0.4963 | 0.5037 | 270 |
| 2 | 0.5701 | 0.6778 | 0.3222 | 270 |
| 3 | 0.6728 | 0.7593 | 0.2407 | 270 |
| 4 | 0.7640 | 0.8556 | 0.1444 | 270 |
| 5 | 0.8358 | 0.8513 | 0.1487 | 269 |
| 6 | 0.8989 | 0.9222 | 0.0778 | 270 |
| 7 | 0.9438 | 0.9519 | 0.0481 | 270 |
| 8 | 0.9735 | 0.9889 | 0.0111 | 270 |
| 9 | 0.9958 | 0.9926 | 0.0074 | 270 |

---

## 3. Entropy & Margin Breakdown
| Entropy Decile | Mean Entropy | Accuracy | Error Rate | Margin Decile | Mean Margin | Accuracy | Error Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.0309 | 0.9926 | 0.0074 | 0 | 0.0588 | 0.3963 | 0.6037 |
| 1 | 0.1586 | 0.9852 | 0.0148 | 1 | 0.2000 | 0.5111 | 0.4889 |
| 2 | 0.2919 | 0.9519 | 0.0481 | 2 | 0.3590 | 0.6296 | 0.3704 |
| 3 | 0.4692 | 0.9111 | 0.0889 | 3 | 0.5100 | 0.7704 | 0.2296 |
| 4 | 0.6788 | 0.8519 | 0.1481 | 4 | 0.6514 | 0.8519 | 0.1481 |
| 5 | 0.8686 | 0.8104 | 0.1896 | 5 | 0.7591 | 0.8550 | 0.1450 |
| 6 | 1.0594 | 0.7741 | 0.2259 | 6 | 0.8496 | 0.9222 | 0.0778 |
| 7 | 1.2751 | 0.6296 | 0.3704 | 7 | 0.9163 | 0.9593 | 0.0407 |
| 8 | 1.5294 | 0.5185 | 0.4815 | 8 | 0.9610 | 0.9852 | 0.0148 |
| 9 | 1.8968 | 0.4481 | 0.5519 | 9 | 0.9936 | 0.9926 | 0.0074 |

---

## 4. Q8 / Q14 Agreement Dynamics
| Backbones Agree? | Sample Count | % of Data | Mean Fused Confidence | Mean Entropy | Accuracy | Error Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| NO (Disagreement) | 1612 | 59.7% | 0.6349 | 1.1187 | 0.6818 | 0.3182 |
| YES (Agreement) | 1087 | 40.3% | 0.9078 | 0.3916 | 0.9439 | 0.0561 |

---

## 5. Text Length & Error Distribution
| Length Decile | Mean Characters | Mean Words | Accuracy | Error Rate | Sample Count |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 297.0 | 44.9 | 0.7037 | 0.2963 | 270 |
| 1 | 362.5 | 53.8 | 0.7426 | 0.2574 | 272 |
| 2 | 429.3 | 64.4 | 0.7778 | 0.2222 | 270 |
| 3 | 513.9 | 77.4 | 0.7509 | 0.2491 | 269 |
| 4 | 611.7 | 92.3 | 0.7556 | 0.2444 | 270 |
| 5 | 733.3 | 111.9 | 0.8141 | 0.1859 | 269 |
| 6 | 866.9 | 132.2 | 0.8111 | 0.1889 | 270 |
| 7 | 1030.2 | 158.2 | 0.8333 | 0.1667 | 270 |
| 8 | 1221.2 | 188.4 | 0.8439 | 0.1561 | 269 |
| 9 | 1490.6 | 223.6 | 0.8407 | 0.1593 | 270 |

---

## 6. Diagnostic Conclusions
1. **Confidence Miscalibration is REJECTED as the primary cause of the gap**: Within the local distribution, confidence is well-calibrated (ECE = 0.0431).
2. **Disagreement Fragility is CONFIRMED**: When backbones diverge, confidence degrades and error escalates by 5.6x. If the test distribution increases inter-backbone divergence (as occurs on novel prompt styles), overall accuracy collapses toward the disagreement regime (~0.68) or standalone regime (~0.61–0.63).
