# EXP64: Disagreement-Aware Confidence Routing for Single-Sample Attribution

## Research Objective
Test whether single-sample confidence, margin, entropy, and Q8/Q14 disagreement features can predict which expert is more likely to be correct, and whether dynamic single-sample routing outperforms the frozen EXP52 baseline (0.60 Q8 + 0.40 Q14).

## Key Results
- **EXP52 Baseline OOF BA**: 0.7873 (Macro F1: 0.7880)
- **Learned Router Nested OOF BA**: 0.7906 (Delta: +0.0033, Macro F1: 0.7908)
- **95% Bootstrap CI**: [-0.0027, +0.0094] (P(delta > 0) = 87.0%)
- **Calibrated Router Nested BA**: 0.7829 (Delta: -0.0045)
- **Random Router Negative Control**: 0.7855 (Delta: -0.0019)
- **Disagreement Subset (N=1,612, 59.7% of data)**:
  - EXP52 BA: 0.6871
  - Router BA: 0.6945 (Delta: +0.0074)
  - Oracle BA: 0.8425
  - Corrections: +40 | Degradations: -31 | Net: +9

## Verdict: YELLOW
The learned single-sample router produces a modest lift of **+0.0033 BA** (net +9 samples out of 2,699). The 95% bootstrap confidence interval spans zero ([-0.0027, +0.0094]). On unseen held-out prompts, overconfident errors cause degradations (31 samples) that offset most corrections (40 samples).

**Production Champion Decision**: Retain EXP52 as frozen champion at 0.632 Public LB / 0.7873 OOF BA.
