# EXP61: Hidden-Surface / Representation-Shift Forensics

## Overview
Comprehensive empirical investigation into the 0.1553 Balanced Accuracy gap between EXP52 local OOF (0.7873) and Kaggle Public LB (0.6320).

## Key Discoveries
1. **Distance-to-Train Manifold Does Not Drive Error**:
   Spearman correlation between sample distance to the training manifold and prediction error is **rho = +0.0071 (p = 0.712)**. Stratification across 5 distance quintiles shows virtually flat accuracy (0.7800 to 0.7927 BA).
2. **Matched Local Surface Falsifies Sparsity as the Gap Driver**:
   Constructing a synthetic local validation surface matching the Kaggle test prompt density (R=2) yields **0.7786 +- 0.0176 BA**. This proves that prompt sparsity explains at most ~0.009 BA, leaving >93% of the gap unexplained.
3. **The True Root Cause: Ensemble Fragility on Disagreement**:
   - When Q8 and Q14 agree (40.3% of data), EXP52 achieves **0.9080 BA**.
   - When Q8 and Q14 disagree (59.7% of data), EXP52 drops to **0.7061 BA**.
   - Disagreement strongly correlates with error (**rho = +0.3142, p = 6.7e-63**).
   - On local OOF, Q8 and Q14 have an artificially low error correlation (r=0.0214), boosting the ensemble by **+0.153 BA**. On the hidden test set, sequence-level ambiguities create correlated errors, reducing the ensemble gain to **+0.012 BA**.

## Artifacts
- `results.csv`: Text length distribution forensics and statistical effect sizes.
- `domain_classifier_results.csv`: Comprehensive 10-point domain shift scorecard.
- `representation_shift.csv`: Accuracy and agreement stratified by distance quintiles.
- `distance_error_results.csv`: Spearman correlations and bootstrap 95% CIs with error.
- `matched_surface_results.csv`: 5-seed validation on matched sparse surface.
- `prediction_shift.csv`: Class prior and predicted distribution comparisons.
- `config.json`: Structured experiment parameters and summary metrics.
- 10 Diagnostic Figures saved in `experiments/EXP61_hidden_surface_forensics/` and mirrored to `plots/`.
