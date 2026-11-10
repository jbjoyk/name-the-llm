# EXP66: Kaggle / OOF Generalization Gap Forensics

## Overview
- **Experiment Purpose**: Forensic root-cause analysis of the 0.1553 Balanced Accuracy gap between the frozen production champion **EXP52** (local prompt-disjoint OOF BA: **0.7873**) and the Kaggle Public Leaderboard (**0.6320**).
- **Secondary Objective**: Diagnose the mechanism causing **EXP64** to score **0.058** on Kaggle.
- **Constraints Enforced**:
  - Zero model retraining.
  - Zero Kaggle submissions or uploads.
  - EXP52, EXP64, EXP65 remain 100% frozen and unmodified.

---

## Deliverables in this Directory

| File | Description |
|---|---|
| [`config.json`](config.json) | Machine-readable experiment configuration and model specifications. |
| [`pre_audit_hashes.json`](pre_audit_hashes.json) | Cryptographic SHA-256 signatures of all input artifacts and frozen baselines. |
| [`local_vs_kaggle_parity.csv`](local_vs_kaggle_parity.csv) | Component-by-component trace comparing local inference vs Kaggle kernel logic. |
| [`class_mapping_audit.json`](class_mapping_audit.json) | Verification of 18 classes in exact alphabetical order across all artifacts. |
| [`submission_audit.json`](submission_audit.json) | Audit of submission parquet/csv formats, columns, nulls, and row order preservation. |
| [`confidence_analysis.csv`](confidence_analysis.csv) | Per-sample OOF confidence, entropy, margin, agreement, and error classifications. |
| [`confidence_report.md`](confidence_report.md) | Calibration analysis across deciles and agreement subsets (ECE = 0.0253). |
| [`surface_shift.csv`](surface_shift.csv) | Statistical tests (Cohen's d, KS tests) comparing train and test surface distributions. |
| [`class_surface_shift.csv`](class_surface_shift.csv) | Model-specific length, vocabulary, and formatting characteristics on train data. |
| [`confusion_analysis.csv`](confusion_analysis.csv) | 18x18 confusion matrix analysis and top 10 error pairs broken down by backbone. |
| [`prompt_structure_analysis.md`](prompt_structure_analysis.md) | Unsupervised semantic distance between test passages and train prompt centroids. |
| [`open_pool_audit.md`](open_pool_audit.md) | Exact overlap, leakage, and provenance audit of the 212,623-sample Open Pool. |
| [`test_regime_analysis.md`](test_regime_analysis.md) | Quantitative analysis of prompt density ($R$) and centering collapse on sparse test data. |
| [`counterfactual_fusion.csv`](counterfactual_fusion.csv) | Diagnostic parametric weight sweeps between Q8 and Q14 on OOF data. |
| [`cause_ranking.csv`](cause_ranking.csv) | Ranked and categorized candidate causes with evidence levels and confidence. |
| [`EXP66_REPORT.md`](EXP66_REPORT.md) | Full comprehensive forensic investigation report. |
| [`plots/`](plots/) | 9 publication-grade forensic figures. |

---

## Generated Diagnostic Plots (`plots/`)
1. [`confidence_vs_error.png`](plots/confidence_vs_error.png): Empirical error rate vs mean confidence across 10 deciles.
2. [`entropy_vs_error.png`](plots/entropy_vs_error.png): Error rate vs predictive entropy.
3. [`margin_vs_error.png`](plots/margin_vs_error.png): Error rate vs top-1/top-2 margin.
4. [`train_test_length_distributions.png`](plots/train_test_length_distributions.png): Character length KDE comparison.
5. [`train_test_token_distributions.png`](plots/train_test_token_distributions.png): Subword/token count KDE comparison.
6. [`class_confusion_matrix.png`](plots/class_confusion_matrix.png): 18x18 normalized confusion matrix heatmap.
7. [`class_surface_statistics.png`](plots/class_surface_statistics.png): Mean character length per model class.
8. [`q8_q14_agreement_analysis.png`](plots/q8_q14_agreement_analysis.png): Accuracy and error rate under backbone agreement vs disagreement.
9. [`fusion_weight_diagnostic.png`](plots/fusion_weight_diagnostic.png): Parametric OOF balanced accuracy curve across fusion weights.
