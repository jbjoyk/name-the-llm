# EXP60: Sparse Test-Regime Calibration

## Overview
Evaluates the prompt-density hypothesis formulated in EXP59:
Does reducing prompt response density from R=18 to R=2 explain the 0.1553 gap between local OOF (0.7873) and Kaggle Public LB (0.6320)?

## Key Findings
1. **Plain EXP52 Model Resilience**: Plain EXP52 achieves 0.7829 BA under dense R=18 and 0.7786 BA (mean over 5 seeds) under sparse R=2. The plain backbone suffers virtually no density degradation.
2. **Centered Twin Attenuation**: The centered twin head provides +0.0044 BA under dense R=18, but drops to -0.0012 BA under sparse R=2. The Difference-in-Differences is +0.0056 BA.
3. **Hypothesis Evaluation**: The total drop from dense canonical EXP52 (0.7873) to sparse recovered EXP52 (0.7774) is **0.0099 BA**, which explains **only 6.37%** of the 0.1553 OOF-to-LB gap.
4. **Conclusion**: The EXP59 prompt-density hypothesis is **PARTIALLY SUPPORTED** as a minor secondary factor, but **DISPROVEN as the primary explanation**. The remaining ~0.1454 BA gap is driven by general hidden-test domain shift and non-zero inter-backbone error correlation.

## Artifacts
- `results.csv`: Aggregate metrics across dense and sparse regimes.
- `seed_results.csv`: Per-seed evaluation across seeds 42, 123, 2024, 3407, 7777.
- `cluster_diagnostics.csv`: Cluster counts, singleton fractions, and pairwise recovery F1.
- `density_curve.csv`: Performance and diagnostics across R in [2, 4, 6, 9, 18].
- `config.json`: Structured experiment configuration and summary statistics.
- Figures: `ba_by_regime_and_variant.png`, `density_effect_differences.png`, `density_scaling_curve.png`, `cluster_size_distribution.png`, `confidence_entropy_shifts.png`, `singleton_fraction_comparison.png`.
