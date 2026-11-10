# EXP65: Llama-3.1-8B Diverse Foundation Backbone Attribution

- **Experiment ID**: `EXP65_llama_diverse`
- **Date**: 2026-11-10 22:15:58
- **Base Model**: `meta-llama/Llama-3.1-8B`
- **Status**: RED (No meaningful complementary improvement or degradation over EXP52. Retain EXP52.)
- **Frozen Champion**: EXP52 (0.60 Q8 + 0.40 Q14, OOF BA = 0.7873, Public LB = 0.632)
- **EXP52 Remains Frozen**: YES
- **Production Changed**: NO
- **Kaggle Submission Made**: NO

## Key Results
- **Qwen3-8B Standalone**: 0.5910 BA
- **Qwen3-14B Standalone**: 0.6346 BA
- **EXP52 Champion Baseline**: 0.7873 BA
- **Llama-3.1-8B Standalone**: 0.3901 BA
- **Nested EXP52 + Llama**: 0.7884 BA ($\Delta = +0.0011$, 95% CI [-0.0063, +0.0083], $P(\Delta > 0) = 62.2\%$)
- **Measured Error Correlation (EXP52 vs Llama)**: $r = 0.1206$
- **Unique Corrections**: 159 samples | **Degradations**: 1231 samples | **Net**: -1072 samples
- **Three-Model Oracle**: 0.8803 BA (Gain over Q8+Q14: +0.0348)
