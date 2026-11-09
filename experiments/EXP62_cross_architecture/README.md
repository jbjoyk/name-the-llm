# EXP62 — Cross-Architecture Complementarity Test

- **Target Architecture**: `answerdotai/ModernBERT-base` (Bidirectional Encoder, 149M params, 768d, RoPE, GeGLU)
- **Baseline**: EXP52 Dual-LoRA (Qwen3-8B $0.60$ + Qwen3-14B $0.40$), OOF BA: **0.7873**
- **ModernBERT Standalone OOF BA**: **0.3093** (LR) / **0.2371** (SVM)
- **EXP52 + ModernBERT Nested CV BA**: **0.7858** (Delta: **-0.0015**)
- **95% Bootstrap CI**: [-0.0039, +0.0006], P(Delta > 0) = 9.4%
- **Error Correlation (EXP52 vs ModernBERT)**: 0.1010
- **Oracle BA (EXP52, ModernBERT)**: 0.8340
- **Disagreement Subset (Q8 != Q14, 59.7% of data)**:
  - EXP52 BA: 0.6871
  - ModernBERT BA: 0.2836
  - ModernBERT Fixes: 117 | ModernBERT Worsens: 779 | Net: -662
- **Verdict**: **RED**
- **Production Decision**: EXP52 remains strictly frozen at 0.632 Public LB / 0.7873 local OOF BA. Stop cross-architecture model stacking with uncalibrated encoders.
