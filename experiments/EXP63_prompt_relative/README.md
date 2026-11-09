# EXP63 — Prompt-Conditional Relative Attribution & Set Assignment

- **Baseline**: EXP52 Dual-LoRA (Qwen3-8B $0.60$ + Qwen3-14B $0.40$), OOF BA: **0.7873**
- **Plain Argmax vs Centered**:
  - EXP52 Independent: **0.7873**
  - Simple Centered LR: **0.7873**
  - Leave-One-Out (LOO) Centered LR: **0.7869**
  - Pairwise Difference + AbsDiff LR: **0.7654**
- **Hungarian 1-to-1 Matching Gain across Density**:
  - $R=18$ (Full Dense Training): **0.9040** (Delta = +0.1167)
  - $R=12$: **0.8431** (Delta = +0.0534)
  - $R=8$: **0.8121** (Delta = +0.0234)
  - $R=4$: **0.7957** (Delta = +0.0102)
  - $R=2$ (Realistic Sparse Test): **0.8021** (Delta = +0.0052)
- **Selection-Safe Nested CV Fusion**: **0.7995** (Delta = +0.0122)
  - 95% Bootstrap CI: [+0.0058, +0.0183], P(Delta > 0) = 100.0%
- **Verdict**: **YELLOW**
  - Strong improvement under dense R=18 Hungarian matching (+0.1167 BA), but completely collapses under sparse R=2 simulation (+0.0052 BA, and LOO relative delta is negative at -0.0085 BA). The gain exploits an artificial artifact of the dense R=18 training set and is not robust for test conditions.
- **Production Status**: EXP52 remains frozen at 0.632 LB / 0.7873 local OOF BA.
