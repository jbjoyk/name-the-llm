# EXP_45 Experiment Freeze Manifest

**Date**: November 2026  
**Auditor**: Antigravity Machine Learning Research Agent  
**Repository**: `/home/rj2826/name-the-llm`  
**Git Commit**: `230d7764ae5b0d9236db0d7436d7a2f161cd79f8`  

---

## 1. System & Software Environment

| Component | Version / Specification |
| :--- | :--- |
| **Python** | 3.14.7 |
| **PyTorch** | 2.15.0+cu132 |
| **CUDA** | 13.2 (Driver 595.99.02) |
| **Transformers** | 5.22.1 |
| **Sentence-Transformers** | 6.2.0 |
| **Scikit-Learn** | 1.9.1 |
| **SciPy** | 1.18.1 |
| **NumPy** | 2.5.4 |
| **Pandas** | 3.1.0 |
| **Hardware** | 2 $\times$ NVIDIA GeForce RTX 3060 12GB, AMD Ryzen 7 7700 8-Core, 32GB RAM |

---

## 2. Immutable Data & Artifact Checksums (SHA-256)

| Artifact Path | SHA-256 Checksum | Description |
| :--- | :--- | :--- |
| [`data/train.csv`](file:///home/rj2826/name-the-llm/data/train.csv) | `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707` | Official competition dataset ($N=2,699$, 150 prompt groups, 18 LLM classes) |
| [`outputs/oof_exp36_predictions.npy`](file:///home/rj2826/name-the-llm/outputs/oof_exp36_predictions.npy) | `475e3c4073246c896d22703b100ad2c6fe59c9fd07ee70f5f25b735d2fb8f333` | Canonical baseline probability matrix ($2,699 \times 18$, float32) |
| [`outputs/oof_fused_gaussian_hungarian.npy`](file:///home/rj2826/name-the-llm/outputs/oof_fused_gaussian_hungarian.npy) | `3a688057b13ab7ac8310547fabfa1af5e263ae9706bcb87409ae74bf6ee20acf` | Canonical EXP_36 discrete predictions ($N=2,699$, int64) |
| [`outputs/oof_exp40_gat_predictions.npy`](file:///home/rj2826/name-the-llm/outputs/oof_exp40_gat_predictions.npy) | `c4047e2d15b8de6b8de65fa4fce280cf91cd820faecee6403993150c804d92da` | EXP_40 MiniLM-GAT blind probability matrix ($2,699 \times 18$, float32) |
| [`outputs/minilm_layer3_mean.npy`](file:///home/rj2826/name-the-llm/outputs/minilm_layer3_mean.npy) | `ddc05a8777197866c2d0d19a2824838a196e417ec606ffd970aa85acf44edf6b` | Pretrained MiniLM-L3 embeddings ($2,699 \times 384$, float32) |
| [`outputs/features_reference_stylometry.npy`](file:///home/rj2826/name-the-llm/outputs/features_reference_stylometry.npy) | `f5940f4a0ca00ca737ed3f52a71963d44bc6b96e3d00553769a879499d0f6e14` | Reference stylometry feature matrix ($2,699 \times 144$, float32) |
| [`outputs/exp44_deberta_embeddings.npy`](file:///home/rj2826/name-the-llm/outputs/exp44_deberta_embeddings.npy) | `fd0bb0c09bd0bee44eb7a98784f85ee771680a43cad60c30c4409444bd2d9653` | Pretrained DeBERTa-v3-small embeddings ($2,699 \times 768$, float32) |
| [`outputs/oof_exp44_hybrid_gat_predictions.npy`](file:///home/rj2826/name-the-llm/outputs/oof_exp44_hybrid_gat_predictions.npy) | `d46b4ba10fbe94c4d36c4431d2af4b501b8a99b78dcd4f3b2b73e2f61823bb1f` | EXP_44-C Hybrid GAT blind probability matrix ($2,699 \times 18$, float32) |

All listed artifacts are frozen and protected from modification.
