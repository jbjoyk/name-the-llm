# Provenance Audit: ATLAS Source-Conditioned Likelihood Model
**Document**: `reports/atlas_training_provenance.md`  
**Date**: 2026-11-06  
**Auditor**: Antigravity Research Engineer  
**Audit Target**: `josephadamski91/ntl-atlas-qwen3-8b-v1` (attached in `the-usual-suspects.ipynb` and evaluated in `EXP_37_atlas`)

---

## 1. Provenance Inventory

| Provenance Attribute | Verified Metadata / Source Evidence |
| :--- | :--- |
| **1. Adapter / Checkpoint Name** | `josephadamski91/ntl-atlas-qwen3-8b-v1` (`atlas_adapter/adapters.safetensors`, 442.8 MB) |
| **2. Base Model** | `Qwen/Qwen3-8B-Base` (Revision: `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`) |
| **3. Adapter Model Architecture** | `ntl_atlas.model.AtlasModel` (18 source-specific `SourceLinear` modules replacing attention projections `q_proj`, `k_proj`, `v_proj`, `o_proj`; rank 16, alpha 32, dropout 0.05, source prefix embedding control `[18, 1, 4096]`) |
| **4. Training Dataset** | Official competition dataset: `data/competition/train.csv` |
| **5. Training Dataset Path & Hash** | Verified SHA-256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707` (identical bit-for-bit to local `data/train.csv`) |
| **6. Number of Training Samples** | Exactly **2,699 rows** |
| **7. Label Usage During Training** | **YES**. All 18 candidate model labels were used as supervised conditioning source targets. |
| **8. Official Training Data Used** | **YES, 100%**. All 2,699 official training passages were trained on for 5 epochs. |
| **9. Official Prompts Used** | **YES, 100%**. All 150 official competition prompts were included in training. |
| **10. Official Responses Used** | **YES, 100%**. The next-token cross-entropy loss was minimized on all 2,699 official texts. |
| **11. Data from 0.66 Reference Notebook** | `the-usual-suspects.ipynb` explicitly imports this bundle via `/kaggle/input/datasets/josephadamski91/ntl-atlas-qwen3-8b-v1`. |
| **12. Training Date / Version** | Initial release: 2026-10-17; Version 3 release: 2026-10-20. 5 epochs, 110 optimizer steps, lr $10^{-4}$, effective batch size 128. |
| **13. Original Code & Config** | Bundled in `atlas_bundle.json`, `checkpoint.json`, `training-source-manifest.json`, and `ntl_atlas/model.py`. |

---

## 2. Cryptographic Verification of Training Data Identity

Verification command:
```bash
sha256sum /home/rj2826/name-the-llm/data/train.csv
# Output: 40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707
```
`training-source-manifest.json`:
```json
{
  "files": {
    "data/competition/train.csv": "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
  }
}
```
`atlas_bundle.json`:
```json
{
  "data_sha256": "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707",
  "data_policy": "Exact official 2699-row training file only; all 18 exact model labels; no external passages",
  "evaluation_policy": "In-sample diagnostics establish training health only; public leaderboard vs frozen 0.549 v3 baseline establishes improvement"
}
```

---

## 3. Provenance Conclusion

1. The ATLAS adapter was explicitly trained on **100% of the official training data (`train.csv`)**.
2. It was designed as an inference-time test model for Kaggle submissions where test prompts are unseen.
3. Therefore, evaluating this pretrained adapter against `train.csv` in any local cross-validation split (even GroupKFold) constitutes **in-sample evaluation**, as every passage was seen during its 5 training epochs.
