# Artifact Provenance Registry: Kaggle "Name That LLM"

Permanent cryptographic record of core model checkpoints, heads, OOF arrays, datasets, and configurations used across experiments EXP46A through EXP58.

---

## 1. Cryptographically Verified Artifacts

| Artifact Key | Workspace File Path | SHA-256 Checksum | Purpose | Origin / Source | Used In |
|:---|:---|:---:|:---|:---|:---|
| **q8_adapter** | `models/adapters_v5/q8v5/q8v5/adapter_model.safetensors` | `a600e552f40bcda00194f147f957d3bbb04e176b6bccf4a1f7fe39b82c8e8f51` | Fine-tuned LoRA24 adapter for Qwen3-8B | `dhruvtibarewal/nlm-l24-adapters-v5` | EXP46A, EXP47, EXP52, EXP53, EXP54, EXP55, EXP56, EXP57A, EXP57B, EXP58 |
| **q8_heads** | `models/adapters_v5/q8v5/q8v5/heads.pt` | `31868cd0e9618fac506c93a08f118ee00202041d67426890e609d13e2cd33f0a` | Classifier readout heads for Qwen3-8B (`plain` + `centered`) | `dhruvtibarewal/nlm-l24-adapters-v5` | EXP46A, EXP47, EXP52, EXP53, EXP54, EXP55, EXP56, EXP57A, EXP57B, EXP58 |
| **q8_config** | `models/adapters_v5/q8v5/q8v5/adapter_config.json` | `621136c127ae11004b136582ac899229a191c2d57443b297f7867e88b42d9204` | LoRA configuration for Qwen3-8B | `dhruvtibarewal/nlm-l24-adapters-v5` | EXP46A, EXP47, EXP52, EXP53, EXP54, EXP55, EXP56, EXP57A, EXP57B, EXP58 |
| **q14_adapter** | `models/adapters_v5/q14v5/q14v5/adapter_model.safetensors` | `7580831f6d7184c9a676ce88c3b69af00e42ffed1e7132fd8812c9ba632616c4` | Fine-tuned LoRA24 adapter for Qwen3-14B | `dhruvtibarewal/nlm-l24-adapters-v5` | EXP51A, EXP52, EXP53, EXP54, EXP55, EXP56, EXP57A, EXP57B, EXP58 |
| **q14_heads** | `models/adapters_v5/q14v5/q14v5/heads.pt` | `cafdbade649017d666c2da623adb6539877e1125a996f2736133127ae6121b3d` | Classifier readout heads for Qwen3-14B (`plain` + `centered`) | `dhruvtibarewal/nlm-l24-adapters-v5` | EXP51A, EXP52, EXP53, EXP54, EXP55, EXP56, EXP57A, EXP57B, EXP58 |
| **q14_config** | `models/adapters_v5/q14v5/q14v5/adapter_config.json` | `1f2c044ca59f117a2865ce91478f126d25f974224282b70d320828e8b504173e` | LoRA configuration for Qwen3-14B | `dhruvtibarewal/nlm-l24-adapters-v5` | EXP51A, EXP52, EXP53, EXP54, EXP55, EXP56, EXP57A, EXP57B, EXP58 |
| **q8_exp47_oof** | `outputs/oof_exp47_qwen_centered_twin.npy` | `647a75f59789f2574bef7578ebf0e5f00eae2a8481724cfd6316bb4199ad22cb` | Prompt-disjoint 5-fold OOF logits for Qwen3-8B Centred Twin | Local EXP47 inference | EXP48, EXP49, EXP50, EXP50B, EXP52, EXP53, EXP54, EXP55, EXP56, EXP57A, EXP57B, EXP58 |
| **q14_q14v5_oof**| `outputs/oof_q14v5_antopil_lora.csv` | `58822e8d71d6b6ae1d77a3eb62a5f3a2d4f1757c239392d1b5d9e227646ab9e9` | Prompt-disjoint 5-fold OOF predictions for Qwen3-14B | Anto Pil / Dhruv Tibarewal (`q14v5`) | EXP51, EXP51A, EXP52, EXP53, EXP54, EXP55, EXP56, EXP57A, EXP57B, EXP58 |
| **train_csv** | `data/train.csv` | `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707` | Official training dataset (2,699 samples, 150 prompts, 18 LLMs) | Kaggle Competition | All Experiments |
| **open_pool_clean** | `data/external/open_pool_clean.parquet` | `0c5b289e9f5f7c3f439462d09b3d1d7ef4648839db20309e3476066f3982d5c9` | Deduplicated 212k external open pool corpus | Kaggle community open pool | EXP49, EXP50, EXP50B, EXP54 |
| **open_pool_36k** | `data/external/open_pool_36k.parquet` | `710c99f8f693ae3245a4c7242dcfff055dda6a28791a39923420e503b555654c` | Balanced 36k sample of open pool | EXP49 external pipeline | EXP49, EXP50, EXP50B |
| **oof_open_pool_svc** | `outputs/oof_open_pool_svc.npy` | `cd7e0833a1d6bc60048a3944ccb55018522a407ce37edf7ccd0ec288728ee5b6` | 5-Fold OOF prediction probabilities for 4-View Open-Pool LinearSVC | EXP28 / EXP49 pipeline | EXP49, EXP50, EXP50B, EXP54 |
| **oof_reference_svc** | `outputs/oof_reference_svc.npy` | `7f0aef0bafca28baac27ff328a4802a2ea027990d33c776b43fc4ff68ef3fda9` | 5-Fold OOF prediction probabilities for Official-Only LinearSVC | EXP27 pipeline | EXP50, EXP50B |
| **oof_stylometry_exp55** | `outputs/oof_stylometry_exp55.parquet` | `cad4df87ee0853148b3252aeac69d37e11a3b9b145f2751ccf6a76634f3b6fe2` | 5-Fold OOF prediction probabilities for character-level stylometry | EXP55 pipeline | EXP55 |
| **exp52_candidate_submission** | `submissions/candidates/EXP52/submission.csv` | `a289fe593b0c45378390f83cefa7ae39b818a9435e80cd175116aa572de7e4e0` | Submission candidate output generated from EXP52 kernel | EXP52 kernel test run | EXP52 |

---

## 2. Retention Policy

The raw large binary artifacts (adapters, heads, external parquets, numpy arrays) are tracked strictly locally and excluded from git commits via `.gitignore` to prevent repository bloat while preserving reproducible cryptographic proof of provenance.
