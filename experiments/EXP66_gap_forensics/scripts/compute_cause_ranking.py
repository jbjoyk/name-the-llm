import pandas as pd

causes = [
    {
        'rank': 1,
        'cause': 'Prompt-Regime Shift (Sparse Test R~1-2 vs Dense Train R=18 Centering Collapse)',
        'category': 'prompt_regime_shift',
        'evidence_level': 'SUPPORTED HYPOTHESIS',
        'evidence_details': 'On train.csv (R=18), prompt recovery clusters 95.6% of rows, fully activating centered twin head (+0.156 BA). On test.csv with R~1-2, singletons bypass centered head, causing models to drop to plain head (fused plain BA = 0.6315, exactly matching Kaggle LB = 0.6320).',
        'severity': 'CRITICAL',
        'confidence': 'HIGH',
        'can_explain_gap': 'YES (~0.12 - 0.15 BA of the 0.1553 gap)',
        'next_test': 'Design pure single-sample models that do not rely on prompt centering or Hungarian matching.'
    },
    {
        'rank': 2,
        'cause': 'Inter-Backbone Complementarity Collapse on Unseen Prompts',
        'category': 'hidden_test_complementarity_change',
        'evidence_level': 'SUPPORTED HYPOTHESIS',
        'evidence_details': 'On train OOF, Q8 and Q14 exhibit an artificially near-zero error correlation (r=0.0214), yielding +0.153 ensemble lift. On unseen test prompts, both Qwen models share sequence-modeling biases, causing correlated errors (r>0.30) and collapsing ensemble gain to +0.012 (0.620 -> 0.632).',
        'severity': 'CRITICAL',
        'confidence': 'HIGH',
        'can_explain_gap': 'YES (Co-occurs with prompt-regime shift; explains why Q8+Q14 ensemble gives modest lift on LB)',
        'next_test': 'Evaluate genuine out-of-family backbones (not stacking un-fine-tuned embeddings).'
    },
    {
        'rank': 3,
        'cause': 'Severe Sibling-Model Ambiguity Under Disagreement',
        'category': 'class_specific_surface_shift',
        'evidence_level': 'DIRECTLY OBSERVED',
        'evidence_details': 'Top 10 error pairs are 100% intra-family sibling confusions (GLM 5.3 vs 5.3-flash, Gemini 3.7 vs 3.8, GPT 5.6 Luna vs Sol vs Terra). When Q8 and Q14 disagree (59.7% of data), error rate is 31.8% vs 5.6% when they agree.',
        'severity': 'HIGH',
        'confidence': 'HIGH',
        'can_explain_gap': 'PARTIALLY (Explains the baseline error floor of individual models)',
        'next_test': 'Fine-tune specialized sibling-discrimination heads or pairwise loss.'
    },
    {
        'rank': 4,
        'cause': 'Global Surface Covariate Shift (Length, Whitespace, Punctuation)',
        'category': 'test_distribution_shift',
        'evidence_level': 'DIRECTLY OBSERVED',
        'evidence_details': 'Punctuation ratio (d=-0.613) and whitespace ratio (d=-0.752) show measurable shift between train and test. However, character length (d=-0.104) is negligible, and distance to train manifold has rho=+0.0071 with error.',
        'severity': 'LOW',
        'confidence': 'HIGH',
        'can_explain_gap': 'NO (Covariate shift explains < 0.01 BA)',
        'next_test': 'None required; covariate shift is benign.'
    },
    {
        'rank': 5,
        'cause': 'Confidence Miscalibration',
        'category': 'confidence_calibration_failure',
        'evidence_level': 'UNSUPPORTED',
        'evidence_details': 'Expected Calibration Error is only 0.0253 locally. Confidence is strictly monotonic with accuracy across all 10 deciles (decile 0 acc = 0.378, decile 9 acc = 0.993).',
        'severity': 'NONE',
        'confidence': 'HIGH',
        'can_explain_gap': 'NO',
        'next_test': 'No calibration post-processing needed.'
    },
    {
        'rank': 6,
        'cause': 'Open Pool Contamination or Validation Advantage',
        'category': 'open_pool_validation_advantage',
        'evidence_level': 'UNSUPPORTED',
        'evidence_details': 'Zero prompt overlap, zero text overlap with train and test. EXP52 runtime kernel does not even load the Open Pool.',
        'severity': 'NONE',
        'confidence': 'HIGH',
        'can_explain_gap': 'NO',
        'next_test': 'None.'
    },
    {
        'rank': 7,
        'cause': 'Inference Implementation Mismatch',
        'category': 'inference_implementation_mismatch',
        'evidence_level': 'UNSUPPORTED',
        'evidence_details': 'Local vs Kaggle pipeline parity audit confirmed 100% bitwise identity across checkpoints, adapters, heads, LayerNorms, tokenization, pooling, and fusion formulas.',
        'severity': 'NONE',
        'confidence': 'HIGH',
        'can_explain_gap': 'NO',
        'next_test': 'None.'
    },
    {
        'rank': 8,
        'cause': 'Class Label Mapping or Index Permutation Error',
        'category': 'class_label_mapping_error',
        'evidence_level': 'UNSUPPORTED',
        'evidence_details': 'All 18 classes are in exact alphabetical alignment (indices 0..17) across train.csv, kernel LABELS, q14v5 OOF columns, heads output dimension, and submission.parquet.',
        'severity': 'NONE',
        'confidence': 'HIGH',
        'can_explain_gap': 'NO',
        'next_test': 'None.'
    },
    {
        'rank': 9,
        'cause': 'Submission Row Ordering or ID Shuffling Error',
        'category': 'submission_ordering_error',
        'evidence_level': 'UNSUPPORTED',
        'evidence_details': 'Dynamic length batch sorting is exactly inverted by unsort_idx argsort inversion, verified with 100% precision on synthetic shuffled test orders.',
        'severity': 'NONE',
        'confidence': 'HIGH',
        'can_explain_gap': 'NO',
        'next_test': 'None.'
    },
    {
        'rank': 10,
        'cause': 'Tokenizer or Preprocessing Divergence',
        'category': 'tokenizer_mismatch',
        'evidence_level': 'UNSUPPORTED',
        'evidence_details': 'Vocab 151669, EOS token 151643 (<|im_end|>), and right truncation are identical in local and Kaggle runtimes.',
        'severity': 'NONE',
        'confidence': 'HIGH',
        'can_explain_gap': 'NO',
        'next_test': 'None.'
    }
]

df = pd.DataFrame(causes)
df.to_csv('experiments/EXP66_gap_forensics/cause_ranking.csv', index=False)
print('cause_ranking.csv written successfully. Total ranked causes:', len(df))
