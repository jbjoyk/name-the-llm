import pandas as pd
import numpy as np
import scipy.special
from sklearn.metrics import confusion_matrix

train_df = pd.read_csv('data/train.csv')
labels = sorted(train_df['model'].unique())
l2i = {l: i for i, l in enumerate(labels)}
i2l = {i: l for i, l in enumerate(labels)}
y_true = train_df['model'].map(l2i).values

p8 = np.load('outputs/oof_exp47_qwen_centered_twin.npy')
q14_df = pd.read_csv('outputs/oof_q14v5_antopil_lora.csv')
p14 = q14_df[labels].values

lp8 = np.log(np.clip(p8, 1e-15, 1.0))
lp14 = np.log(np.clip(p14, 1e-15, 1.0))
lp_fused = 0.60 * lp8 + 0.40 * lp14
p_fused = np.exp(lp_fused - scipy.special.logsumexp(lp_fused, axis=1, keepdims=True))

pred_8 = p8.argmax(axis=1)
pred_14 = p14.argmax(axis=1)
pred_fused = p_fused.argmax(axis=1)

sorted_p = np.sort(p_fused, axis=1)
margins = sorted_p[:, -1] - sorted_p[:, -2]
confidences = p_fused.max(axis=1)

cm = confusion_matrix(y_true, pred_fused, labels=range(18))

# Find all confusion pairs (i != j)
pairs = []
for true_i in range(18):
    for pred_j in range(18):
        if true_i == pred_j:
            continue
        err_count = cm[true_i, pred_j]
        if err_count == 0:
            continue
        
        # Mask of samples belonging to this confusion pair
        mask = (y_true == true_i) & (pred_fused == pred_j)
        rev_count = cm[pred_j, true_i]
        
        asymmetry = abs(err_count - rev_count)
        norm_rate = err_count / cm[true_i, :].sum()
        
        # Breakdown by Q8, Q14
        q8_err = (pred_8[mask] == pred_j).mean()
        q14_err = (pred_14[mask] == pred_j).mean()
        both_err = ((pred_8[mask] == pred_j) & (pred_14[mask] == pred_j)).mean()
        disagree_rate = (pred_8[mask] != pred_14[mask]).mean()
        
        mean_conf = confidences[mask].mean()
        mean_margin = margins[mask].mean()
        
        # Primary driver
        if both_err > 0.5:
            driver = 'BOTH_BACKBONES'
        elif q8_err > q14_err:
            driver = 'PRIMARY_Q8'
        elif q14_err > q8_err:
            driver = 'PRIMARY_Q14'
        else:
            driver = 'FUSION_INTERACTION'
            
        conf_regime = 'HIGH_CONFIDENCE' if mean_conf > 0.65 else 'LOW_CONFIDENCE'
        
        pairs.append({
            'true_class': i2l[true_i],
            'pred_class': i2l[pred_j],
            'error_count': int(err_count),
            'reverse_error_count': int(rev_count),
            'pair_asymmetry': int(asymmetry),
            'normalized_rate': float(norm_rate),
            'q8_error_rate': float(q8_err),
            'q14_error_rate': float(q14_err),
            'both_error_rate': float(both_err),
            'disagreement_rate': float(disagree_rate),
            'mean_confidence': float(mean_conf),
            'mean_margin': float(mean_margin),
            'error_driver': driver,
            'confidence_regime': conf_regime
        })

df_pairs = pd.DataFrame(pairs).sort_values(by='error_count', ascending=False).reset_index(drop=True)
df_pairs.to_csv('experiments/EXP66_gap_forensics/confusion_analysis.csv', index=False)
print('confusion_analysis.csv written successfully. Top 10 failure pairs:')
print(df_pairs[['true_class', 'pred_class', 'error_count', 'mean_confidence', 'error_driver']].head(10))
