import pandas as pd
import numpy as np
import scipy.special
from sklearn.metrics import balanced_accuracy_score, f1_score

train_df = pd.read_csv('data/train.csv')
labels = sorted(train_df['model'].unique())
l2i = {l: i for i, l in enumerate(labels)}
y_true = train_df['model'].map(l2i).values

p8 = np.load('outputs/oof_exp47_qwen_centered_twin.npy')
q14_df = pd.read_csv('outputs/oof_q14v5_antopil_lora.csv')
p14 = q14_df[labels].values

lp8 = np.log(np.clip(p8, 1e-15, 1.0))
lp14 = np.log(np.clip(p14, 1e-15, 1.0))

weights = [
    ('Q8_only (1.00 / 0.00)', 1.00, 0.00),
    ('0.90 Q8 / 0.10 Q14', 0.90, 0.10),
    ('0.75 Q8 / 0.25 Q14', 0.75, 0.25),
    ('0.60 Q8 / 0.40 Q14 (EXP52 Production)', 0.60, 0.40),
    ('0.50 Q8 / 0.50 Q14', 0.50, 0.50),
    ('0.40 Q8 / 0.60 Q14', 0.40, 0.60),
    ('0.25 Q8 / 0.75 Q14', 0.25, 0.75),
    ('Q14_only (0.00 / 1.00)', 0.00, 1.00)
]

rows = []
for label, w8, w14 in weights:
    if w14 == 0.0:
        lp_fused = lp8
    elif w8 == 0.0:
        lp_fused = lp14
    else:
        lp_fused = w8 * lp8 + w14 * lp14
        
    p_fused = np.exp(lp_fused - scipy.special.logsumexp(lp_fused, axis=1, keepdims=True))
    preds = p_fused.argmax(axis=1)
    
    ba = balanced_accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='macro')
    err_cnt = int((preds != y_true).sum())
    mean_conf = float(p_fused.max(axis=1).mean())
    entropy = float(-np.sum(p_fused * np.log(np.clip(p_fused, 1e-15, 1.0)), axis=1).mean())
    
    rows.append({
        'configuration': label,
        'weight_q8': w8,
        'weight_q14': w14,
        'balanced_accuracy': float(ba),
        'macro_f1': float(f1),
        'error_count': err_cnt,
        'delta_vs_exp52': float(ba - 0.787313),
        'mean_confidence': mean_conf,
        'mean_entropy': entropy
    })

df_counter = pd.DataFrame(rows)
df_counter.to_csv('experiments/EXP66_gap_forensics/counterfactual_fusion.csv', index=False)
print('counterfactual_fusion.csv written successfully:')
print(df_counter[['configuration', 'balanced_accuracy', 'delta_vs_exp52', 'error_count']])
