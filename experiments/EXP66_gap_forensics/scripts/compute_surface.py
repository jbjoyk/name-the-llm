import pandas as pd
import numpy as np
import scipy.stats
import string

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

def extract_surface_metrics(texts):
    chars = np.array([len(t) for t in texts])
    words = np.array([len(t.split()) for t in texts])
    tokens = np.array([len(t.split()) * 1.3 for t in texts])
    
    punct_count = np.array([sum(1 for c in t if c in string.punctuation) for t in texts])
    punct_ratio = punct_count / np.clip(chars, 1, None)
    
    space_count = np.array([sum(1 for c in t if c.isspace()) for t in texts])
    space_ratio = space_count / np.clip(chars, 1, None)
    
    digit_count = np.array([sum(1 for c in t if c.isdigit()) for t in texts])
    digit_ratio = digit_count / np.clip(chars, 1, None)
    
    upper_count = np.array([sum(1 for c in t if c.isupper()) for t in texts])
    upper_ratio = upper_count / np.clip(chars, 1, None)
    
    code_indicators = np.array([1.0 if ('```' in t or 'def ' in t or ('{' in t and '}' in t)) else 0.0 for t in texts])
    
    md_indicators = np.array([1.0 if any(line.strip().startswith(('#', '*', '-')) for line in t.split('\n')) else 0.0 for t in texts])
    
    newline_count = np.array([t.count('\n') for t in texts])
    newline_density = newline_count / np.clip(words, 1, None)
    
    return {
        'char_length': chars,
        'word_count': words,
        'token_count': tokens,
        'punct_ratio': punct_ratio,
        'whitespace_ratio': space_ratio,
        'digit_ratio': digit_ratio,
        'uppercase_ratio': upper_ratio,
        'code_formatting': code_indicators,
        'markdown_frequency': md_indicators,
        'newline_density': newline_density
    }

train_metrics = extract_surface_metrics(train_df['text'].fillna(''))
test_metrics = extract_surface_metrics(test_df['text'].fillna(''))

surface_rows = []
for k in train_metrics.keys():
    tr_v = train_metrics[k]
    te_v = test_metrics[k]
    
    mean_tr = float(np.mean(tr_v))
    std_tr = float(np.std(tr_v))
    mean_te = float(np.mean(te_v))
    std_te = float(np.std(te_v))
    
    pooled_std = np.sqrt((std_tr**2 + std_te**2) / 2.0)
    cohens_d = float((mean_te - mean_tr) / (pooled_std + 1e-9))
    
    ks_stat, ks_pval = scipy.stats.ks_2samp(tr_v, te_v)
    
    surface_rows.append({
        'feature': k,
        'train_mean': mean_tr,
        'train_std': std_tr,
        'test_mean': mean_te,
        'test_std': std_te,
        'cohens_d': cohens_d,
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'shift_magnitude': 'NEGLIGIBLE' if abs(cohens_d) < 0.2 else ('MODERATE' if abs(cohens_d) < 0.5 else 'LARGE')
    })

df_surface = pd.DataFrame(surface_rows)
df_surface.to_csv('experiments/EXP66_gap_forensics/surface_shift.csv', index=False)
print('surface_shift.csv written successfully:')
print(df_surface[['feature', 'train_mean', 'test_mean', 'cohens_d', 'shift_magnitude']])

labels = sorted(train_df['model'].unique())
class_rows = []
for l in labels:
    sub = train_df[train_df['model'] == l]
    sub_m = extract_surface_metrics(sub['text'].fillna(''))
    
    class_rows.append({
        'model_class': l,
        'sample_count': len(sub),
        'mean_char_length': float(np.mean(sub_m['char_length'])),
        'std_char_length': float(np.std(sub_m['char_length'])),
        'mean_word_count': float(np.mean(sub_m['word_count'])),
        'mean_token_count': float(np.mean(sub_m['token_count'])),
        'mean_punct_ratio': float(np.mean(sub_m['punct_ratio'])),
        'mean_whitespace_ratio': float(np.mean(sub_m['whitespace_ratio'])),
        'mean_uppercase_ratio': float(np.mean(sub_m['uppercase_ratio'])),
        'code_formatting_rate': float(np.mean(sub_m['code_formatting'])),
        'markdown_rate': float(np.mean(sub_m['markdown_frequency'])),
        'newline_density': float(np.mean(sub_m['newline_density']))
    })

df_class_surface = pd.DataFrame(class_rows)
df_class_surface.to_csv('experiments/EXP66_gap_forensics/class_surface_shift.csv', index=False)
print('class_surface_shift.csv written successfully. Total classes:', len(df_class_surface))
