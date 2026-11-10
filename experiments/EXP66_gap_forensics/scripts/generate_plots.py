import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

os.makedirs('experiments/EXP66_gap_forensics/plots', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Load data
df_conf = pd.read_csv('experiments/EXP66_gap_forensics/confidence_analysis.csv')
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
df_surface = pd.read_csv('experiments/EXP66_gap_forensics/class_surface_shift.csv')
df_counter = pd.read_csv('experiments/EXP66_gap_forensics/counterfactual_fusion.csv')

# Deciles
df_conf['conf_decile'] = pd.qcut(df_conf['conf_fused'], 10, labels=False)
df_conf['entropy_decile'] = pd.qcut(df_conf['entropy'], 10, labels=False)
df_conf['margin_decile'] = pd.qcut(df_conf['margin'], 10, labels=False)

# 1. Plot: Confidence vs Error
plt.figure(figsize=(8, 5), dpi=150)
conf_agg = df_conf.groupby('conf_decile').agg({'conf_fused': 'mean', 'error': 'mean'}).reset_index()
plt.plot(conf_agg['conf_fused'], conf_agg['error'] * 100, marker='o', linewidth=2.5, color='#d9534f', label='Empirical Error Rate (%)')
plt.title('EXP52: Confidence Decile vs Attribution Error Rate', fontsize=14, fontweight='bold')
plt.xlabel('Mean Fused Confidence (p_fused)', fontsize=12)
plt.ylabel('Attribution Error Rate (%)', fontsize=12)
plt.ylim(0, 70)
for _, r in conf_agg.iterrows():
    plt.annotate(f"{r['error']*100:.1f}%", (r['conf_fused'], r['error']*100 + 2), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('experiments/EXP66_gap_forensics/plots/confidence_vs_error.png')
plt.close()

# 2. Plot: Entropy vs Error
plt.figure(figsize=(8, 5), dpi=150)
ent_agg = df_conf.groupby('entropy_decile').agg({'entropy': 'mean', 'error': 'mean'}).reset_index()
plt.plot(ent_agg['entropy'], ent_agg['error'] * 100, marker='s', linewidth=2.5, color='#f0ad4e', label='Error Rate (%)')
plt.title('EXP52: Predictive Entropy vs Attribution Error Rate', fontsize=14, fontweight='bold')
plt.xlabel('Mean Predictive Entropy (nats)', fontsize=12)
plt.ylabel('Attribution Error Rate (%)', fontsize=12)
plt.ylim(0, 70)
for _, r in ent_agg.iterrows():
    plt.annotate(f"{r['error']*100:.1f}%", (r['entropy'], r['error']*100 + 2), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('experiments/EXP66_gap_forensics/plots/entropy_vs_error.png')
plt.close()

# 3. Plot: Margin vs Error
plt.figure(figsize=(8, 5), dpi=150)
mar_agg = df_conf.groupby('margin_decile').agg({'margin': 'mean', 'error': 'mean'}).reset_index()
plt.plot(mar_agg['margin'], mar_agg['error'] * 100, marker='^', linewidth=2.5, color='#5bc0de', label='Error Rate (%)')
plt.title('EXP52: Top-1 vs Top-2 Probability Margin vs Error Rate', fontsize=14, fontweight='bold')
plt.xlabel('Mean Margin (p_top1 - p_top2)', fontsize=12)
plt.ylabel('Attribution Error Rate (%)', fontsize=12)
plt.ylim(0, 70)
for _, r in mar_agg.iterrows():
    plt.annotate(f"{r['error']*100:.1f}%", (r['margin'], r['error']*100 + 2), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('experiments/EXP66_gap_forensics/plots/margin_vs_error.png')
plt.close()

# 4. Plot: Train vs Test Length Distributions
plt.figure(figsize=(8, 5), dpi=150)
sns.kdeplot(train_df['text'].str.len(), color='#0275d8', label='Train Texts (N=2,699)', fill=True, alpha=0.3)
sns.kdeplot(test_df['text'].str.len(), color='#d9534f', label='Test Texts (N=3 sample)', fill=True, alpha=0.3)
plt.title('Train vs Test Character Length Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Character Length', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('experiments/EXP66_gap_forensics/plots/train_test_length_distributions.png')
plt.close()

# 5. Plot: Train vs Test Token Distributions
plt.figure(figsize=(8, 5), dpi=150)
sns.kdeplot(train_df['text'].str.split().str.len() * 1.3, color='#5cb85c', label='Train Tokens (approx)', fill=True, alpha=0.3)
sns.kdeplot(test_df['text'].str.split().str.len() * 1.3, color='#f0ad4e', label='Test Tokens (approx)', fill=True, alpha=0.3)
plt.title('Train vs Test Approximate Token Count Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Token Count', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('experiments/EXP66_gap_forensics/plots/train_test_token_distributions.png')
plt.close()

# 6. Plot: Class Confusion Matrix Heatmap
labels = sorted(train_df['model'].unique())
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(df_conf['true_idx'], df_conf['pred_fused_idx'])
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10), dpi=150)
sns.heatmap(cm_norm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('EXP52: 18x18 Normalized Confusion Matrix (OOF)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Model Class', fontsize=11)
plt.ylabel('True Model Class', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig('experiments/EXP66_gap_forensics/plots/class_confusion_matrix.png')
plt.close()

# 7. Plot: Class Surface Statistics
plt.figure(figsize=(12, 6), dpi=150)
df_surface_sorted = df_surface.sort_values(by='mean_char_length', ascending=False)
plt.bar(df_surface_sorted['model_class'], df_surface_sorted['mean_char_length'], color='#0275d8', alpha=0.85)
plt.title('Mean Character Length per Model Class (Training Data)', fontsize=14, fontweight='bold')
plt.xlabel('Model Class', fontsize=11)
plt.ylabel('Mean Character Length', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig('experiments/EXP66_gap_forensics/plots/class_surface_statistics.png')
plt.close()

# 8. Plot: Q8/Q14 Agreement Analysis
plt.figure(figsize=(7, 5), dpi=150)
agree_counts = df_conf['q8_q14_agreement'].value_counts()
agree_accs = df_conf.groupby('q8_q14_agreement')['correct'].mean() * 100
categories = ['Disagreement\n(59.7% of data)', 'Agreement\n(40.3% of data)']
acc_vals = [agree_accs[False], agree_accs[True]]
err_vals = [100 - agree_accs[False], 100 - agree_accs[True]]

x = np.arange(2)
plt.bar(x - 0.15, acc_vals, width=0.3, label='Accuracy (%)', color='#5cb85c')
plt.bar(x + 0.15, err_vals, width=0.3, label='Error Rate (%)', color='#d9534f')
plt.xticks(x, categories, fontsize=11)
plt.ylabel('Percentage (%)', fontsize=12)
plt.title('Attribution Accuracy & Error: Agreement vs Disagreement', fontsize=13, fontweight='bold')
plt.ylim(0, 105)
for i in range(2):
    plt.text(i - 0.15, acc_vals[i] + 2, f"{acc_vals[i]:.1f}%", ha='center', fontweight='bold')
    plt.text(i + 0.15, err_vals[i] + 2, f"{err_vals[i]:.1f}%", ha='center', fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('experiments/EXP66_gap_forensics/plots/q8_q14_agreement_analysis.png')
plt.close()

# 9. Plot: Fusion Weight Diagnostic
plt.figure(figsize=(8, 5), dpi=150)
plt.plot(df_counter['weight_q14'], df_counter['balanced_accuracy'], marker='o', linewidth=2.5, color='#6f42c1')
plt.axvline(0.40, color='#d9534f', linestyle='--', label='EXP52 Production (w_14=0.40)')
plt.title('EXP52: Balanced Accuracy across Fusion Weights (Q8 vs Q14)', fontsize=14, fontweight='bold')
plt.xlabel('Q14 Weight (w_14) [Q8 Weight = 1 - w_14]', fontsize=12)
plt.ylabel('OOF Balanced Accuracy', fontsize=12)
plt.ylim(0.55, 0.82)
for _, r in df_counter.iterrows():
    plt.annotate(f"{r['balanced_accuracy']:.4f}", (r['weight_q14'], r['balanced_accuracy'] + 0.008), ha='center', fontsize=8)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('experiments/EXP66_gap_forensics/plots/fusion_weight_diagnostic.png')
plt.close()

print('All 9 publication plots generated successfully in experiments/EXP66_gap_forensics/plots/')
