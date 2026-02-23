import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('outputs', exist_ok=True)

df = pd.read_csv('data/secom_clean.csv')

# Separate pass and fail groups
pass_df = df[df['label'] == -1].drop(columns=['label'])
fail_df = df[df['label'] ==  1].drop(columns=['label'])
param_cols = [c for c in df.columns if c != 'label']

print(f'Pass wafers: {len(pass_df)}')
print(f'Fail wafers: {len(fail_df)}')
print(f'Running t-test on {len(param_cols)} parameters...')

# ── RUN T-TEST FOR EVERY PARAMETER ───────────────────────────
results = []
for col in param_cols:
    p_vals  = pass_df[col].dropna()
    f_vals  = fail_df[col].dropna()

    if len(p_vals) < 3 or len(f_vals) < 3:
        continue  # Skip if not enough data

    t_stat, p_value = stats.ttest_ind(p_vals, f_vals, equal_var=False)

    results.append({
        'parameter':    col,
        'p_value':      p_value,
        'pass_mean':    round(p_vals.mean(), 4),
        'fail_mean':    round(f_vals.mean(), 4),
        'pass_std':     round(p_vals.std(),  4),
        'fail_std':     round(f_vals.std(),  4),
        'mean_diff':    round(f_vals.mean() - p_vals.mean(), 4),
        'pct_diff':     round((f_vals.mean() - p_vals.mean()) / (abs(p_vals.mean()) + 1e-9) * 100, 2)
    })

# Sort by p-value ascending — most significant parameters first
results_df = pd.DataFrame(results).sort_values('p_value').reset_index(drop=True)

# Save full results
results_df.to_csv('outputs/parameter_significance.csv', index=False)

# Show top 20 most significant
top20 = results_df.head(20)
print('\nTop 20 yield-limiting parameters:')
print(top20[['parameter','p_value','pass_mean','fail_mean','pct_diff']].to_string())

sig_count = (results_df['p_value'] < 0.05).sum()
print(f'\nParameters with p < 0.05: {sig_count} of {len(results_df)}')
print(f'Full results saved to outputs/parameter_significance.csv')

# ── CHART 1: TOP 20 PARAMETERS BY SIGNIFICANCE ───────────────
fig, ax = plt.subplots(figsize=(12, 7))
colors  = ['#B71C1C' if p < 0.001 else '#E53935' if p < 0.01 else '#EF9A9A'
           for p in top20['p_value']]
bars = ax.barh(top20['parameter'], -np.log10(top20['p_value']), color=colors)
ax.axvline(-np.log10(0.05), color='orange', linestyle='--', linewidth=1.5, label='p=0.05 threshold')
ax.axvline(-np.log10(0.01), color='red',    linestyle='--', linewidth=1.5, label='p=0.01 threshold')
ax.set_xlabel('-log10(p-value)  —  Higher = More Significant', fontsize=11)
ax.set_title('Top 20 Parameters by Yield Significance (SECOM Dataset)', fontweight='bold', fontsize=13)
ax.legend(fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/top20_significance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: outputs/top20_significance.png')

# ── CHART 2: PASS VS FAIL DISTRIBUTION FOR TOP PARAMETER ─────
top_param = top20.iloc[0]['parameter']
fig, ax   = plt.subplots(figsize=(9, 5))
pass_vals = pass_df[top_param].dropna()
fail_vals = fail_df[top_param].dropna()
ax.hist(pass_vals, bins=40, alpha=0.6, color='#1565C0', label=f'Pass (n={len(pass_vals)})')
ax.hist(fail_vals, bins=20, alpha=0.7, color='#B71C1C', label=f'Fail (n={len(fail_vals)})')
ax.set_title(f'Pass vs Fail Distribution — {top_param}  (p={top20.iloc[0]["p_value"]:.2e})',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Parameter Value')
ax.set_ylabel('Count')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('outputs/top_param_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: outputs/top_param_distribution.png')
