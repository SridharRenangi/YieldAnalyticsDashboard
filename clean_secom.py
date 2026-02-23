import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

df = pd.read_csv('data/secom_combined.csv')

# Separate features and labels
y    = df['label']
X    = df.drop(columns=['label'])
cols = list(X.columns)

print(f'Starting parameters: {X.shape[1]}')

# ── STEP 1: REMOVE ALL-NaN COLUMNS ───────────────────────────
# Some columns are entirely missing — useless for analysis
all_nan_cols = X.columns[X.isnull().all()].tolist()
X = X.drop(columns=all_nan_cols)
print(f'After removing all-NaN columns:  {X.shape[1]} parameters ({len(all_nan_cols)} removed)')

# ── STEP 2: FILL REMAINING MISSING VALUES WITH COLUMN MEDIAN ─
# Median is more robust to outliers than mean
X = X.fillna(X.median())
print(f'Missing values after imputation: {X.isnull().sum().sum()}')

# ── STEP 3: REMOVE NEAR-ZERO VARIANCE COLUMNS ────────────────
# Parameters that barely change cannot explain yield variation
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)
surviving_mask = selector.get_support()
surviving_cols = [c for c, keep in zip(X.columns, surviving_mask) if keep]
X = pd.DataFrame(X_filtered, columns=surviving_cols)
print(f'After variance filter:           {X.shape[1]} parameters')

# ── COMBINE AND SAVE ──────────────────────────────────────────
df_clean = X.copy()
df_clean['label'] = y.values
df_clean.to_csv('data/secom_clean.csv', index=False)

print(f'\nClean dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]-1} parameters')
print(f'Saved to data/secom_clean.csv')

# Save the list of surviving parameter names for reference
with open('data/surviving_params.txt', 'w') as f:
    f.write('\n'.join(surviving_cols))
print(f'Parameter list saved to data/surviving_params.txt')
