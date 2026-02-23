import pandas as pd
import numpy as np

# ── LOAD ──────────────────────────────────────────────────────
# secom.data is space-separated with no header row
# 1567 rows (wafer runs), 590 columns (process parameters)
X = pd.read_csv('data/secom.data', sep=' ', header=None)

# secom_labels.data has two columns: label (-1=pass, 1=fail) and timestamp
labels = pd.read_csv('data/secom_labels.data', sep=' ', header=None,
                     names=['label', 'timestamp'])

# Combine into one DataFrame
# Give columns descriptive names: param_0 through param_589
X.columns = [f'param_{i}' for i in range(X.shape[1])]
df = pd.concat([X, labels['label']], axis=1)

print('=== DATASET OVERVIEW ===')
print(f'Shape:          {df.shape}')           # (1567, 591)
print(f'Wafer runs:     {len(df)}')
print(f'Parameters:     {X.shape[1]}')
print()

print('=== YIELD BREAKDOWN ===')
pass_count = (df['label'] == -1).sum()
fail_count = (df['label'] ==  1).sum()
print(f'Passing wafers: {pass_count} ({pass_count/len(df)*100:.1f}%)')
print(f'Failing wafers: {fail_count} ({fail_count/len(df)*100:.1f}%)')
print()

print('=== MISSING VALUES ===')
total_cells    = df.shape[0] * df.shape[1]
missing_cells  = df.isnull().sum().sum()
missing_cols   = (df.isnull().sum() > 0).sum()
print(f'Missing cells:   {missing_cells} of {total_cells} ({missing_cells/total_cells*100:.1f}%)')
print(f'Columns with any missing value: {missing_cols}')
print()

print('=== SAMPLE DATA ===')
print(df[['param_0','param_1','param_2','param_3','label']].head(8))

# Save the combined DataFrame for use in the next steps
df.to_csv('data/secom_combined.csv', index=False)
print('\nSaved combined data to data/secom_combined.csv')
