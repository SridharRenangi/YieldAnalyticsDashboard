import pandas as pd
import numpy as np

df        = pd.read_csv('data/secom_clean.csv')
results   = pd.read_csv('outputs/parameter_significance.csv')
top20     = results.head(20)
top_cols  = top20['parameter'].tolist()

# ── EXPORT 1: YIELD SUMMARY ───────────────────────────────────
# Overall pass/fail counts and rates — for KPI cards in Tableau
summary = pd.DataFrame({
    'metric': ['Total Wafers', 'Passing Wafers', 'Failing Wafers', 'Yield Rate (%)', 'Failure Rate (%)'],
    'value':  [len(df),
               int((df['label']==-1).sum()),
               int((df['label']==1).sum()),
               round((df['label']==-1).mean()*100, 2),
               round((df['label']==1).mean()*100, 2)]
})
summary.to_csv('outputs/tableau_yield_summary.csv', index=False)
print('Saved: outputs/tableau_yield_summary.csv')

# ── EXPORT 2: PARAMETER COMPARISON TABLE ─────────────────────
# Top 20 parameters with pass/fail means — for bar chart in Tableau
comparison_rows = []
pass_df = df[df['label']==-1]
fail_df = df[df['label']==1]

for _, row in top20.iterrows():
    col = row['parameter']
    comparison_rows.append({'parameter': col, 'group': 'Pass',
                            'mean_value': round(pass_df[col].mean(), 4),
                            'p_value': row['p_value']})
    comparison_rows.append({'parameter': col, 'group': 'Fail',
                            'mean_value': round(fail_df[col].mean(), 4),
                            'p_value': row['p_value']})

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv('outputs/tableau_param_comparison.csv', index=False)
print('Saved: outputs/tableau_param_comparison.csv')

# ── EXPORT 3: WAFER-LEVEL DATA FOR TOP 5 PARAMETERS ──────────
# Individual wafer readings for scatter plots in Tableau
top5_cols = top20.head(5)['parameter'].tolist()
wafer_df  = df[top5_cols + ['label']].copy()
wafer_df['outcome'] = wafer_df['label'].map({-1: 'Pass', 1: 'Fail'})
wafer_df['wafer_id'] = range(1, len(wafer_df)+1)
wafer_df.drop(columns=['label'], inplace=True)
wafer_df.to_csv('outputs/tableau_wafer_level.csv', index=False)
print('Saved: outputs/tableau_wafer_level.csv')

print('\nAll Tableau export files ready in outputs/')
