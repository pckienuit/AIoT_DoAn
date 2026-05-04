import pandas as pd
df = pd.read_csv('results/eval_v9_fixed.csv')
ranked = df.sort_values('combined_loss').reset_index(drop=True)
cols = ['model_name','combined_loss','cls_f1','cls_auc_roc','landmark_nme','landmark_mae']
print("\n=== RANKING (Combined Loss - lower is better) ===")
print(ranked[cols].to_string(index=False))
print("\n=== v9 specifically ===")
v9 = df[df['model_name'].str.contains('v9')]
print(v9[cols].to_string(index=False))
