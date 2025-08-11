#%%
# test_linear_classifier.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# === Config ===
label = 'sex'                # or 'ftd', 'ad'
cohort_train = 'ukb' 
cohort_test = '4rtni'             # external test cohort
model = 'sfcne'              # model to test (must match trained model)
csv= 'ftd-cn'

feature_csv = f'../../features/{cohort_test}/{model}/_features.csv'
label_csv   = f'../../data/{cohort_test}/{csv}.csv'
model_path  = f'../../models/{model}/{label}_{cohort_train}.joblib'  # model trained on PPMI
output_csv  = f'../../scores/{model}/test/{cohort_test}/{label}.csv'
#%%
# === Load data
df_features = pd.read_csv(feature_csv)
df_labels = pd.read_csv(label_csv)
print(df_features)
print(df_labels)
if 'eid__sequence' in df_labels.columns:
    df_labels = df_labels.rename(columns={'eid__sequence': 'eid'})

# Merge features and labels
df = df_features.merge(df_labels[['eid', label]], on='eid', how='inner')
print(df)
# Extract features and labels
feature_cols = [col for col in df.columns if col.isdigit()]
X = df[feature_cols].values.astype(np.float32)
y = df[label].values
eids = df['eid'].values
#%%
# Load model
clf = joblib.load(model_path)

# Predict
y_score = clf.predict_proba(X)[:, 1]
y_pred = (y_score > 0.5).astype(int)
#%%
# Save scores
df_out = pd.DataFrame({
    'eid': eids,
    'label': y,
    'prediction': y_score,
    'pred_class': y_pred
})
df_out.to_csv(output_csv, index=False)
print(f"✅ Saved scores to: {output_csv}")

# Optional: Evaluate
auroc = roc_auc_score(y, y_score)
acc = accuracy_score(y, y_pred)
print(f"🎯 External Test on {cohort_test.upper()} — AUROC: {auroc:.4f}, ACC: {acc:.4f}")
#%%
# Optional: ROC plot
fpr, tpr, _ = roc_curve(y, y_score)
roc_auc = auc(fpr, tpr)
fontsize=20
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"{model.upper()} on {cohort_test.upper()} (AUROC = {roc_auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve — {model.upper()} on {cohort_test.upper()}", fontsize=16)
plt.legend(fontsize=fontsize)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
