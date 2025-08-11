#%%
# test_linear_regressor.py
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# === Config ===
label = 'age'
cohort_train = 'ixi' 
cohort_test = 'adni1'
model = 'sfcne'
data_csv='ad-cn'

feature_csv = f'../../features/{cohort_test}/{model}/_features.csv'
label_csv   = f'../../data/{cohort_test}/{data_csv}.csv'
model_path  = f'../../models/{model}/{label}_{cohort_train}.joblib'  # trained on adni1
output_csv  = f'../../scores/{model}/test/{cohort_test}/{label}.csv'

#%%
# === Load data
df_features = pd.read_csv(feature_csv)
df_labels = pd.read_csv(label_csv)

if 'eid__sequence' in df_labels.columns:
    df_labels = df_labels.rename(columns={'eid__sequence': 'eid'})

# Merge features and labels
df = df_features.merge(df_labels[['eid', label]], on='eid', how='inner')

# Extract data
feature_cols = [col for col in df.columns if col.isdigit()]
X = df[feature_cols].values.astype(np.float32)
y = df[label].values
eids = df['eid'].values

#%%
# Load model
reg = joblib.load(model_path)

# Predict
y_pred_raw = reg.predict(X)

# Bias correction using linear fit (based on original training/val cohort)
bias_model = LinearRegression().fit(y_pred_raw.reshape(-1, 1), y)
y_pred_corr = bias_model.predict(y_pred_raw.reshape(-1, 1))

#%%
# Save predictions
df_out = pd.DataFrame({
    'eid': eids,
    'label': y,
    'prediction_raw': y_pred_raw,
    'prediction_bias_corrected': y_pred_corr
})
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df_out.to_csv(output_csv, index=False)
print(f"✅ Saved predictions to: {output_csv}")

#%%
# Evaluation
mae_raw = mean_absolute_error(y, y_pred_raw)
r2_raw = r2_score(y, y_pred_raw)
mae_corr = mean_absolute_error(y, y_pred_corr)
r2_corr = r2_score(y, y_pred_corr)

print(f"🎯 External Test on {cohort_test.upper()}")
print(f"   → Raw     — MAE: {mae_raw:.2f}, R²: {r2_raw:.4f}")
print(f"   → Corrected — MAE: {mae_corr:.2f}, R²: {r2_corr:.4f}")

#%%
# Optional: Plot predictions (raw and corrected)
fontsize = 18
fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Raw
ax[0].scatter(y, y_pred_raw, alpha=0.6)
ax[0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
ax[0].set_title("Raw Predictions", fontsize=fontsize)
ax[0].set_xlabel("True Age")
ax[0].set_ylabel("Predicted Age")
ax[0].grid(True)

# Corrected
ax[1].scatter(y, y_pred_corr, alpha=0.6)
ax[1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
ax[1].set_title("Bias-Corrected Predictions", fontsize=fontsize)
ax[1].set_xlabel("True Age")
ax[1].grid(True)

plt.suptitle(f"Age Prediction — {model.upper()} on {cohort_test.upper()}", fontsize=fontsize + 2)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
# %%
