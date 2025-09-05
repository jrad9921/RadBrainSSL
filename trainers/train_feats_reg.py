
#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import joblib

# === Config ===
label = 'bpf'
cohort = 'mspaths'
model = 'sfcne'
csv = 'bpf'
feature_csv = f'../../features/{cohort}/{model}/_features.csv'

train_csv = f'../../data/{cohort}/train/{csv}.csv'
test_csv  = f'../../data/{cohort}/test/{csv}.csv'

# Output paths
model_dir = f'../../models/{model}'
os.makedirs(model_dir, exist_ok=True)
model_path = f'{model_dir}/{label}_{cohort}.joblib'
scaler_path = f'{model_dir}/{label}_{cohort}_scaler.joblib'
bias_path   = f'{model_dir}/{label}_{cohort}_bias.joblib'
plot_path   = f'{model_dir}/{label}_{cohort}_pred_vs_true.png'

# === Load features & splits
df_features = pd.read_csv(feature_csv)
# feature columns are "0","1",..."2999"
feature_cols = [c for c in df_features.columns if c.isdigit()]
assert len(feature_cols) > 0, "No numeric-named feature columns found."

df_train = pd.read_csv(train_csv)
df_test  = pd.read_csv(test_csv)

# Standardize EID column name
for df in (df_train, df_test):
    if 'eid__sequence' in df.columns:
        df.rename(columns={'eid__sequence': 'eid'}, inplace=True)

# Merge features (inner to ensure alignment)
df_train = df_train.merge(df_features, on='eid', how='inner')
df_test  = df_test.merge(df_features, on='eid', how='inner')

# Drop rows with missing target
df_train = df_train.dropna(subset=[label]).reset_index(drop=True)
df_test  = df_test.dropna(subset=[label]).reset_index(drop=True)

# Extract X/y
X_train = df_train[feature_cols].to_numpy(dtype=np.float32)
y_train = df_train[label].to_numpy(dtype=np.float32)
X_test  = df_test[feature_cols].to_numpy(dtype=np.float32)
y_test  = df_test[label].to_numpy(dtype=np.float32)

# === Scale features (fit on train only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)
joblib.dump(scaler, scaler_path)

# === Train ridge regression
reg = Ridge(alpha=1.0, random_state=42)
reg.fit(X_train, y_train)
joblib.dump(reg, model_path)
print(f"✅ Model saved to: {model_path}")
print(f"✅ Scaler saved to: {scaler_path}")

# === Bias calibration (optional but recommended)
# Fit on train predictions -> train targets
train_pred = reg.predict(X_train)
bias_model = LinearRegression().fit(train_pred.reshape(-1, 1), y_train)
joblib.dump(bias_model, bias_path)
print(f"✅ Bias model saved to: {bias_path}")

def apply_bias(model_lin, y_pred):
    return model_lin.predict(y_pred.reshape(-1, 1))

# === Evaluate
def eval_and_print(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    try:
        r, _ = pearsonr(y_true, y_pred)
    except Exception:
        r = np.nan
    print(f"{name:>24} — MAE: {mae:.3f} | R²: {r2:.4f} | r: {r:.4f}")
    return mae, r2, r

test_pred_raw = reg.predict(X_test)
eval_and_print("Test (raw)", y_test, test_pred_raw)

test_pred_cal = apply_bias(bias_model, test_pred_raw)
mae, r2, r = eval_and_print("Test (bias-calibrated)", y_test, test_pred_cal)

# === Scatter plot: True vs Predicted
plt.figure(figsize=(6,6))  # square
plt.scatter(y_test, test_pred_cal, s=18, alpha=0.6, edgecolor='none')

# identity line
mn = float(np.min([y_test.min(), test_pred_cal.min()]))
mx = float(np.max([y_test.max(), test_pred_cal.max()]))
pad = 0.05 * (mx - mn) if mx > mn else 1.0
lo, hi = mn - pad, mx + pad
plt.plot([lo, hi], [lo, hi], linewidth=2.5, linestyle='--')

plt.xlabel("True", fontsize=16)
plt.ylabel("Predicted (calibrated)", fontsize=16)
plt.xlim(lo, hi)
plt.ylim(lo, hi)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, alpha=0.3)

# Metrics box
txt = f"MAE = {mae:.2f}\nR² = {r2:.3f}\nr = {r:.3f}"
plt.gca().text(0.05, 0.95, txt, transform=plt.gca().transAxes,
               va='top', ha='left', fontsize=14,
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.tight_layout()
plt.show()
#plt.savefig(plot_path, dpi=200)
plt.close()


# %%

# %%
