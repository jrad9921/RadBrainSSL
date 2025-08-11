#%%
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import joblib  # for saving model

# === Config ===
label = 'age'
cohort = 'ixi'
model = 'sfcne'
csv = 'demographics'
feature_csv = f'../../features/{cohort}/{model}/_features.csv'

train_csv = f'../../data/{cohort}/train/{csv}.csv'
val_csv   = f'../../data/{cohort}/val/{csv}.csv'
test_csv  = f'../../data/{cohort}/test/{csv}.csv'

#%%
# Output path for saved model
model_dir = f'../../models/{model}'
os.makedirs(model_dir, exist_ok=True)
model_path = f'{model_dir}/{label}_{cohort}.joblib'

# === Load features
df_features = pd.read_csv(feature_csv)
feature_cols = [col for col in df_features.columns if col.isdigit()]

# Load splits
df_train = pd.read_csv(train_csv)
df_val   = pd.read_csv(val_csv)
df_test  = pd.read_csv(test_csv)

# Standardize EID column name
for df in [df_train, df_val, df_test]:
    if 'eid__sequence' in df.columns:
        df.rename(columns={'eid__sequence': 'eid'}, inplace=True)

# Merge features with each split
df_train = df_train.merge(df_features, on='eid', how='inner')
df_val   = df_val.merge(df_features, on='eid', how='inner')
df_test  = df_test.merge(df_features, on='eid', how='inner')

# Extract X and y
X_train, y_train = df_train[feature_cols].values, df_train[label].values
X_val,   y_val   = df_val[feature_cols].values,   df_val[label].values
X_test,  y_test  = df_test[feature_cols].values,  df_test[label].values

# === Train regression model
reg = Ridge(alpha=1.0)
reg.fit(X_train, y_train)

# === Save model
joblib.dump(reg, model_path)
print(f"✅ Model saved to: {model_path}")

# === Evaluate
def evaluate(name, X, y):
    y_pred = reg.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"{name} — MAE: {mae:.2f}, R²: {r2:.4f}")

evaluate("✅ Val", X_val, y_val)
evaluate("✅ Test", X_test, y_test)
# %%
from sklearn.linear_model import LinearRegression

def fit_bias_model(y_true, y_pred):
    model = LinearRegression().fit(y_pred.reshape(-1, 1), y_true)
    return model

def apply_bias_model(model, y_pred):
    return model.predict(y_pred.reshape(-1, 1))

def evaluate(name, X, y, reg_model, bias_model=None):
    y_pred = reg_model.predict(X)
    if bias_model is not None:
        y_pred = apply_bias_model(bias_model, y_pred)

    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"{name} — MAE: {mae:.2f}, R²: {r2:.4f}")
    return y_pred


# %%
# Raw predictions
y_val_pred = reg.predict(X_val)
bias_model = fit_bias_model(y_val, y_val_pred)

# Evaluate
evaluate("✅ Val (bias corrected)", X_val, y_val, reg_model=reg, bias_model=bias_model)
evaluate("✅ Test (bias corrected)", X_test, y_test, reg_model=reg, bias_model=bias_model)

# %%
