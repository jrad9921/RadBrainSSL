#%%
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib  # for saving model

# === Config ===
label = 'label'  # or 'ftd', 'pd'
cohort = 'mspaths'
model = 'sfcne'
csv = 'ms-cn'
feature_csv = f'../../features/{cohort}/{model}/_features.csv'
train_csv = f'../../data/{cohort}/train/{csv}.csv'
test_csv  = f'../../data/{cohort}/test/{csv}.csv'
#%%
# Output path for saved model
model_dir = f'../../models/{model}'
os.makedirs(model_dir, exist_ok=True)
model_path = f'{model_dir}/{csv}-{cohort}.joblib'

# === Load features
df_features = pd.read_csv(feature_csv)
feature_cols = [col for col in df_features.columns if col.isdigit()]

# Load splits
df_train = pd.read_csv(train_csv)
df_test  = pd.read_csv(test_csv)

# Standardize EID column name
for df in [df_train, df_test]:
    if 'eid__sequence' in df.columns:
        df.rename(columns={'eid__sequence': 'eid'}, inplace=True)

# Merge features with each split
df_train = df_train.merge(df_features, on='eid', how='inner')
df_test  = df_test.merge(df_features, on='eid', how='inner')

# Extract X and y
X_train, y_train = df_train[feature_cols].values, df_train[label].values
X_test,  y_test  = df_test[feature_cols].values,  df_test[label].values

# === Train linear classifier
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)

# === Save model
joblib.dump(clf, model_path)
print(f"✅ Model saved to: {model_path}")

# === Evaluate
def evaluate(name, X, y):
    y_pred = clf.predict_proba(X)[:, 1]
    auroc = roc_auc_score(y, y_pred)
    acc = accuracy_score(y, y_pred > 0.5)
    print(f"{name} — AUROC: {auroc:.4f}, ACC: {acc:.4f}")

evaluate("✅ Test", X_test, y_test)

# %%

