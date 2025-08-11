#%%
import sys
sys.path.append('../dataloaders')
sys.path.append('../architectures')

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataloader_new
import sfcn_cls           # scratch SFCN
import sfcn_ssl2          # backbone only
import head           # RegressorHeadMLP_
import monai

#%%
# ===== Basic Parameters =====
n_folds = 1
training_mode = 'linear'   # 'sfcn', 'linear', 'ssl-finetuned'
column_name = 'age'        # continuous target column
test_cohort = 'dlbs'
img_size = 96

# these should match the training config used to create the checkpoint name
num_epochs = 1000
batch_size = 32
lr = 1e-3
nrows = None

dev = 'cuda:1'
n_channels = 1

# ===== Test Parameters =====
batch_size_test = 16
nrows_test = None
n_boot = 1000              # bootstrap iterations for CIs
rng = np.random.RandomState(42)

# SSL (only needed for linear/ssl-finetuned)
ssl_cohort = 'ukb'
pretrained_model = f'/mnt/bulk-neptune/radhika/project/models/ssl/sfcn/ukb/ukb96/final_model_b16_e1000.pt'  # not used for loading (we load your trained regressor), but needed to build backbone shape if your head uses it

# ===== Paths =====
unique_name = f"{column_name}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_im{img_size}"
model_path = f'/mnt/bulk-neptune/radhika/project/models/{training_mode}/'
tensor_dir = f'/mnt/bulk-neptune/radhika/project/images/{test_cohort}/npy{img_size}'
csv_test   = f'/mnt/bulk-neptune/radhika/project/data/{test_cohort}/demographics.csv'
output_path = f'/mnt/bulk-neptune/radhika/project/scores/{training_mode}/test/{test_cohort}/{column_name}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_im{img_size}_ssl-{ssl_cohort}'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ===== Device / seed =====
if torch.cuda.is_available():
    torch.cuda.set_device(dev)
torch.manual_seed(42)
np.random.seed(42)

#%%
def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def pearson_r(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size < 2:
        return np.nan
    c = np.corrcoef(y_true, y_pred)
    return float(c[0, 1])

def bootstrap_ci(metric_func, y_true, y_pred, n_boot=1000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        vals.append(metric_func(y_true[idx], y_pred[idx]))
    vals = np.sort(vals)
    lo = vals[int((alpha/2)   * len(vals))]
    hi = vals[int((1-alpha/2)* len(vals)) - 1]
    return float(lo), float(hi)

#%%
# run external evaluation
summary_rows = []

for fold in range(1, n_folds + 1):
    print(f'External regression eval for fold {fold}')

    # ----- build model skeleton identical to training -----
    if training_mode == 'sfcn':
        # direct SFCN with 1 output for regression
        model = sfcn_cls.SFCN(output_dim=1).to(dev)

    elif training_mode in ['linear', 'ssl-finetuned']:
        backbone = sfcn_ssl2.SFCN()
        # use your regressor head that mirrors ClassifierHeadMLP_ (expects backbone(x, return_projection=False))
        # make sure you added this to head_cls.py earlier
        if hasattr(head_cls, 'RegressorHeadMLP_'):
            model = head_cls.RegressorHeadMLP_(backbone, output_dim=1).to(dev)
        else:
            raise RuntimeError("head_cls.RegressorHeadMLP_ not found. Please add it as discussed.")

    else:
        raise ValueError(f"Invalid training_mode: {training_mode}")

    # ----- load trained weights -----
    model_file = os.path.join(model_path, f"{unique_name}.pth")
    print(f"Loading {training_mode} model: {model_file}")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    checkpoint = torch.load(model_file, map_location=dev)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        else:
            state = checkpoint
    else:
        state = checkpoint
    model.load_state_dict(state, strict=False)
    print("✓ Model loaded")

    model.eval()

    # ----- dataset -----
    test_dataset = dataloader_new.BrainDataset(
        csv_file=csv_test,
        root_dir=tensor_dir,
        column_name=column_name,
        num_rows=nrows_test,
        num_classes=None,
        task='regression'
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, num_workers=8, drop_last=False)

    # ----- inference -----
    test_eids = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        for eid, images, targets in tqdm(test_loader, desc=f"Testing fold {fold}"):
            test_eids.extend(eid)
            images = images.to(dev)
            targets = targets.float().to(dev).view(-1, 1)
            outputs = model(images)
            y_true.extend(targets.squeeze(1).cpu().tolist())
            y_pred.extend(outputs.squeeze(1).cpu().tolist())

    # ----- metrics -----
    _mae = mae(y_true, y_pred)
    _rmse = rmse(y_true, y_pred)
    _r = pearson_r(y_true, y_pred)

    mae_lo, mae_hi   = bootstrap_ci(mae,   np.array(y_true), np.array(y_pred), n_boot=n_boot, alpha=0.05, rng=rng)
    rmse_lo, rmse_hi = bootstrap_ci(rmse,  np.array(y_true), np.array(y_pred), n_boot=n_boot, alpha=0.05, rng=rng)
    r_lo, r_hi       = bootstrap_ci(pearson_r, np.array(y_true), np.array(y_pred), n_boot=n_boot, alpha=0.05, rng=rng)

    print(f"MAE : {_mae:.4f}  (95% CI: {mae_lo:.4f}–{mae_hi:.4f})")
    print(f"RMSE: {_rmse:.4f}  (95% CI: {rmse_lo:.4f}–{rmse_hi:.4f})")
    print(f"r    : {_r:.4f}    (95% CI: {r_lo:.4f}–{r_hi:.4f})")

    # ----- save per-scan predictions -----
    output_csv = f'{output_path}.csv'
    pd.DataFrame({"eid": test_eids, "label": y_true, "prediction": y_pred}).to_csv(output_csv, index=False)
    print(f'Predictions saved to {output_csv}')

    # ----- fold summary -----
    summary_rows.append({
        "fold": fold,
        "MAE": _mae, "MAE_CI_lower": mae_lo, "MAE_CI_upper": mae_hi,
        "RMSE": _rmse, "RMSE_CI_lower": rmse_lo, "RMSE_CI_upper": rmse_hi,
        "PearsonR": _r, "R_CI_lower": r_lo, "R_CI_upper": r_hi,
        "n": len(y_true)
    })

# save summary
summary_df = pd.DataFrame(summary_rows)
summary_csv = f'{output_path}_summary.csv'
summary_df.to_csv(summary_csv, index=False)
print(f'Summary results saved to {summary_csv}')
#%%
