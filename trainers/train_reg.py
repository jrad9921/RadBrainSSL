#%%
# 0. Imports
import os
import sys
import time
import datetime
import random
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import monai
import matplotlib.pyplot as plt
# project paths
sys.path.append('../dataloaders')
sys.path.append('../architectures')
import dataloader_new
import sfcn_cls, sfcn_ssl2, head

# =============== small fallback regressor head (used if head_cls doesn't expose one) ===============
class _RegressorHeadMLP_(nn.Module):
    """
    Expect backbone(images) -> features or logits.
    If backbone returns class logits, set `take_features=False` and add a pooling.
    If your sfcn_ssl2.SFCN returns features via .forward_features, prefer that path.
    """
    def __init__(self, backbone, in_dim=512, hidden=256, take_features=True):
        super().__init__()
        self.backbone = backbone
        self.take_features = take_features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, 1)
        )

    def forward_features(self, x):
        # Try common feature hooks
        if hasattr(self.backbone, 'forward_features'):
            return self.backbone.forward_features(x)
        out = self.backbone(x)
        return out

    def forward(self, x):
        feats = self.forward_features(x) if self.take_features else self.backbone(x)
        # flatten if needed
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        return self.mlp(feats)

#%%
# 1. Parameters
cohort = 'ukb'
training_mode = 'linear'     # 'sfcn', 'dense', 'linear', 'ssl-finetuned'
column_name = 'age'          # <- continuous target column
csv_name = 'demographics'
task = 'regression'
img_size = 96

# Training params
batch_size = 32
num_epochs = 1000
n_splits = 3
nrows = None
dev = "cuda:1"
lr = 1e-3                   # smaller LR typical for regression heads

seed = 42
best_val_loss = float('inf')
n_channels = 1

# Input Paths
csv_train = f'/mnt/bulk-neptune/radhika/project/data/{cohort}/train/{csv_name}.csv'
unique_name = f"{column_name}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_im{img_size}"
tensor_dir = f'/mnt/bulk-neptune/radhika/project/images/{cohort}/npy{img_size}'

# logging / outputs
save_model = f'/mnt/bulk-neptune/radhika/project/models/{training_mode}'
scores_train = f'/mnt/bulk-neptune/radhika/project/scores/{training_mode}/train/{unique_name}'
scores_val = f'/mnt/bulk-neptune/radhika/project/scores/{training_mode}/val/{unique_name}'
timelog_dir = f'/mnt/bulk-neptune/radhika/project/logs/timelog/{training_mode}/'
trainlog_dir = f'/mnt/bulk-neptune/radhika/project/logs/trainlog/{training_mode}/'
vallog_dir = f'/mnt/bulk-neptune/radhika/project/logs/vallog/{training_mode}/'
metrics_dir = f'/mnt/bulk-neptune/radhika/project/logs/metrics/{training_mode}/'

for p in [scores_train, scores_val, timelog_dir, trainlog_dir, vallog_dir, metrics_dir, save_model]:
    os.makedirs(p, exist_ok=True)

# SSL params
ssl_batch_size = 16
ssl_n_epochs = 1000
pretrained_model = f'/mnt/bulk-neptune/radhika/project/models/ssl/sfcn/ukb/ukb96/final_model_b{ssl_batch_size}_e{ssl_n_epochs}.pt'

# Swin params (only used if you plug in a Swin backbone)
patch_size = [8, 8, 8]
window_size = [16, 16, 16]
num_heads = [3, 6, 12, 24]
depths = [2, 2, 2, 2]
feature_size = 96

# Early stopping
patience = 20

# Device & seeds
if torch.cuda.is_available():
    torch.cuda.set_device(dev)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

#%%
# 2. Data overview
df = pd.read_csv(csv_train)
if nrows is not None:
    df = df.head(nrows)
print(df.head())
print(df[column_name].describe())

#%%
# 3. Dataset / CV
train_dataset = dataloader_new.BrainDataset(
    csv_train, tensor_dir, column_name,
    task='regression', num_classes=None, num_rows=nrows
)

# KFold (no stratification for regression)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

#%%
# 4. Helpers: metrics for regression
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def pearson_r(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    r, _ = pearsonr(y_true, y_pred)
    return float(r)

#%%
# 5. Training
trainlog_file = os.path.join(trainlog_dir, f"{unique_name}.txt")
with open(trainlog_file, "a") as log:
    log.write('Fold,Epoch,TrainLoss(MSE),ValLoss(MSE),ValMAE,ValRMSE,ValPearsonR\n')

total_time = 0.0

for fold, (train_ids, val_ids) in enumerate(kf.split(np.arange(len(train_dataset)))):
    train_subset = Subset(train_dataset, train_ids)
    val_subset   = Subset(train_dataset, val_ids)

    train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True)

    # ===== model per mode =====
    if training_mode == 'sfcn':
        # direct SFCN with 1 output (regression)
        model = sfcn_cls.SFCN(output_dim=1).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr)
        print("Using SFCN (regression)")

    elif training_mode == 'dense':
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=n_channels, out_channels=1).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr)
        print("Using DenseNet121 (regression)")

    elif training_mode in ['linear', 'ssl-finetuned']:
        backbone = sfcn_ssl2.SFCN()
        checkpoint = torch.load(pretrained_model, map_location=dev)
        backbone.load_state_dict(checkpoint['state_dict'], strict=False)

        # prefer a regressor head from your repo if it exists
        RegrHead = getattr(head, 'RegressorHeadMLP_', None)
        if RegrHead is not None:
            model = RegrHead(backbone, output_dim=1).to(dev)
        else:
            # fallback: guess feature dim; adjust if your backbone’s feature dim differs
            model = _RegressorHeadMLP_(backbone, in_dim=512, hidden=256, take_features=True).to(dev)

        if training_mode == 'linear':
            for p in model.backbone.parameters():
                p.requires_grad = False
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr)
            print("Backbone frozen — training only regression head.")
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr)
            print("Finetuning entire model.")
    else:
        raise ValueError(f"Invalid training_mode: {training_mode}")

    print(model)

    # scheduler & loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss().to(dev)

    best_val_loss = float('inf')
    early_stop_counter = 0
    start_time = time.time()

    # holders for best predictions
    best_val_eids, best_val_targets, best_val_preds = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_train = 0.0

        train_eids, train_targets, train_preds = [], [], []

        for eids, images, targets in train_loader:
            images = images.to(dev)
            targets = targets.float().to(dev).view(-1, 1)  # (B,1)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train += loss.item() * images.size(0)

            train_eids += list(eids)
            train_targets += targets.detach().cpu().view(-1).tolist()
            train_preds += outputs.detach().cpu().view(-1).tolist()

        train_loss = running_train / len(train_subset)

        # ====== validation ======
        model.eval()
        running_val = 0.0
        val_eids, val_targets, val_preds = [], [], []

        with torch.no_grad():
            for eids, images, targets in val_loader:
                images = images.to(dev)
                targets = targets.float().to(dev).view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, targets)
                running_val += loss.item() * images.size(0)

                val_eids += list(eids)
                val_targets += targets.detach().cpu().view(-1).tolist()
                val_preds += outputs.detach().cpu().view(-1).tolist()

        val_loss = running_val / len(val_subset)
        val_mae  = mae(np.array(val_targets), np.array(val_preds))
        val_rmse = rmse(np.array(val_targets), np.array(val_preds))
        val_r    = pearson_r(np.array(val_targets), np.array(val_preds))

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Fold {fold+1}] Epoch {epoch+1}/{num_epochs} | "
              f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | "
              f"MAE: {val_mae:.3f} | RMSE: {val_rmse:.3f} | r: {val_r:.3f} | lr: {current_lr:.2e}")

        # logging per epoch
        with open(trainlog_file, "a") as log:
            log.write(f'{fold+1},{epoch+1},{train_loss:.6f},{val_loss:.6f},{val_mae:.6f},{val_rmse:.6f},{val_r:.6f}\n')

        # checkpoint on best val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_val_eids = val_eids
            best_val_targets = val_targets
            best_val_preds = val_preds

            checkpoint = {"epoch": epoch+1,
                          "state_dict": model.state_dict(),
                          "optimizer": optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(save_model, f"{unique_name}.pth"))
            print(f"✓ Saved best model to {save_model}/{unique_name}.pth (val MSE {val_loss:.4f})")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

    # ===== save predictions for this fold =====
    # (train predictions from last epoch, val predictions from best epoch)
    train_df = pd.DataFrame({
        'eid': train_eids,
        'label': train_targets,
        'prediction': train_preds
    })
    val_df = pd.DataFrame({
        'eid': best_val_eids,
        'label': best_val_targets,
        'prediction': best_val_preds
    })

    train_df.to_csv(f'{scores_train}.csv', index=False)
    val_df.to_csv(f'{scores_val}.csv', index=False)
    print("Predictions saved.")

    # log fold summary
    end_time = time.time()
    duration = end_time - start_time
    total_time += duration

    with open(os.path.join(vallog_dir, f"{unique_name}.txt"), "a") as log:
        log.write(f'Fold {fold+1} Best Val MSE: {best_val_loss:.6f}\n')
        log.write(f'Duration: {duration:.2f} sec\n')

    with open(os.path.join(timelog_dir, f"{unique_name}.txt"), "a") as log:
        log.write(f"Fold {fold+1} - Duration: {duration:.2f} s - "
                  f"Start: {datetime.datetime.fromtimestamp(start_time)} - "
                  f"End: {datetime.datetime.fromtimestamp(end_time)} - "
                  f"Params: {sum(p.numel() for p in model.parameters())}\n")

    print(f"---------------- Fold {fold+1} complete ----------------")
    break  # remove this break to run all folds

fold_time = total_time / n_splits
print(f"Mean fold training time (approx): {fold_time:.2f} s")
#%%
