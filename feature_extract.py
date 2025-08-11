#%%
import sys
sys.path.append('../main/architectures')
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import dataloader
import sfcn_ssl2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

#%%
# ==== Config (Just modify column_name below!) ====
#column_name = 'age'  # or 'ad' or 'pd'
img_size = 96
test_cohort = 'dlbs'
model='sfcne'
n_classes = 2
batch_size = 128
nrows_test = None
dev = 'cuda:1'
csv_file = None
# Dynamic disease name from column_name
#disease_name = column_name.upper()

# Paths
pretrained_model = f'/mnt/bulk-neptune/radhika/project/models/ssl/sfcn/ukb/ukb96/final_model_b16_e1000.pt'
tensor_dir = f'/mnt/bulk-neptune/radhika/project/images/{test_cohort}/npy{img_size}'

# Load model
backbone = sfcn_ssl2.SFCN()
checkpoint = torch.load(pretrained_model, map_location=dev)
backbone.load_state_dict(checkpoint['state_dict'], strict=False)
backbone = backbone.to(dev)
backbone.eval()

#%%
# Feature extraction
def extract_ssl_features(images):
    with torch.no_grad():
        features, _ = backbone(images, return_projection=True)  # Only use backbone features (x)
        return features.view(features.size(0), -1)

# DataLoader
# Updated feature extraction code
test_dataset = dataloader.BrainDataset(
    root_dir=tensor_dir
)

# Updated DataLoader
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, drop_last=False)

# Updated extraction loop
all_features, all_eids = [], []
with torch.no_grad():
    for eid, images in tqdm(test_loader):  # Note: only eid and images now
        images = images.to(dev)
        features = extract_ssl_features(images).cpu().numpy()
        all_features.append(features)
        all_eids.extend(eid)

all_features = np.vstack(all_features)
print(f"✅ Extracted backbone features: {all_features.shape}")
#%%
# Stack features
all_features = np.vstack(all_features)

# Create DataFrame with eid as first column
df = pd.DataFrame(all_features)
df.insert(0, 'eid', all_eids)

# Save
feat_dir = f'/mnt/bulk-neptune/radhika/project/features/{test_cohort}/{model}/'
os.makedirs(feat_dir, exist_ok=True)
feat_file = '_features.csv'
df.to_csv(os.path.join(feat_dir, feat_file), index=False)
print(f"💾 Saved features with eid to: {feat_dir}")


#%%
