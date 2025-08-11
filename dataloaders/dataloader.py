import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class BrainDataset(Dataset):
    def __init__(self, root_dir, num_rows=None, transform=None):
        """
        Dataset for feature extraction from brain .npy files (SSL-style, no labels)

        Args:
            root_dir (str): Directory containing .npy files
            csv_file (str, optional): Optional CSV file with 'eid' column
            num_rows (int, optional): Limit number of samples (for debugging)
            transform (callable, optional): Optional transform to apply to the images
        """
        self.root_dir = root_dir
        self.transform = transform

        self.eids = sorted([f.replace('.npy', '') for f in os.listdir(root_dir) if f.endswith('.npy')])
        if num_rows is not None:
            self.eids = self.eids[:num_rows]
        print(f"📁 Loaded {len(self.eids)} .npy files from folder.")

    def __len__(self):
        return len(self.eids)

    def __getitem__(self, index):
        eid = self.eids[index]
        npy_path = os.path.join(self.root_dir, f"{eid}.npy")
        img = np.load(npy_path).astype(np.float32)
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # Shape: (1, D, H, W)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return eid, img_tensor
