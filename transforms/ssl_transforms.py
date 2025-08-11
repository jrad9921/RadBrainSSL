
#%%
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, RandZoomd, RandRotated, RandFlipd, RandGaussianNoised,
    RandAdjustContrastd, RandBiasFieldd, CopyItemsd, Compose
)
import numpy as np

#%%
def double_view_transform(img_size=96):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),

        # === View 1: zoomed out + rotated ===
        RandZoomd(
            keys=["image"],
            min_zoom=0.8, max_zoom=0.95,  # zoom OUT
            mode="trilinear",
            align_corners=True,
            keep_size=True,
            prob=1.0
        ),
        RandRotated(
            keys=["image"],
            range_x=np.pi/9, range_y=np.pi/18, range_z=np.pi/9,
            mode='bilinear',
            prob=1.0
        ),
        RandBiasFieldd(keys=["image"], prob=0.3),
        RandAdjustContrastd(keys=["image"], gamma=(0.9, 1.1), prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.3, std=0.02),

        # === View 2: zoomed in + flipped ===
        RandZoomd(
            keys=["image_2"],
            min_zoom=1.3, max_zoom=1.5,  # zoom IN
            mode="trilinear",
            align_corners=True,
            keep_size=True,
            prob=1.0
        ),
        RandFlipd(
            keys=["image_2"],
            spatial_axis=[1],  # coronal flip
            prob=1.0
        ),
        RandBiasFieldd(keys=["image_2"], prob=0.3),
        RandAdjustContrastd(keys=["image_2"], gamma=(0.8, 1.3), prob=0.6),
        RandGaussianNoised(keys=["image_2"], prob=0.4, std=0.04),
    ])
