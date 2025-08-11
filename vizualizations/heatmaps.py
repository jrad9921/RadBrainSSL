#%%
import sys
sys.path.append('../dataloaders')
sys.path.append('../architectures')
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch
import torch.nn as nn
import monai
import sfcn_cls, sfcn_ssl2, head_cls
import torch.nn.functional as F
import nibabel as nib
import pandas as pd
from torch.serialization import safe_globals
#Set cuda
dev = "cuda:0"
torch.cuda.set_device(dev)

#%%
def get_image(path):
    img_data = np.load(path)
    # For the numpy version (display), preserve T1w contrast
    img_np = img_data.copy()
    # Optional: You might want to clip extreme values
    p1, p99 = np.percentile(img_np, (1, 99))
    img_np = np.clip(img_np, p1, p99)
    # Scale to 0-255 while preserving contrast
    img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
    
    # For the tensor version (model input), keep the original normalization
    img_t = torch.from_numpy(img_data).float()
    img_t = torch.stack([img_t]).unsqueeze(0).to(dev)
    
    return img_t, img_np
    


def load_model(path, training_mode, n_classes):

    n_channels = 2
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if training_mode == 'sfcn':
        model = sfcn_cls.SFCN(output_dim=n_classes).to(dev)
        print("Using SFCN")

    elif training_mode == 'dense':
        model = monai.networks.nets.DenseNet121(
            spatial_dims=3,
            in_channels=n_channels,
            out_channels=n_classes
        ).to(dev)
        print("Using DenseNet121")

    elif training_mode in ['linear', 'ssl-finetuned']:
        backbone = sfcn_ssl2.SFCN()
        model = head_cls.ClassifierHeadMLP_(backbone, output_dim=n_classes).to(dev)

        if training_mode == 'linear':
            for param in model.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen — test-time only")
        else:
            for param in model.backbone.parameters():
                param.requires_grad = True
            print("Finetuning model — test-time only")
    else:
        raise ValueError(f"Invalid training_mode: {training_mode}")

    # === Fix for PyTorch 2.6 UnpicklingError ===
    with safe_globals([np.dtype, np.core.multiarray.scalar]):
        chkpt = torch.load(path, map_location='cpu', weights_only=False)

    model.load_state_dict(chkpt['state_dict'], strict=False)
    print(model)
    return model.to(dev)


def compute_attention_maps(model, image):
    model = model.eval()
    image = image.detach()
    image.requires_grad = True
    model.zero_grad()

    # Get model outputs (logits)
    logits = model(image)
    probs = torch.softmax(logits, dim=1)
    
    # Get predicted class and its probability
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()
    
    # Compute gradients w.r.t predicted class
    loss = probs[0, pred_class]
    loss.backward()

    gradcam = (image.grad * image).detach().cpu().numpy()
    gradcam = abs(gradcam)
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())

    return gradcam, pred_class, confidence

def compute_average_heatmap(heatmaps, weights=None):
    if weights is not None:
        # Normalize weights to sum to 1
        weights = np.array(weights) / np.sum(weights)
        # Compute weighted average
        weighted_sum = np.zeros_like(heatmaps[0])
        for hmap, weight in zip(heatmaps, weights):
            weighted_sum += hmap * weight
        return weighted_sum
    else:
        return np.mean(np.array(heatmaps), axis=0)

#%%
def visualize_multi_view_heatmaps(heatmap_data, representative_image, label, img_size, outdir, is_single=False, eid=None):
    os.makedirs(outdir, exist_ok=True)

    # Handle both single heatmap and averaged heatmap
    if is_single:
        heatmap = heatmap_data[0, 0]  # Single heatmap
    else:
        heatmap = heatmap_data[0, 0]  # Averaged heatmap
    
    image = representative_image
    # Normalize heatmap for overlay
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Create figure
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    
    views = ['Axial', 'Coronal', 'Sagittal']
    slices = [
        [image.shape[2] // 4, image.shape[2] // 2, image.shape[2] * 3 // 4],
        [image.shape[1] // 4, image.shape[1] // 2, image.shape[1] * 3 // 4],
        [image.shape[0] // 4, image.shape[0] // 2, image.shape[0] * 3 // 4]
    ]

    for i, view in enumerate(views):
        for j, slice_idx in enumerate(slices[i]):
            if i == 0:  # Axial
                img_slice = representative_image[:, :, slice_idx]
                heatmap_slice = heatmap_normalized[:, :, slice_idx]
            elif i == 1:  # Coronal
                img_slice = representative_image[:, slice_idx, :]
                heatmap_slice = heatmap_normalized[:, slice_idx, :]
            else:  # Sagittal
                img_slice = representative_image[slice_idx, :, :]
                heatmap_slice = heatmap_normalized[slice_idx, :, :]

            # Plot raw T1w image with proper contrast
            axs[i, j].imshow(img_slice, cmap='gray', vmin=0, vmax=255)
            
            # Overlay heatmap with custom colormap and transparency
            im = axs[i, j].imshow(heatmap_slice, 
                                cmap='Reds',  # or 'Reds'
                                alpha=0.8,    # Adjust transparency
                                vmin=0, 
                                vmax=1)
            
            axs[i, j].axis('off')

    # Add colorbar
    #cbar = fig.colorbar(im, ax=axs, location='bottom', fraction=0.05, pad=0.08)
    #cbar.set_label('Attention Intensity', fontsize=20, labelpad=10)
    #cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    
    # Save with appropriate filename including EID if provided
    if eid is not None:
        filename = f"{eid}_{label}.png"
        heatmap_filename = f"{eid}_heatmap.nii.gz"
        brain_filename = f"{eid}_brain.nii.gz"
    elif is_single:
        filename = "single.png"
        heatmap_filename = "single_heatmap.nii.gz"
        brain_filename = "single_brain.nii.gz"
    else:
        filename = "avg.png"
        heatmap_filename = "heatmap.nii.gz"
        brain_filename = "sample.nii.gz"
        
    plt.savefig(os.path.join(outdir, filename), dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

    # Save the heatmap as a NIfTI file
    heatmap_nifti = nib.Nifti1Image(heatmap_normalized, affine=np.eye(4))
    heatmap_out_path = os.path.join(outdir, heatmap_filename)
    nib.save(heatmap_nifti, heatmap_out_path)
    print(f"Heatmap saved to: {heatmap_out_path}")

    # Save the representative brain image as a NIfTI file
    brain_nifti = nib.Nifti1Image(representative_image, np.eye(4))
    brain_out_path = os.path.join(outdir, brain_filename)
    nib.save(brain_nifti, brain_out_path)
    print(f"Brain image saved to: {brain_out_path}")


# %%
# === AAL3 region ID to name dictionary ===
label_dict = {
    1: "Precentral_L", 2: "Precentral_R", 3: "Frontal_Sup_2_L", 4: "Frontal_Sup_2_R",
    5: "Frontal_Mid_2_L", 6: "Frontal_Mid_2_R", 7: "Frontal_Inf_Oper_L", 8: "Frontal_Inf_Oper_R",
    9: "Frontal_Inf_Tri_L", 10: "Frontal_Inf_Tri_R", 11: "Frontal_Inf_Orb_2_L", 12: "Frontal_Inf_Orb_2_R",
    13: "Rolandic_Oper_L", 14: "Rolandic_Oper_R", 15: "Supp_Motor_Area_L", 16: "Supp_Motor_Area_R",
    17: "Olfactory_L", 18: "Olfactory_R", 19: "Frontal_Sup_Medial_L", 20: "Frontal_Sup_Medial_R",
    21: "Frontal_Med_Orb_L", 22: "Frontal_Med_Orb_R", 23: "Rectus_L", 24: "Rectus_R",
    25: "OFCmed_L", 26: "OFCmed_R", 27: "OFCant_L", 28: "OFCant_R", 29: "OFCpost_L", 30: "OFCpost_R",
    31: "OFClat_L", 32: "OFClat_R", 33: "Insula_L", 34: "Insula_R",
    37: "Cingulate_Mid_L", 38: "Cingulate_Mid_R", 39: "Cingulate_Post_L", 40: "Cingulate_Post_R",
    41: "Hippocampus_L", 42: "Hippocampus_R", 43: "ParaHippocampal_L", 44: "ParaHippocampal_R",
    45: "Amygdala_L", 46: "Amygdala_R", 47: "Calcarine_L", 48: "Calcarine_R",
    49: "Cuneus_L", 50: "Cuneus_R", 51: "Lingual_L", 52: "Lingual_R",
    53: "Occipital_Sup_L", 54: "Occipital_Sup_R", 55: "Occipital_Mid_L", 56: "Occipital_Mid_R",
    57: "Occipital_Inf_L", 58: "Occipital_Inf_R", 59: "Fusiform_L", 60: "Fusiform_R",
    61: "Postcentral_L", 62: "Postcentral_R", 63: "Parietal_Sup_L", 64: "Parietal_Sup_R",
    65: "Parietal_Inf_L", 66: "Parietal_Inf_R", 67: "SupraMarginal_L", 68: "SupraMarginal_R",
    69: "Angular_L", 70: "Angular_R", 71: "Precuneus_L", 72: "Precuneus_R",
    73: "Paracentral_Lobule_L", 74: "Paracentral_Lobule_R",
    75: "Caudate_L", 76: "Caudate_R", 77: "Putamen_L", 78: "Putamen_R",
    79: "Pallidum_L", 80: "Pallidum_R",
    83: "Heschl_L", 84: "Heschl_R", 85: "Temporal_Sup_L", 86: "Temporal_Sup_R",
    87: "Temporal_Pole_Sup_L", 88: "Temporal_Pole_Sup_R", 89: "Temporal_Mid_L", 90: "Temporal_Mid_R",
    91: "Temporal_Pole_Mid_L", 92: "Temporal_Pole_Mid_R", 93: "Temporal_Inf_L", 94: "Temporal_Inf_R",
    95: "Cerebellum_Crus1_L", 96: "Cerebellum_Crus1_R", 97: "Cerebellum_Crus2_L", 98: "Cerebellum_Crus2_R",
    99: "Cerebellum_3_L", 100: "Cerebellum_3_R", 101: "Cerebellum_4_5_L", 102: "Cerebellum_4_5_R",
    103: "Cerebellum_6_L", 104: "Cerebellum_6_R", 105: "Cerebellum_7b_L", 106: "Cerebellum_7b_R",
    107: "Cerebellum_8_L", 108: "Cerebellum_8_R", 109: "Cerebellum_9_L", 110: "Cerebellum_9_R",
    111: "Cerebellum_10_L", 112: "Cerebellum_10_R",
    113: "Vermis_1_2", 114: "Vermis_3", 115: "Vermis_4_5", 116: "Vermis_6", 117: "Vermis_7", 118: "Vermis_8", 119: "Vermis_9", 120: "Vermis_10",
    121: "Thal_AV_L", 122: "Thal_AV_R", 123: "Thal_LP_L", 124: "Thal_LP_R",
    125: "Thal_VA_L", 126: "Thal_VA_R", 127: "Thal_VL_L", 128: "Thal_VL_R",
    129: "Thal_VPL_L", 130: "Thal_VPL_R", 131: "Thal_IL_L", 132: "Thal_IL_R",
    133: "Thal_Re_L", 134: "Thal_Re_R", 135: "Thal_MDm_L", 136: "Thal_MDm_R",
    137: "Thal_MDl_L", 138: "Thal_MDl_R", 139: "Thal_LGN_L", 140: "Thal_LGN_R",
    141: "Thal_MGN_L", 142: "Thal_MGN_R", 143: "Thal_PuI_L", 144: "Thal_PuI_R",
    145: "Thal_PuM_L", 146: "Thal_PuM_R", 147: "Thal_PuA_L", 148: "Thal_PuA_R",
    149: "Thal_PuL_L", 150: "Thal_PuL_R",
    151: "ACC_sub_L", 152: "ACC_sub_R", 153: "ACC_pre_L", 154: "ACC_pre_R",
    155: "ACC_sup_L", 156: "ACC_sup_R", 157: "N_Acc_L", 158: "N_Acc_R",
    159: "VTA_L", 160: "VTA_R", 161: "SN_pc_L", 162: "SN_pc_R", 163: "SN_pr_L", 164: "SN_pr_R",
    165: "Red_N_L", 166: "Red_N_R", 167: "LC_L", 168: "LC_R", 169: "Raphe_D", 170: "Raphe_M"
}
#%%
def quantify_heatmap_regions(heatmap, atlas_path, label_dict, normalize_by_size=True, top_n=None):
    atlas = nib.load(atlas_path).get_fdata().astype(int)
    assert atlas.shape == heatmap.shape

    region_values = {}
    for region_id in np.unique(atlas):
        if region_id == 0:
            continue
        mask = atlas == region_id
        heat = heatmap[mask].sum()
        if normalize_by_size:
            heat /= mask.sum()
        region_name = label_dict.get(region_id, f"Region_{region_id}")
        region_values[region_name] = heat

    if top_n:
        top_items = sorted(region_values.items(), key=lambda x: x[1], reverse=True)[:top_n]
        regions, values = zip(*top_items)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.barh(regions[::-1], values[::-1])
        plt.xlabel("Mean Heat per Voxel")
        plt.title("Top Brain Regions Focused by Model")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    return region_values

#%%
training_mode = 'linear'
def main(cohort='ppmi', single_heatmap=False, top_individual=False, top_n_individual=3):
    import os
    import pandas as pd
    from tqdm import tqdm

    # --- Parameters ---
    column_name = 'ad'
    csv_name = 'ad-cn'
    training_mode = 'linear'
    img_size = 96
    n_classes = 2
    N = 1000
    top_n = 3
    ssl_batch_size = 16
    ssl_n_epochs = 1000
    
    # --- Paths ---
    df = pd.read_csv(f"/mnt/bulk-neptune/radhika/project/data/{cohort}/test/{csv_name}.csv")
    root_dir = f"/mnt/bulk-neptune/radhika/project/images/{cohort}/npy{img_size}"
    model_path = f"/mnt/bulk-neptune/radhika/project/models/{training_mode}/{csv_name}_e1000_nNone_b16_lr0.01_im{img_size}_ssl-ukb.pth"
    outdir = f"/mnt/bulk-neptune/radhika/project/vizualizations/heatmaps/{csv_name}/{training_mode}/{cohort}/"

    # Match files
    if cohort == 'ukb':
        df['eid_clean'] = df['eid'].str.replace('_', '').astype(int)
        ids = df['eid_clean'].tolist()
        all_files = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        pics = [f"{root_dir}/{f}" for f in all_files if int(f.replace('.npy', '').replace('_', '')) in ids]
    else:
        df['eid'] = df['eid'].astype(str).str.replace('.npy', '').str.strip()
        ids = df['eid'].tolist()
        all_files = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        pics = [f"{root_dir}/{f}" for f in all_files if f.replace('.npy', '') in ids]

    print(f"[INFO] Found {len(pics)} matching files out of {len(df)} EIDs")

    pic_ids = [os.path.splitext(f.split('/')[-1])[0] for f in pics][:N]
    pics = pics[:N]

    # --- Load model ---
    model = load_model(model_path, training_mode, n_classes)
    representative_image = None
    all_results = []
    all_region_scores = []

    # --- Loop over images ---
    for id, pic in zip(tqdm(pic_ids), pics):
        image_t, image_np = get_image(pic)
        if representative_image is None:
            representative_image = image_np

        if cohort == 'ukb':
            id_clean = int(id.replace('_', ''))
            matching_row = df[df['eid_clean'] == id_clean]
        else:
            matching_row = df[df['eid'] == id]

        if matching_row.empty:
            print(f"[!] No match found for ID: {id}")
            continue

        grads, pred_class, confidence = compute_attention_maps(model, image_t)

        # --- NEW: Quantify attention per region and store ---
        region_scores = quantify_heatmap_regions(
            heatmap=grads[0, 0],
            atlas_path="atlas_resampled_96.nii.gz",
            label_dict=label_dict,
            normalize_by_size=True,
            top_n=None
        )
        region_scores["eid"] = id
        all_region_scores.append(region_scores)

        all_results.append({
            'heatmap': grads,
            'confidence': confidence,
            'image': image_np,
            'eid': id,
            'pred_class': pred_class
        })

        if single_heatmap:
            print(f"[INFO] Using single heatmap from ID: {id} with confidence: {confidence:.4f}")
            break

    # --- Save regional attention pivoted CSV ---
    if all_region_scores:
        df_attn = pd.DataFrame(all_region_scores)
        df_attn = df_attn.set_index("eid")
        df_attn.to_csv(f"regional_{cohort}.csv")
        print(f"✅ Saved pivoted regional attention CSV to: regional_{cohort}.csv")

    # --- Visualization ---
    if all_results:
        if top_individual:
            positive_results = [r for r in all_results if r['pred_class'] == 1]
            if not positive_results:
                print("[WARNING] No positive predictions found!")
                return
            top_positive = sorted(positive_results, key=lambda x: x['confidence'], reverse=True)[:top_n_individual]
            print(f"[INFO] Generating heatmaps for top {len(top_positive)} positive predictions:")
            for i, result in enumerate(top_positive):
                visualize_multi_view_heatmaps(
                    result['heatmap'],
                    result['image'],
                    f"rank{i+1}_conf{result['confidence']:.3f}",
                    img_size,
                    outdir,
                    is_single=True,
                    eid=result['eid']
                )
        elif single_heatmap:
            single_result = all_results[0]
            visualize_multi_view_heatmaps(
                single_result['heatmap'],
                single_result['image'],
                'single',
                img_size,
                outdir,
                is_single=True,
                eid=single_result['eid']
            )
        else:
            top_results = sorted(all_results, key=lambda x: x['confidence'], reverse=True)[:top_n]
            top_heatmaps = [r['heatmap'] for r in top_results]
            avg_heatmap = compute_average_heatmap(top_heatmaps)
            visualize_multi_view_heatmaps(
                avg_heatmap,
                representative_image,
                'top',
                img_size,
                outdir,
                is_single=False
            )
    else:
        print("[WARNING] No heatmaps were computed.")


#%%
if __name__ == "__main__":
    # To get individual heatmaps for top 3 positive cases
    main('adni1', top_individual=True, top_n_individual=3)
    
    # Other options:
    # main('ppmi', single_heatmap=True)  # For single heatmap
    # main('ppmi', single_heatmap=False)  # For averaged heatmap (original behavior)
# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the pivoted CSV
df = pd.read_csv(f"{training_mode}/regional_4rtni.csv")
cohort = '4rtni'
# Set 'eid' as index
df = df.set_index("eid")

# Compute mean attention across all subjects
region_means = df.mean().sort_values(ascending=False)

# Optional: top N regions
top_n = 1000
top_regions = region_means.head(top_n)
plt.figure(figsize=(10, 30))
plt.barh(top_regions.index[::-1], top_regions.values[::-1])
plt.xlabel("Mean Attention", fontsize=14)
plt.title(f"Top {top_n} Brain Regions Attended by Model ({cohort})", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# %%
