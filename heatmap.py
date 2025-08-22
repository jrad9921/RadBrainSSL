#%%
# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Core settings
    'cohort': 'oasis',
    'csv_name': 'ad-cn', 
    'training_mode': 'linear',
    'device': 'cuda:1',
    'img_size': 96,
    'n_classes': 2,
    'max_samples': 1000,
    
    # Visualization options
    'mode': 'top_individual',  # 'single', 'average', 'top_individual'
    'top_n': 3,
    
    # Paths (auto-generated)
    'base_path': '/mnt/bulk-neptune/radhika/project',
    'atlas_path': 'atlas_resampled_96.nii.gz'
}
#%%
# ============================================================================
# IMPORTS & SETUP
# ============================================================================

import sys
sys.path.extend(['../dataloaders', '../architectures'])
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import nibabel as nib
from torch.serialization import safe_globals
import monai
import sfcn_cls, head, sfcn_ssl2
from atlas_labels import LABEL_DICT

def setup_paths(config):
    """Generate all paths from config"""
    base = config['base_path']
    cohort = config['cohort']
    csv_name = config['csv_name']
    mode = config['training_mode']
    size = config['img_size']
    
    return {
        'data_csv': f"{base}/data/{cohort}/test/{csv_name}.csv",
        'images_dir': f"{base}/images/{cohort}/npy{size}",
        'model': f"{base}/models/{mode}/{csv_name}_e1000_nNone_b16_lr0.01_im{size}_ssl-ukb.pth",
        'output_dir': f"{base}/vizualizations/heatmaps/{csv_name}/{mode}/{cohort}/",
        'regional_csv': f"{mode}/regional_{cohort}.csv"
    }
#%%
# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_image(path, device):
    """Load and preprocess image"""
    img_data = np.load(path)
    
    # Display version
    img_np = np.clip(img_data, *np.percentile(img_data, (1, 99)))
    img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
    
    # Model input version
    img_t = torch.from_numpy(img_data).float().unsqueeze(0).unsqueeze(0).to(device)
    
    return img_t, img_np

def load_model(path, mode, n_classes, device):
    """Load model based on training mode"""
    models = {
        'sfcn': lambda: sfcn_cls.SFCN(output_dim=n_classes),
        'dense': lambda: monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=2, out_channels=n_classes),
        'linear': lambda: head.ClassifierHeadMLP_(sfcn_ssl2.SFCN(), output_dim=n_classes),
        'ssl-finetuned': lambda: head.ClassifierHeadMLP_(sfcn_ssl2.SFCN(), output_dim=n_classes)
    }
    
    model = models[mode]().to(device)
    
    # Freeze backbone for linear mode
    if mode == 'linear':
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    # Load weights
    with safe_globals([np.dtype, np.core.multiarray.scalar]):
        chkpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(chkpt['model_state_dict'], strict=False)
    
    return model.eval()

def compute_gradcam(model, image):
    """Compute GradCAM attention map"""
    image.requires_grad_(True)
    model.zero_grad()
    
    logits = model(image)
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()
    
    # Backprop through predicted class
    probs[0, pred_class].backward()
    
    # Compute gradcam
    gradcam = (image.grad * image).detach().cpu().numpy()
    gradcam = np.abs(gradcam)
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
    
    return gradcam, pred_class, confidence

def match_files(df, images_dir, cohort):
    """Match dataframe IDs with image files"""
    all_files = [f for f in os.listdir(images_dir) if f.endswith('.npy')]
    
    if cohort == 'ukb':
        df['eid_clean'] = df['eid'].str.replace('_', '').astype(int)
        pics = [f"{images_dir}/{f}" for f in all_files 
                if int(f.replace('.npy', '').replace('_', '')) in df['eid_clean'].tolist()]
    else:
        df['eid'] = df['eid'].astype(str).str.replace('.npy', '').str.strip()
        pics = [f"{images_dir}/{f}" for f in all_files 
                if f.replace('.npy', '') in df['eid'].tolist()]
    
    pic_ids = [os.path.splitext(f.split('/')[-1])[0] for f in pics]
    return pics, pic_ids

def quantify_regions(heatmap, atlas_path, label_dict):
    """Quantify attention per brain region"""
    atlas = nib.load(atlas_path).get_fdata().astype(int)
    
    region_scores = {}
    for region_id in np.unique(atlas):
        if region_id == 0:
            continue
        mask = atlas == region_id
        heat = heatmap[mask].mean()  # Mean instead of sum/size
        region_scores[label_dict.get(region_id, f"Region_{region_id}")] = heat
    
    return region_scores

def save_visualization(heatmap, image, filename, output_dir):
    """Save multi-view heatmap visualization and NIfTI files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save visualization PNG
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    
    views = ['Axial', 'Coronal', 'Sagittal']
    for i, view in enumerate(views):
        for j in range(3):
            # Get slice indices
            if i == 0:    # Axial
                slice_idx = image.shape[2] * (j + 1) // 4
                img_slice = image[:, :, slice_idx]
                heat_slice = heatmap[:, :, slice_idx]
            elif i == 1:  # Coronal  
                slice_idx = image.shape[1] * (j + 1) // 4
                img_slice = image[:, slice_idx, :]
                heat_slice = heatmap[:, slice_idx, :]
            else:         # Sagittal
                slice_idx = image.shape[0] * (j + 1) // 4
                img_slice = image[slice_idx, :, :]
                heat_slice = heatmap[slice_idx, :, :]
            
            axs[i, j].imshow(img_slice, cmap='gray', vmin=0, vmax=255)
            axs[i, j].imshow(heat_slice, cmap='Reds', alpha=0.8, vmin=0, vmax=1)
            axs[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Save NIfTI files
    # Save heatmap as NIfTI
    heatmap_nifti = nib.Nifti1Image(heatmap, affine=np.eye(4))
    nib.save(heatmap_nifti, f"{output_dir}/{filename}_heatmap.nii.gz")
    
    # Save brain image as NIfTI  
    brain_nifti = nib.Nifti1Image(image, affine=np.eye(4))
    nib.save(brain_nifti, f"{output_dir}/{filename}_brain.nii.gz")
    
    print(f"Saved: {filename}.png, {filename}_heatmap.nii.gz, {filename}_brain.nii.gz")
#%%
# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(config):
    """Main processing pipeline"""
    torch.cuda.set_device(config['device'])
    device = torch.device(config['device'])
    paths = setup_paths(config)
    
    print(f"Processing {config['cohort']} - {config['training_mode']} - {config['mode']}")
    
    # Load data
    df = pd.read_csv(paths['data_csv'])
    pics, pic_ids = match_files(df, paths['images_dir'], config['cohort'])
    pics, pic_ids = pics[:config['max_samples']], pic_ids[:config['max_samples']]
    
    print(f"Found {len(pics)} matching files")
    
    # Load model
    model = load_model(paths['model'], config['training_mode'], config['n_classes'], device)
    
    # Process images
    results = []
    all_region_scores = []
    
    for pic_id, pic_path in zip(tqdm(pic_ids), pics):
        image_t, image_np = load_image(pic_path, device)
        gradcam, pred_class, confidence = compute_gradcam(model, image_t)
        
        # Store results
        results.append({
            'heatmap': gradcam[0, 0],
            'image': image_np, 
            'confidence': confidence,
            'pred_class': pred_class,
            'eid': pic_id
        })
        
        # Quantify regions
        region_scores = quantify_regions(gradcam[0, 0], config['atlas_path'], LABEL_DICT)
        region_scores['eid'] = pic_id
        all_region_scores.append(region_scores)
        
        if config['mode'] == 'single':
            break
    
    # Save regional data
    if all_region_scores:
        pd.DataFrame(all_region_scores).set_index('eid').to_csv(paths['regional_csv'])
        print(f"Saved regional data to {paths['regional_csv']}")
    
    # Generate visualizations
    if config['mode'] == 'single':
        result = results[0]
        save_visualization(result['heatmap'], result['image'], 
                         f"single_{result['eid']}", paths['output_dir'])
    
    elif config['mode'] == 'average':
        # Average top results
        top_results = sorted(results, key=lambda x: x['confidence'], reverse=True)[:config['top_n']]
        avg_heatmap = np.mean([r['heatmap'] for r in top_results], axis=0)
        save_visualization(avg_heatmap, results[0]['image'], 
                         "average", paths['output_dir'])
    
    elif config['mode'] == 'top_individual':
        # Save top individual results
        positive_results = [r for r in results if r['pred_class'] == 1]
        top_positive = sorted(positive_results, key=lambda x: x['confidence'], reverse=True)[:config['top_n']]
        
        for i, result in enumerate(top_positive):
            save_visualization(result['heatmap'], result['image'],
                             f"top{i+1}_{result['eid']}_conf{result['confidence']:.3f}", 
                             paths['output_dir'])
    
    print(f"Visualizations saved to {paths['output_dir']}")
    return results
#%%
def analyze_regions(config):
    """Analyze regional attention patterns"""
    paths = setup_paths(config)
    
    try:
        df = pd.read_csv(paths['regional_csv'], index_col='eid')
        region_means = df.mean().sort_values(ascending=False)
        
        # Plot top regions
        top_regions = region_means.head(50)  # Show top 50
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_regions)), top_regions.values)
        plt.yticks(range(len(top_regions)), top_regions.index, fontsize=8)
        plt.xlabel('Mean Attention')
        plt.title(f'Top Brain Regions - {config["cohort"].upper()}')
        plt.tight_layout()
        plt.show()
        
        return region_means
    except FileNotFoundError:
        print("Regional CSV not found. Run main() first.")
        return None
#%%
# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run analysis
    results = main(CONFIG)
    
    # Analyze regions
    regional_analysis = analyze_regions(CONFIG)
# %%
