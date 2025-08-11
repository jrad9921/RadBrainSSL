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
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import dataloader_new
import sfcn_cls  # scratch
import sfcn_ssl2  # backbone only
import head_cls  # classification head
import monai
#%%
# Basic Parameters
n_folds = 1
training_mode = 'sfcn'  # 'sfcn', 'linear', 'ssl-finetuned'
column_name = 'sex'
test_cohort = 'dlbs'
n_classes = 2
img_size = 96
gender = 'f'
sub = 'sub5p'
csv_name_train = f'demographics'
csv_name_test = f'demographics'
dev = 'cuda:1'
#Training Parameters
batch_size = 32
num_epochs = 1000
n_splits = 3
nrows = None
n_classes = 2
lr = 0.01
n_channels=1

#Test Parameters
batch_size_test = 16
nrows_test = None

# ssl parameters
ssl_cohort= 'ukb'

#Paths
unique_name = f"{column_name}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_im{img_size}"
model_path = f'/mnt/bulk-neptune/radhika/project/models/{training_mode}/'
tensor_dir = f'/mnt/bulk-neptune/radhika/project/images/{test_cohort}/npy{img_size}'
csv_test = f'/mnt/bulk-neptune/radhika/project/data/{test_cohort}/{csv_name_test}.csv'
output_path = f'/mnt/bulk-neptune/radhika/project/scores/{training_mode}/test/{test_cohort}/{csv_name_test}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_im{img_size}_ssl-{ssl_cohort}'
# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.set_device(dev)

torch.manual_seed(42)
#%%
auroc_list = []
auroc_list = []
for fold in range(1, n_folds + 1):
    print(f'External validation for fold {fold}')

    if training_mode == 'sfcn':
        model = sfcn_cls.SFCN(output_dim=n_classes).to(dev)
    elif training_mode in ['linear', 'ssl-finetuned']:
        # Reconstruct the full model architecture
        backbone = sfcn_ssl2.SFCN()
        model = head_cls.ClassifierHeadMLP_(backbone, output_dim=n_classes).to(dev)    
    else:
        raise ValueError(f"Invalid training_mode: {training_mode}")   
    
    # Load the trained model
    model_file = os.path.join(model_path, f"{unique_name}.pth")
    print(f"Loading {training_mode} model: {model_file}")
    
    # Check if model file exists
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    try:
        checkpoint = torch.load(model_file, map_location=dev, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume the checkpoint is the state dict itself
                model.load_state_dict(checkpoint)
        else:
            # If checkpoint is directly the state dict
            model.load_state_dict(checkpoint)
        
        print(f"Successfully loaded {training_mode} model")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
        
    model.eval()
    
    # Test dataset
    test_dataset = dataloader_new.BrainDataset(
        csv_file=csv_test, 
        root_dir=tensor_dir, 
        column_name=column_name, 
        num_rows=nrows_test, 
        num_classes=n_classes, 
        task='classification'
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, num_workers=8, drop_last=False)

    test_outputs_binary = []
    test_labels = []
    test_eids = []

    with torch.no_grad():
        for eid, images, labels in tqdm(test_loader, desc=f"Testing fold {fold}"):
            test_eids.extend(eid)
            images = images.to(dev)
            labels = labels.float().to(dev)
            binary_labels = labels[:, 1]
            test_labels.extend(binary_labels.tolist())

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            binary_outputs = probs[:, 1]
            test_outputs_binary.extend(binary_outputs.tolist())

    auroc = roc_auc_score(test_labels, test_outputs_binary)
    auprc = average_precision_score(test_labels, test_outputs_binary)

    # Compute 95% CI for AUROC using bootstrapping
    bootstrapped_auroc = []
    bootstrapped_auprc = []
    rng = np.random.RandomState(42)
    for _ in range(1000):
        indices = rng.randint(0, len(test_labels), len(test_labels))
        y_true_sample = np.array(test_labels)[indices]
        y_pred_sample = np.array(test_outputs_binary)[indices]

        if len(np.unique(y_true_sample)) < 2:
            continue  # Skip if only one class is present in the sample

        bootstrapped_auroc.append(roc_auc_score(y_true_sample, y_pred_sample))
        bootstrapped_auprc.append(average_precision_score(y_true_sample, y_pred_sample))

    # AUROC CI
    sorted_auroc = np.sort(bootstrapped_auroc)
    ci_auroc_lower = sorted_auroc[int(0.025 * len(sorted_auroc))]
    ci_auroc_upper = sorted_auroc[int(0.975 * len(sorted_auroc))]

    # AUPRC CI
    sorted_auprc = np.sort(bootstrapped_auprc)
    ci_auprc_lower = sorted_auprc[int(0.025 * len(sorted_auprc))]
    ci_auprc_upper = sorted_auprc[int(0.975 * len(sorted_auprc))]

    print(f'AUROC for fold {fold}: {auroc:.4f} (95% CI: {ci_auroc_lower:.4f}–{ci_auroc_upper:.4f})')
    print(f'AUPRC for fold {fold}: {auprc:.4f} (95% CI: {ci_auprc_lower:.4f}–{ci_auprc_upper:.4f})')

    auroc_list.append({
        'fold': fold,
        'AUROC': auroc,
        'AUROC_CI_lower': ci_auroc_lower,
        'AUROC_CI_upper': ci_auroc_upper,
        'AUPRC': auprc,
        'AUPRC_CI_lower': ci_auprc_lower,
        'AUPRC_CI_upper': ci_auprc_upper
    })

    output_csv = f'{output_path}.csv'
    pd.DataFrame(data={"eid": test_eids, "label": test_labels, "prediction": test_outputs_binary}).to_csv(output_csv, index=False)
    print(f'Predictions saved to {output_csv}')

# Save summary results
summary_df = pd.DataFrame(auroc_list)
summary_csv = f'{output_path}_summary.csv'
summary_df.to_csv(summary_csv, index=False)
print(f'Summary results saved to {summary_csv}')

# %%
