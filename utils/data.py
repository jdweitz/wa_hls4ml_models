# data.py

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split

def load_and_preprocess_data(
    data_dir, 
    max_samples=None, 
    batch_size=512, 
    train_frac=0.7, 
    val_frac=0.15, 
    seed=42
):
    """Loads, normalizes, and splits the data for the transformer model."""
    data_dir = Path(data_dir)
    feat_np_raw = np.load(data_dir / "combined_features.npy")
    label_np = np.load(data_dir / "combined_labels.npy")

    if max_samples is not None:
        feat_np_raw = feat_np_raw[:max_samples]
        label_np = label_np[:max_samples]

    # Build padding mask (True = padded layer)
    pad_mask_np = np.all(feat_np_raw == -1, axis=-1)  # (N, 18)

    # Collect all non-padded rows into one big matrix for stats
    valid_rows = feat_np_raw[~pad_mask_np]
    means = valid_rows.mean(axis=0)
    stds = valid_rows.std(axis=0)
    stds[stds < 1e-5] = 1.0  # Avoid division by zero

    # Zero-fill original -1’s
    feat_np = np.where(feat_np_raw == -1, 0, feat_np_raw)
    feat_np = (feat_np - means) / stds

    # To torch tensors
    feat = torch.from_numpy(feat_np).float()
    pad_mask = torch.from_numpy(pad_mask_np)
    label = torch.from_numpy(label_np).float()

    dataset = TensorDataset(feat, pad_mask, label)
    N = len(dataset)
    train_size = int(train_frac * N)
    val_size = int(val_frac * N)
    test_size = N - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Also return raw features for advanced plotting (aligned with split indices)
    return (
        train_loader, val_loader, test_loader,
        train_ds, val_ds, test_ds,
        feat_np_raw, label_np
    )

def get_first_non_padded_layer(arr):
    """
    Given arr shape (18, 16), return the first non-padded layer (shape (16,)).
    """
    mask = ~(np.all(arr == -1, axis=-1))
    if np.any(mask):
        return arr[mask][0]
    else:
        return np.zeros(16)
    
## To use:
# from data import load_and_preprocess_data, get_first_non_padded_layer

# # Example usage in run.py
# train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, feat_np_raw, label_np = \
#     load_and_preprocess_data(
#         data_dir="../../../../ddemler/dima_stuff/wa_remake/May_15_processed",
#         max_samples=10000,
#         batch_size=512
#     )

# # Get X_raw_test for plotting:
# test_indices = test_ds.indices
# X_raw_test = feat_np_raw[test_indices]
# X_raw_test_1layer = np.stack([get_first_non_padded_layer(arr) for arr in X_raw_test])