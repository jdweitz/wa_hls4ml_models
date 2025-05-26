import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm


class FPGAGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for FPGA resource utilization and latency estimation.
    Each sample is a neural network represented as a graph, where nodes are layers
    and edges represent the flow of data between layers.
    """
    def __init__(self, features_path, labels_path, transform=None, pre_transform=None, stats=None):
        super(FPGAGraphDataset, self).__init__(None, transform, pre_transform) # root=None as we load from numpy
        print(f"Loading features from: {features_path}")
        self.features_np = np.load(features_path) # commented out for now
        # self.features_np = np.load(features_path)[:10000]
        print(f"Loading labels from: {labels_path}")
        self.labels_np = np.load(labels_path) # commented out for now
        # self.labels_np = np.load(labels_path)[:10000]

        # Feature indices from the dataset
        self.feature_indices_map = {
            "d_in1": 0, "d_in2": 1, "d_in3": 2,
            "d_out1": 3, "d_out2": 4, "d_out3": 5,
            "prec": 6, "rf": 7, "strategy": 8,
            "layer_type": 9, "activation_type": 10,
            "filters": 11, "kernel_size": 12,
            "stride": 13, "padding": 14, "pooling": 15
        }

        # Define numerical feature keys
        self.numerical_feature_keys = [
            "d_in1", "d_in2", "d_in3", "d_out1", "d_out2", "d_out3",
            "prec", "rf", "filters", "kernel_size", "stride", "pooling"
        ]
        # Get integer indices for numerical features
        self.numerical_feature_indices = torch.tensor(
            [self.feature_indices_map[k] for k in self.numerical_feature_keys], dtype=torch.long
        )
        self.num_numerical_features = len(self.numerical_feature_keys)

        # Define mappings for categorical features 
        self.layer_type_mapping_provided = { 
            'activation': 0, 'QDense': 1, 'QConv1D': 2, 'QConv2D': 3,
            'QSeparableConv1D': 4, 'QSeparableConv2D': 5, 'QDepthwiseConv1D': 6, 'QDepthwiseConv2D': 7,
            'Flatten': 8, 'MaxPooling1D': 9, 'MaxPooling2D': 9,
            'AveragePooling1D': 10, 'AveragePooling2D': 10
        }
        self.activation_mapping_provided = { 
            'NA': 0, 'linear': 1, 'quantized_relu': 2, 'quantized_tanh': 3,
            'quantized_sigmoid': 4, 'quantized_softmax': 5
        }
        self.padding_mapping_provided = {
            'NA': 0, 'valid': 1, 'same': 2,
        }

        # Get the number of unique categorical values
        self.num_layer_types = len(set(self.layer_type_mapping_provided.values())) # Should be 11 (0-10)
        self.num_activation_types = len(self.activation_mapping_provided)        # Should be 6 (0-5)
        self.num_padding_types = len(self.padding_mapping_provided)   # Should be 3 (0-2)

        # Specific indices for easier access
        self.strategy_idx = self.feature_indices_map["strategy"]
        self.layer_type_idx = self.feature_indices_map["layer_type"]
        self.activation_type_idx = self.feature_indices_map["activation_type"]
        self.padding_idx = self.feature_indices_map["padding"]

        # Calculate and store normalization statistics (mean and std) for both features and labels
        if stats:
            self.feature_means, self.feature_stds, self.label_means, self.label_stds = stats
            print("Using provided normalization statistics.")
        else:
            print("Calculating normalization statistics from the dataset...")
            self.feature_means, self.feature_stds, self.label_means, self.label_stds = self._calculate_normalization_stats()
            print(f"Computed Feature Means: {self.feature_means}")
            print(f"Computed Feature Stds: {self.feature_stds}")
            print(f"Computed Label Means: {self.label_means}")
            print(f"Computed Label Stds: {self.label_stds}")

        # Total node feature dimension after processing
        self.node_feature_dim = (self.num_numerical_features +
                                 self.num_layer_types +
                                 self.num_activation_types +
                                 self.num_padding_types)
        print(f"Processed node feature dimension will be: {self.node_feature_dim}")


    def _calculate_normalization_stats(self):
        """
        Calculates mean and std for numerical features and labels across the entire dataset
        """
        # Accumulator for numerical values for each feature
        numerical_values_accumulator = [[] for _ in range(self.num_numerical_features)]
        
        # Accumulator for labels
        label_values_accumulator = []

        print("Iterating through dataset to compute normalization stats...")
        for model_idx in tqdm(range(self.features_np.shape[0]), desc="Calculating Stats"): #iterating through all models
            model_features = self.features_np[model_idx] # (max_layers, num_raw_features)
            model_labels = self.labels_np[model_idx] # (num_targets,)
            
            # Collect labels
            label_values_accumulator.append(model_labels)
            
            for layer_idx in range(model_features.shape[0]):
                layer_data = model_features[layer_idx]

                # Check if the layer is a padded row (all -1s)
                if np.all(layer_data == -1):
                    continue # Skip padded layers

                # This layer is valid, extract numerical features
                for i, feature_col_idx in enumerate(self.numerical_feature_indices.tolist()):
                    val = layer_data[feature_col_idx]
                    # Only accumulate valid numerical values for stats calculation
                    if val != -1:
                        numerical_values_accumulator[i].append(val)

        # Calculate feature statistics
        feature_means = []
        feature_stds = []
        for i, vals in enumerate(numerical_values_accumulator):
            if vals: # if list is not empty
                feature_means.append(np.mean(vals))
                std_val = np.std(vals)
                feature_stds.append(std_val if std_val > 1e-5 else 1.0) # Avoid division by zero or very small std
            else:
                num_feat_original_idx = self.numerical_feature_indices[i].item()
                print(f"Warning: No valid data points found for numerical feature '{self.numerical_feature_keys[i]}' (original index {num_feat_original_idx}). Using default mean=0, std=1.")
                feature_means.append(0.0)
                feature_stds.append(1.0)

        # Calculate label statistics
        all_labels = np.vstack(label_values_accumulator)  # (num_samples, num_targets)
        label_means = np.mean(all_labels, axis=0)
        label_stds = np.std(all_labels, axis=0)
        
        # Avoid division by zero for label stds
        label_stds = np.where(label_stds > 1e-5, label_stds, 1.0)

        return (torch.tensor(feature_means, dtype=torch.float), 
                torch.tensor(feature_stds, dtype=torch.float),
                torch.tensor(label_means, dtype=torch.float),
                torch.tensor(label_stds, dtype=torch.float))

    def _one_hot_encode(self, raw_value, num_classes):
        """
        Helper function for one-hot encoding.
        raw_value: The integer code for the category.
        num_classes: The total number of possible classes for this category.
        """
        try:
            # Values from numpy array might be floats (e.g., 3.0) but represent integers
            value = int(raw_value)
        except ValueError:
            print(f"Warning: Could not convert raw_value '{raw_value}' to int for OHE. Returning zero vector.")
            return torch.zeros(num_classes, dtype=torch.float)

        tensor = torch.zeros(num_classes, dtype=torch.float)

        if 0 <= value < num_classes:
            tensor[value] = 1.0
        else:
            # This case means the integer value from data is outside the expected range [0, num_classes-1]
            # For example, layer_type = 15, but self.num_layer_types = 11. Or layer_type = -1.
            # This could indicate an issue with data or that -1 padding wasn't fully caught for categorical features.
            print(f"Warning: Integer value {value} (from raw '{raw_value}') is out of bounds for OHE with {num_classes} classes. Returning zero vector.")
        return tensor

    def len(self):
        """Returns the number of graphs in the dataset."""
        return self.features_np.shape[0]
    
    def save_normalization_stats(self, filepath):
        """
        Save the calculated normalization statistics (means and stds) to a file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save means and stds as a dictionary
        stats_dict = {
            'feature_means': self.feature_means.numpy(),
            'feature_stds': self.feature_stds.numpy(),
            'label_means': self.label_means.numpy(),
            'label_stds': self.label_stds.numpy(),
            'feature_keys': self.numerical_feature_keys  # Save keys for verification
        }
        
        # Save to file
        np.save(filepath, stats_dict)
        print(f"Normalization statistics saved to {filepath}")
    

    @staticmethod
    def load_normalization_stats(filepath):
        """
        Load normalization statistics from a file
        
        Args:
            filepath: Path to the saved statistics file
            
        Returns:
            tuple: (feature_means, feature_stds, label_means, label_stds) as torch tensors
        """
        stats_dict = np.load(filepath, allow_pickle=True).item()
        
        # Convert numpy arrays to torch tensors
        feature_means = torch.tensor(stats_dict['feature_means'], dtype=torch.float)
        feature_stds = torch.tensor(stats_dict['feature_stds'], dtype=torch.float)
        label_means = torch.tensor(stats_dict['label_means'], dtype=torch.float)
        label_stds = torch.tensor(stats_dict['label_stds'], dtype=torch.float)
        
        #Print the loaded feature keys for verification
        print(f"Loaded statistics for features: {stats_dict['feature_keys']}")
        
        return feature_means, feature_stds, label_means, label_stds
    
    def denormalize_labels(self, normalized_labels):
        """
        Denormalize labels using the stored statistics
        """
        return normalized_labels * self.label_stds + self.label_means

    def get(self, idx):
        """
        Processes and returns a single graph data object for the given index.
        """
        model_features_raw = self.features_np[idx]  # (max_layers, num_raw_features)
        model_labels = self.labels_np[idx]        # (num_targets,)

        node_features_list = []
        valid_layer_original_indices = [] # To keep track of original indices if needed

        for i in range(model_features_raw.shape[0]):
            layer_data_raw = model_features_raw[i]
            # If all features in the layer are -1, it's a padded layer row
            if np.all(layer_data_raw == -1):
                continue # Skip this padded layer slot

            valid_layer_original_indices.append(i) # Store original index of this valid layer
            
            # 1. Numerical Features
            # Extract raw numerical features for this layer
            current_numerical_feats_raw = torch.from_numpy(
                layer_data_raw[self.numerical_feature_indices.numpy()]
            ).float()

            # Normalize. Handle -1s if they mean "not applicable" for a specific numerical feature
            # by treating them as 0 for normalization purposes: (0 - mean) / std.
            numerical_feats_processed = torch.zeros_like(current_numerical_feats_raw)
            for j in range(self.num_numerical_features):
                val = current_numerical_feats_raw[j]
                if val == -1.0: # If -1 signifies 'not applicable' for this specific numerical feature
                    # Normalize 0.0 instead. This means "not applicable" maps to a specific value relative to the distribution.
                    numerical_feats_processed[j] = (0.0 - self.feature_means[j]) / self.feature_stds[j]
                else:
                    numerical_feats_processed[j] = (val - self.feature_means[j]) / self.feature_stds[j]
            
            # 2. Categorical Features (One-Hot Encoded)
            # Values are assumed to be integer codes already present in the numpy array.
            layer_type_val = layer_data_raw[self.layer_type_idx]
            activation_type_val = layer_data_raw[self.activation_type_idx]
            padding_val = layer_data_raw[self.padding_idx]

            ohe_layer_type = self._one_hot_encode(layer_type_val, self.num_layer_types)
            ohe_activation_type = self._one_hot_encode(activation_type_val, self.num_activation_types)
            ohe_padding = self._one_hot_encode(padding_val, self.num_padding_types)

            # Concatenate all processed features for the current node
            current_node_features = torch.cat([
                numerical_feats_processed,
                ohe_layer_type,
                ohe_activation_type,
                ohe_padding
            ], dim=0)
            node_features_list.append(current_node_features)

        
        x = torch.stack(node_features_list) # Stack all valid node feature tensors
        # Global graph feature: strategy (binary 0 or 1)
        # Taken from the first valid layer found. Assumed to be consistent for the graph.
        first_valid_layer_data = model_features_raw[valid_layer_original_indices[0]]
        strategy_val = int(first_valid_layer_data[self.strategy_idx])

        # Edges: Layer i -> Layer i+1 for the sequence of valid layers found
        num_valid_nodes = x.shape[0]
        if num_valid_nodes > 1:
            # Edges are sequential between the valid nodes identified
            source_nodes = torch.arange(0, num_valid_nodes - 1, dtype=torch.long)
            target_nodes = torch.arange(1, num_valid_nodes, dtype=torch.long)
            edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        else:
            # No edges if 0 or 1 node
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Labels (graph-level targets)
        labels_tensor = torch.tensor(model_labels, dtype=torch.float)
        normalized_labels = (labels_tensor - self.label_means) / self.label_stds
        y = normalized_labels.unsqueeze(0) # Ensures shape [1, num_targets]

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        data.strategy = torch.tensor([strategy_val], dtype=torch.float).unsqueeze(1)

        return data
    
    def get_transformer(self, idx):
        """
        Returns features, pad_mask, and normalized labels for a transformer model.
        - x: shape (num_layers, node_feature_dim)
        - pad_mask: shape (num_layers,) (True if padded)
        - y: shape (num_targets,)
        """
        model_features_raw = self.features_np[idx]  # (max_layers, num_raw_features)
        model_labels = self.labels_np[idx]          # (num_targets,)

        node_features_list = []
        pad_mask = []
        for i in range(model_features_raw.shape[0]):
            layer_data_raw = model_features_raw[i]
            # Check for padded row (all -1s)
            if np.all(layer_data_raw == -1):
                # Create a dummy feature (all zeros) for the pad
                node_features_list.append(torch.zeros(self.node_feature_dim, dtype=torch.float))
                pad_mask.append(True)
                continue
            pad_mask.append(False)

            # Numerical features
            current_numerical_feats_raw = torch.from_numpy(
                layer_data_raw[self.numerical_feature_indices.numpy()]
            ).float()
            numerical_feats_processed = torch.zeros_like(current_numerical_feats_raw)
            for j in range(self.num_numerical_features):
                val = current_numerical_feats_raw[j]
                if val == -1.0:
                    numerical_feats_processed[j] = (0.0 - self.feature_means[j]) / self.feature_stds[j]
                else:
                    numerical_feats_processed[j] = (val - self.feature_means[j]) / self.feature_stds[j]
            
            # Categorical one-hot
            layer_type_val = layer_data_raw[self.layer_type_idx]
            activation_type_val = layer_data_raw[self.activation_type_idx]
            padding_val = layer_data_raw[self.padding_idx]
            ohe_layer_type = self._one_hot_encode(layer_type_val, self.num_layer_types)
            ohe_activation_type = self._one_hot_encode(activation_type_val, self.num_activation_types)
            ohe_padding = self._one_hot_encode(padding_val, self.num_padding_types)

            current_node_features = torch.cat([
                numerical_feats_processed,
                ohe_layer_type,
                ohe_activation_type,
                ohe_padding
            ], dim=0)
            node_features_list.append(current_node_features)
        
        x = torch.stack(node_features_list)                   # (num_layers, node_feature_dim)
        pad_mask_tensor = torch.tensor(pad_mask, dtype=torch.bool)  # (num_layers,)
        labels_tensor = torch.tensor(model_labels, dtype=torch.float)
        normalized_labels = (labels_tensor - self.label_means) / self.label_stds

        return x, pad_mask_tensor, normalized_labels    


    def __getitem__(self, idx):
        # Decide which logic to use
        if getattr(self, "mode", "gnn") == "transformer":
            return self.get_transformer(idx)
        else:
            return self.get(idx)

def create_dataloaders(feature_path, labels_path,
                       stats_load_path=None, stats_save_path=None,
                       batch_size=32, train_val_test_split=(0.7, 0.15, 0.15),
                       random_seed=42, num_workers=0, pin_memory=True):
    """
    Creates train, validation, and test DataLoaders for the FPGAGraphDataset.
    Handles loading or calculating normalization statistics ONLY from training data.

    Args:
        feature_path (str): Path to the features .npy file.
        labels_path (str): Path to the labels .npy file.
        stats_load_path (str, optional): Path to load pre-calculated normalization stats (.npy).
        stats_save_path (str, optional): Path to save newly calculated normalization stats (.npy).
        batch_size (int): Batch size for DataLoaders.
        train_val_test_split (tuple): Ratios for train, validation, test splits. Must sum to 1.0.
        random_seed (int): Random seed for shuffling and splitting.
        num_workers (int): Number of workers for DataLoader.
        pin_memory (bool): If True, DataLoaders will copy Tensors into CUDA pinned memory.

    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset, node_feature_dim, num_targets)
    """
    import numpy as np
    import os
    import tempfile
    from sklearn.model_selection import train_test_split
    import torch

    # UNCOMMENT
    # Load full dataset features and labels
    print(f"Loading features from: {feature_path}")
    full_features = np.load(feature_path)
    print(f"Loading labels from: {labels_path}")
    full_labels = np.load(labels_path)

    # # ADDED FOR TESTING
    # full_features = np.load(feature_path)[:10000]
    # full_labels = np.load(labels_path)[:10000]
    
    num_samples = full_features.shape[0]
    print(f"Loaded dataset with {num_samples} samples")
    
    # Check if we can use pre-calculated statistics
    initial_stats = None
    if stats_load_path and os.path.exists(stats_load_path):
        try:
            print(f"Attempting to load normalization stats from: {stats_load_path}")
            feature_means, feature_stds, label_means, label_stds = FPGAGraphDataset.load_normalization_stats(stats_load_path)
            initial_stats = (feature_means, feature_stds, label_means, label_stds)
            print("Successfully loaded normalization stats.")
        except Exception as e:
            print(f"Warning: Could not load stats from {stats_load_path}: {e}. Stats will be recalculated.")
            initial_stats = None
    
    # Split dataset indices - always do this to ensure consistent splitting
    indices = list(range(num_samples))
    
    train_ratio, val_ratio, test_ratio = train_val_test_split
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Train, validation, and test ratios must sum to 1.0. Got: {train_ratio+val_ratio+test_ratio}")

    # First split: training and temporary (validation + test)
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True
    )

    # Second split: validation and test from temporary
    # Calculate the proportion of the validation set relative to the temp set
    if (val_ratio + test_ratio) == 0:  # Avoid division by zero if val and test are 0
        if val_ratio == 0 and test_ratio == 0:
            val_indices = []
            test_indices = []
        else:  # This case should ideally not happen if sum is 1 and train_ratio < 1
            raise ValueError("val_ratio + test_ratio is 0, but individual ratios might not be.")
    else:
        val_relative_ratio = val_ratio / (val_ratio + test_ratio)
        if len(temp_indices) > 0:  # only split if temp_indices is not empty
            val_indices, test_indices = train_test_split(
                temp_indices,
                train_size=val_relative_ratio,  # train_size here means size of the first returned list (val_indices)
                random_state=random_seed,  # Use same seed for consistent split of this subset
                shuffle=True
            )
        else:  # if temp_indices is empty, val and test are also empty
            val_indices, test_indices = [], []

    print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test samples.")
    
    # If we need to calculate statistics, do it only on training data
    if initial_stats is None:
        print("Calculating normalization statistics from TRAINING DATA ONLY")
        
        # Create temporary files for training data only
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_features_file, \
             tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_labels_file:
            
            # Extract training features and labels
            train_features = full_features[train_indices]
            train_labels = full_labels[train_indices]
            
            # Save to temporary files
            tmp_features_path = tmp_features_file.name
            tmp_labels_path = tmp_labels_file.name
            np.save(tmp_features_path, train_features)
            np.save(tmp_labels_path, train_labels)
            
            print(f"Saved temporary training features to {tmp_features_path}")
            print(f"Saved temporary training labels to {tmp_labels_path}")
            
            # Create temporary dataset just to calculate statistics
            try:
                temp_dataset = FPGAGraphDataset(
                    features_path=tmp_features_path,
                    labels_path=tmp_labels_path,
                    stats=None  # Force calculation of stats
                )
                
                # Extract the calculated stats
                feature_means = temp_dataset.feature_means
                feature_stds = temp_dataset.feature_stds
                label_means = temp_dataset.label_means
                label_stds = temp_dataset.label_stds
                initial_stats = (feature_means, feature_stds, label_means, label_stds)
                
                # Save the stats if requested
                if stats_save_path:
                    print(f"Saving training-set normalization stats to: {stats_save_path}")
                    temp_dataset.save_normalization_stats(stats_save_path)
                
            finally:
                # Clean up temporary files
                os.unlink(tmp_features_path)
                os.unlink(tmp_labels_path)
                print("Cleaned up temporary files")
        
    # Now create the full dataset with the statistics from training data
    full_dataset = FPGAGraphDataset(
        features_path=feature_path,
        labels_path=labels_path,
        stats=initial_stats  # Pass the statistics calculated from training data
    )
    
    node_feature_dim = full_dataset.node_feature_dim
    num_targets = full_labels.shape[1] if full_labels.ndim == 2 else 1

    # Create PyTorch Subset objects
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Create PyTorch Geometric DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, full_dataset, node_feature_dim, num_targets




# Usage example:
# train_loader, val_loader, test_loader, dataset, node_feature_dim, num_targets = create_dataloaders(
#     feature_path=FEATURES_PATH,
#     labels_path=LABELS_PATH,
#     stats_load_path=STATS_PATH, # Load if exists
#     stats_save_path=STATS_PATH, # Save if calculated on training set
#     batch_size=BATCH_SIZE,
#     train_val_test_split=(0.7, 0.15, 0.15), # Example split
#     random_seed=42,
#     num_workers=0, # Adjust based on your system
#     pin_memory=True if device.type == 'cuda' else False
# )