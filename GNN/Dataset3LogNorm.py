import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import os
from tqdm import tqdm


class FPGAGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for FPGA resource utilization and latency estimation.
    Each sample is a neural network represented as a graph, where nodes are layers
    and edges represent the flow of data between layers.
    """
    def __init__(self, features_path, labels_path, transform=None, pre_transform=None, 
             stats=None, use_log_transform=False, log_epsilon=1e-6, log_shift=None):
        super(FPGAGraphDataset, self).__init__(None, transform, pre_transform) # root=None as we load from numpy
        print(f"Loading features from: {features_path}")
        self.features_np = np.load(features_path)
        print(f"Loading labels from: {labels_path}")
        self.labels_np = np.load(labels_path)
        
        # Log transformation settings
        self.use_log_transform = use_log_transform
        self.log_epsilon = log_epsilon
        print(f"Using log transformation: {self.use_log_transform}")
        if self.use_log_transform:
            print(f"Log epsilon: {self.log_epsilon}")

        # Feature indices from the dataset
        self.feature_indices_map = {
            "d_in1": 0, "d_in2": 1, "d_in3": 2,
            "d_out1": 3, "d_out2": 4, "d_out3": 5,
            "prec": 6, "rf": 7, "strategy": 8,
            "layer_type": 9, "activation_type": 10,
            "filters": 11, "kernel_size": 12,
            "stride": 13, "padding": 14, "pooling": 15,
            "batchnorm": 16, "io_type": 17
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
            'AveragePooling1D': 10, 'AveragePooling2D': 10, 'BatchNormalization': 11
        }
        self.activation_mapping_provided = { 
            'NA': 0, 'linear': 1, 'quantized_relu': 2, 'quantized_tanh': 3,
            'quantized_sigmoid': 4, 'quantized_softmax': 5
        }
        self.padding_mapping_provided = {
            'NA': 0, 'valid': 1, 'same': 2,
        }
        self.io_type_mapping_provided = {
            'io_parallel': 0, 'io_stream': 1,
        }

        # Get the number of unique categorical values
        self.num_layer_types = len(set(self.layer_type_mapping_provided.values()))
        self.num_activation_types = len(self.activation_mapping_provided)
        self.num_padding_types = len(self.padding_mapping_provided)
        self.num_io_types = len(self.io_type_mapping_provided)

        # Specific indices for easier access
        self.strategy_idx = self.feature_indices_map["strategy"]
        self.layer_type_idx = self.feature_indices_map["layer_type"]
        self.activation_type_idx = self.feature_indices_map["activation_type"]
        self.padding_idx = self.feature_indices_map["padding"]
        self.io_type_idx = self.feature_indices_map["io_type"]

        # Add this as a class attribute
        self.log_shift = 0.0  # Initialize

        # Apply log transformation to labels if requested
        if self.use_log_transform:
            if log_shift is not None:
                self.log_shift = log_shift
                # Apply log transform with the provided shift
                self.labels_np = np.log(self.labels_np + self.log_shift)
            else:
                self.labels_np = self._apply_log_transform(self.labels_np)

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

    def _apply_log_transform(self, labels):
        """Apply log transformation to labels with epsilon to handle zeros/small values."""
        min_val = np.min(labels)
        if min_val <= 0:
            print(f"Warning: Found non-positive values in labels (min: {min_val}). Adding shift.")
            self.log_shift = self.log_epsilon + abs(min_val)
        else:
            self.log_shift = self.log_epsilon
        
        labels = labels + self.log_shift
        log_labels = np.log(labels)
        
        print(f"Applied log transform with shift: {self.log_shift}")
        print(f"Original range: [{np.min(labels-self.log_shift):.2e}, {np.max(labels-self.log_shift):.2e}]")
        print(f"Log-transformed range: [{np.min(log_labels):.2f}, {np.max(log_labels):.2f}]")
        return log_labels

    def _calculate_normalization_stats(self):
        """
        Calculates mean and std for numerical features and labels across the entire dataset
        """
        # Accumulator for numerical values for each feature
        numerical_values_accumulator = [[] for _ in range(self.num_numerical_features)]
        
        # Accumulator for labels
        label_values_accumulator = []

        print("Iterating through dataset to compute normalization stats...")
        for model_idx in tqdm(range(self.features_np.shape[0]), desc="Calculating Stats"):
            model_features = self.features_np[model_idx]
            model_labels = self.labels_np[model_idx]
            
            # Collect labels
            label_values_accumulator.append(model_labels)
            
            for layer_idx in range(model_features.shape[0]):
                layer_data = model_features[layer_idx]

                # Check if the layer is a padded row (all -1s)
                if np.all(layer_data == -1):
                    continue

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
            if vals:
                feature_means.append(np.mean(vals))
                std_val = np.std(vals)
                feature_stds.append(std_val if std_val > 1e-5 else 1.0)
            else:
                num_feat_original_idx = self.numerical_feature_indices[i].item()
                print(f"Warning: No valid data points found for numerical feature '{self.numerical_feature_keys[i]}' (original index {num_feat_original_idx}). Using default mean=0, std=1.")
                feature_means.append(0.0)
                feature_stds.append(1.0)

        # Calculate label statistics (on potentially log-transformed labels)
        all_labels = np.vstack(label_values_accumulator)
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
            value = int(raw_value)
        except ValueError:
            print(f"Warning: Could not convert raw_value '{raw_value}' to int for OHE. Returning zero vector.")
            return torch.zeros(num_classes, dtype=torch.float)

        tensor = torch.zeros(num_classes, dtype=torch.float)

        if 0 <= value < num_classes:
            tensor[value] = 1.0
        else:
            print(f"Warning: Integer value {value} (from raw '{raw_value}') is out of bounds for OHE with {num_classes} classes. Returning zero vector.")
        return tensor

    def len(self):
        """Returns the number of graphs in the dataset."""
        return self.features_np.shape[0]
    
    def save_normalization_stats(self, filepath):
        """Save the calculated normalization statistics (means and stds) to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        stats_dict = {
            'feature_means': self.feature_means.numpy(),
            'feature_stds': self.feature_stds.numpy(),
            'label_means': self.label_means.numpy(),
            'label_stds': self.label_stds.numpy(),
            'feature_keys': self.numerical_feature_keys,
            'use_log_transform': self.use_log_transform,
            'log_epsilon': self.log_epsilon,
            'log_shift': self.log_shift  # Add this
        }
        
        np.save(filepath, stats_dict)
        print(f"Normalization statistics saved to {filepath}")

    
    @staticmethod
    def load_normalization_stats(filepath):
        """Load normalization statistics from a file"""
        stats_dict = np.load(filepath, allow_pickle=True).item()
        
        feature_means = torch.tensor(stats_dict['feature_means'], dtype=torch.float)
        feature_stds = torch.tensor(stats_dict['feature_stds'], dtype=torch.float)
        label_means = torch.tensor(stats_dict['label_means'], dtype=torch.float)
        label_stds = torch.tensor(stats_dict['label_stds'], dtype=torch.float)
        
        print(f"Loaded statistics for features: {stats_dict['feature_keys']}")
        
        # Return transformation settings along with stats
        use_log_transform = stats_dict.get('use_log_transform', False)
        log_epsilon = stats_dict.get('log_epsilon', 1e-6)
        log_shift = stats_dict.get('log_shift', log_epsilon)
        
        return feature_means, feature_stds, label_means, label_stds, use_log_transform, log_epsilon, log_shift
    
    

    def denormalize_labels(self, normalized_labels):
        """Denormalize labels using the stored statistics"""
        # First denormalize from z-score
        denormalized = normalized_labels * self.label_stds + self.label_means
        
        # Then apply inverse log transform if it was used
        if self.use_log_transform:
            denormalized = torch.exp(denormalized) - self.log_shift  # Use log_shift instead of just epsilon
            # Handle potential negative values from numerical precision
            denormalized = torch.clamp(denormalized, min=0.0)
        
        return denormalized

    def get(self, idx):
        """Processes and returns a single graph data object for the given index."""
        model_features_raw = self.features_np[idx]
        model_labels = self.labels_np[idx]

        node_features_list = []
        valid_layer_original_indices = []

        for i in range(model_features_raw.shape[0]):
            layer_data_raw = model_features_raw[i]
            if np.all(layer_data_raw == -1):
                continue

            valid_layer_original_indices.append(i)
            
            # 1. Numerical Features
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
            
            # 2. Categorical Features (One-Hot Encoded)
            layer_type_val = layer_data_raw[self.layer_type_idx]
            activation_type_val = layer_data_raw[self.activation_type_idx]
            padding_val = layer_data_raw[self.padding_idx]

            ohe_layer_type = self._one_hot_encode(layer_type_val, self.num_layer_types)
            ohe_activation_type = self._one_hot_encode(activation_type_val, self.num_activation_types)
            ohe_padding = self._one_hot_encode(padding_val, self.num_padding_types)

            # Concatenate node features
            current_node_features = torch.cat([
                numerical_feats_processed,
                ohe_layer_type,
                ohe_activation_type,
                ohe_padding
            ], dim=0)
            node_features_list.append(current_node_features)

        x = torch.stack(node_features_list)
        
        # Global graph features: strategy AND io_type (both one-hot encoded)
        first_valid_layer_data = model_features_raw[valid_layer_original_indices[0]]
        strategy_val = int(first_valid_layer_data[self.strategy_idx])
        io_type_val = int(first_valid_layer_data[self.io_type_idx])
        
        # One-hot encode global features
        strategy_ohe = self._one_hot_encode(strategy_val, 2)
        io_type_ohe = self._one_hot_encode(io_type_val, self.num_io_types)

        # Edges
        num_valid_nodes = x.shape[0]
        if num_valid_nodes > 1:
            source_nodes = torch.arange(0, num_valid_nodes - 1, dtype=torch.long)
            target_nodes = torch.arange(1, num_valid_nodes, dtype=torch.long)
            edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Labels (already log-transformed if use_log_transform=True)
        labels_tensor = torch.tensor(model_labels, dtype=torch.float)
        normalized_labels = (labels_tensor - self.label_means) / self.label_stds
        y = normalized_labels.unsqueeze(0)

        # Create PyG Data object with one-hot encoded global features
        data = Data(x=x, edge_index=edge_index, y=y)
        data.strategy = strategy_ohe.unsqueeze(0)
        data.io_type = io_type_ohe.unsqueeze(0)

        return data


def create_dataloaders_from_split_data(
    train_features_path, train_labels_path,
    val_features_path, val_labels_path,
    test_features_path, test_labels_path,
    stats_load_path=None, stats_save_path=None,
    batch_size=32, num_workers=0, pin_memory=True,
    use_log_transform=False, log_epsilon=1e-6
):
    """
    Creates train, validation, and test DataLoaders from pre-split numpy arrays.
    Calculates normalization statistics from training data only.
    """
    
    # Initialize stats to None
    stats = None
    
    # Handle normalization statistics
    if stats_load_path and os.path.exists(stats_load_path):
        print(f"Loading existing normalization stats from: {stats_load_path}")
        loaded_stats = FPGAGraphDataset.load_normalization_stats(stats_load_path)
        if len(loaded_stats) == 7:  # New format with log transform info and log_shift
            feature_means, feature_stds, label_means, label_stds, loaded_use_log, loaded_epsilon, loaded_log_shift = loaded_stats
            if loaded_use_log != use_log_transform:
                print(f"Warning: Loaded log transform setting ({loaded_use_log}) differs from requested ({use_log_transform})")
                print("Using requested setting and recalculating stats...")
                stats = None
            else:
                stats = (feature_means, feature_stds, label_means, label_stds)
                print("Successfully loaded normalization stats with log transform info.")
        elif len(loaded_stats) == 6:  # Format without log_shift
            feature_means, feature_stds, label_means, label_stds, loaded_use_log, loaded_epsilon = loaded_stats
            if loaded_use_log != use_log_transform:
                print(f"Warning: Loaded log transform setting ({loaded_use_log}) differs from requested ({use_log_transform})")
                print("Using requested setting and recalculating stats...")
                stats = None
            else:
                stats = (feature_means, feature_stds, label_means, label_stds)
                print("Successfully loaded normalization stats with log transform info.")
        else:  # Old format without log transform info
            feature_means, feature_stds, label_means, label_stds = loaded_stats
            print("Loaded old format stats. Recalculating for log transform compatibility...")
            stats = None
    else:
        stats = None  # Explicitly set to None if no stats file
    
    if stats is None:
        print("Calculating normalization statistics from training data...")
        # Create training dataset to calculate stats
        train_dataset_temp = FPGAGraphDataset(
            train_features_path, train_labels_path, 
            stats=None, use_log_transform=use_log_transform, log_epsilon=log_epsilon
        )
        stats = (train_dataset_temp.feature_means, train_dataset_temp.feature_stds,
                train_dataset_temp.label_means, train_dataset_temp.label_stds)
        
        # Save stats if requested
        if stats_save_path:
            print(f"Saving normalization stats to: {stats_save_path}")
            train_dataset_temp.save_normalization_stats(stats_save_path)

    # Continue with the rest of the function...

    # Create datasets with shared normalization statistics
    print("Creating datasets with shared normalization statistics...")
    train_dataset = FPGAGraphDataset(
        train_features_path, train_labels_path, 
        stats=stats, use_log_transform=use_log_transform, log_epsilon=log_epsilon
    )
    val_dataset = FPGAGraphDataset(
        val_features_path, val_labels_path, 
        stats=stats, use_log_transform=use_log_transform, log_epsilon=log_epsilon
    )
    test_dataset = FPGAGraphDataset(
        test_features_path, test_labels_path, 
        stats=stats, use_log_transform=use_log_transform, log_epsilon=log_epsilon
    )
    
    # Get dataset info
    node_feature_dim = train_dataset.node_feature_dim
    num_targets = train_dataset.labels_np.shape[1] if train_dataset.labels_np.ndim == 2 else 1
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader, node_feature_dim, num_targets