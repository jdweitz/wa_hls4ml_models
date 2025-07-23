# generate_plots.py
# Code to generate plots after training your GNN model

import torch
import numpy as np
from .plot import plot_loss, plot_box_plots_symlog, plot_results_simplified
import os
import torch.nn as nn
import torch.nn.functional as F



# def generate_all_plots_old(model, dataset, train_loader, test_loader, train_losses, val_losses, 
#                       output_dir="results", device=None):
#     """
#     Generate all plots for the trained GNN model.
    
#     Args:
#         model: Trained GNN model
#         dataset: Dataset object with denormalization methods
#         train_loader: Training data loader
#         test_loader: Test data loader  
#         train_losses: List of training losses per epoch
#         val_losses: List of validation losses per epoch
#         output_dir: Directory to save plots
#         device: Device to run inference on
#     """
    
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Define output feature names based on your model
#     # output_features = ['CYCLES', 'II', 'FF', 'LUT', 'BRAM', 'DSP']
#     # output_features =['CYCLES', 'FF', 'LUT', 'BRAM', 'DSP', 'II']
#     output_features = ['CYCLES', 'FF', 'LUT', 'BRAM', 'DSP', 'II'] 
    
    
#     print("Generating plots...")
    
#     # 1. Plot training curves
#     print("1. Plotting training curves...")
#     plot_loss(train_losses, val_losses, outdir=f"{output_dir}/plots")
    
#     # 2. Get test predictions for other plots
#     print("2. Getting test predictions...")
#     model.eval()
#     test_predictions = []
#     test_targets = []
    
#     with torch.no_grad():
#         for batch in test_loader:
#             batch = batch.to(device)
#             predictions = model(batch)
#             targets = batch.y.squeeze(1)
            
#             test_predictions.append(predictions.cpu())
#             test_targets.append(targets.cpu())
    
#     # Concatenate and denormalize for plotting
#     test_predictions = torch.cat(test_predictions, dim=0)
#     test_targets = torch.cat(test_targets, dim=0)
    
#     # Denormalize for interpretable plots
#     test_predictions_denorm = dataset.denormalize_labels(test_predictions).numpy()
#     test_targets_denorm = dataset.denormalize_labels(test_targets).numpy()

#     # In generate_all_plots, after denormalization:
#     print(f"Test predictions range: [{test_predictions_denorm.min():.2f}, {test_predictions_denorm.max():.2f}]")
#     print(f"Test targets range: [{test_targets_denorm.min():.2f}, {test_targets_denorm.max():.2f}]")
#     print(f"Ratio of ranges: {(test_predictions_denorm.max() - test_predictions_denorm.min()) / (test_targets_denorm.max() - test_targets_denorm.min()):.2f}")
    
#     print(f"Test set size: {test_predictions_denorm.shape[0]} samples")
#     print(f"Number of targets: {test_predictions_denorm.shape[1]}")
    
#     # 3. Plot box plots with prediction errors
#     print("3. Plotting box plots...")
#     plot_box_plots_symlog(test_predictions_denorm, test_targets_denorm, output_dir)
    
#     # 4. Plot scatter plots (actual vs predicted)
#     print("4. Plotting scatter plots...")
#     plot_results_simplified(
#         name="GNN_Test_Results",
#         mpl_plots=True,  # Generate matplotlib plots
#         y_test=test_targets_denorm,
#         y_pred=test_predictions_denorm,
#         output_features=output_features,
#         folder_name=output_dir
#     )
    
#     print(f"All plots saved to {output_dir}/plots/")
    
#     # 5. Calculate and display metrics using the calculate_metrics function
#     print("5. Calculating metrics...")
    
#     # Convert back to tensors for metric calculation (using denormalized values)
#     test_predictions_tensor = torch.tensor(test_predictions_denorm, dtype=torch.float32)
#     test_targets_tensor = torch.tensor(test_targets_denorm, dtype=torch.float32)
    
#     metrics = calculate_metrics(test_predictions_tensor, test_targets_tensor, output_features)
    
#     # Display metrics
#     print("\nOverall Metrics:")
#     print("="*50)
#     for metric_name, value in metrics['overall'].items():
#         if metric_name in ['MAPE', 'SMAPE']:
#             print(f"{metric_name}: {value:.2f}%")
#         else:
#             print(f"{metric_name}: {value:.5e}")
    
#     print("\nPer-Feature Metrics:")
#     print("="*50)
#     for feature_name, feature_metrics in metrics['per_feature'].items():
#         print(f"{feature_name}:")
#         for metric_name, value in feature_metrics.items():
#             if metric_name in ['MAPE', 'SMAPE']:
#                 print(f"  {metric_name}: {value:.2f}%")
#             else:
#                 print(f"  {metric_name}: {value:.5e}")
#         print()
    
#     return metrics

def generate_all_plots(model, dataset, train_loader, test_loader, train_losses, val_losses, 
                      output_dir="results", device=None, test_features_path=None):
    """
    Generate all plots for the trained GNN model.
    
    Args:
        model: Trained GNN model
        dataset: Dataset object with denormalization methods
        train_loader: Training data loader
        test_loader: Test data loader  
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        output_dir: Directory to save plots
        device: Device to run inference on
        test_features_path: Path to test features for model type classification
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define output feature names based on your model
    output_features = ['CYCLES', 'FF', 'LUT', 'BRAM', 'DSP', 'II'] 
    
    print("Generating plots...")
    
    # 1. Plot training curves
    print("1. Plotting training curves...")
    plot_loss(train_losses, val_losses, outdir=f"{output_dir}/plots")
    
    # 2. Get test predictions for other plots
    print("2. Getting test predictions...")
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            predictions = model(batch)
            targets = batch.y.squeeze(1)
            
            test_predictions.append(predictions.cpu())
            test_targets.append(targets.cpu())
    
    # Concatenate and denormalize for plotting
    test_predictions = torch.cat(test_predictions, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    # Denormalize for interpretable plots
    test_predictions_denorm = dataset.denormalize_labels(test_predictions).numpy()
    test_targets_denorm = dataset.denormalize_labels(test_targets).numpy()

    # Get model types for colored plotting
    model_types = None
    if test_features_path and os.path.exists(test_features_path):
        print("3. Loading test features for model type classification...")
        test_features_np = np.load(test_features_path)
        model_types = get_model_types(test_features_np)
        print(f"   Found {len(model_types)} test samples")
        
        # Count model types
        type_counts = {}
        for mt in model_types:
            type_counts[mt] = type_counts.get(mt, 0) + 1
        print(f"   Model type distribution: {type_counts}")
    else:
        print("3. No test features path provided, using default coloring...")

    print(f"Test predictions range: [{test_predictions_denorm.min():.2f}, {test_predictions_denorm.max():.2f}]")
    print(f"Test targets range: [{test_targets_denorm.min():.2f}, {test_targets_denorm.max():.2f}]")
    print(f"Ratio of ranges: {(test_predictions_denorm.max() - test_predictions_denorm.min()) / (test_targets_denorm.max() - test_targets_denorm.min()):.2f}")
    
    print(f"Test set size: {test_predictions_denorm.shape[0]} samples")
    print(f"Number of targets: {test_predictions_denorm.shape[1]}")
    
    # 4. Plot box plots with prediction errors
    print("4. Plotting box plots...")
    plot_box_plots_symlog(test_predictions_denorm, test_targets_denorm, output_dir)
    
    # 5. Plot scatter plots (actual vs predicted) with model type coloring
    print("5. Plotting scatter plots...")
    plot_results_simplified(
        name="GNN_Test_Results",
        mpl_plots=True,  # Generate matplotlib plots
        y_test=test_targets_denorm,
        y_pred=test_predictions_denorm,
        output_features=output_features,
        folder_name=output_dir,
        model_types=model_types  # Pass the model types here!
    )
    
    print(f"All plots saved to {output_dir}/plots/")
    
    # 6. Calculate and display metrics using the calculate_metrics function
    print("6. Calculating metrics...")
    
    # Convert back to tensors for metric calculation (using denormalized values)
    test_predictions_tensor = torch.tensor(test_predictions_denorm, dtype=torch.float32)
    test_targets_tensor = torch.tensor(test_targets_denorm, dtype=torch.float32)
    
    metrics = calculate_metrics(test_predictions_tensor, test_targets_tensor, output_features)
    
    # Display metrics
    print("\nOverall Metrics:")
    print("="*50)
    for metric_name, value in metrics['overall'].items():
        if metric_name in ['MAPE', 'SMAPE']:
            print(f"{metric_name}: {value:.2f}%")
        else:
            print(f"{metric_name}: {value:.5e}")
    
    print("\nPer-Feature Metrics:")
    print("="*50)
    for feature_name, feature_metrics in metrics['per_feature'].items():
        print(f"{feature_name}:")
        for metric_name, value in feature_metrics.items():
            if metric_name in ['MAPE', 'SMAPE']:
                print(f"  {metric_name}: {value:.2f}%")
            else:
                print(f"  {metric_name}: {value:.5e}")
        print()
    
    return metrics


def calculate_metrics(predictions, targets, feature_names=None):
    """Calculate various regression metrics."""
    metrics = {}
    
    # Overall metrics
    mae = torch.mean(torch.abs(predictions - targets))
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    mape = torch.mean(torch.abs((targets - predictions) / (targets + 1e-8))) * 100
    # mape = torch.mean(torch.abs((targets - predictions) / (torch.abs(targets) + epsilon))) * 100


    
    # SMAPE (Symmetric Mean Absolute Percentage Error)
    smape = torch.mean(torch.abs(predictions - targets) / ((torch.abs(predictions) + torch.abs(targets)) / 2 + 1e-8)) * 100
    
    # R² (Coefficient of determination)
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    metrics['overall'] = {
        'MAE': mae.item(),
        'MSE': mse.item(),
        'RMSE': rmse.item(),
        'MAPE': mape.item(),
        'SMAPE': smape.item(),
        'R2': r2.item()
    }
    
    # Per-feature metrics if feature names provided
    if feature_names is not None:
        metrics['per_feature'] = {}
        for i, name in enumerate(feature_names):
            feat_mae = torch.mean(torch.abs(predictions[:, i] - targets[:, i]))
            feat_mse = torch.mean((predictions[:, i] - targets[:, i]) ** 2)
            feat_rmse = torch.sqrt(feat_mse)
            feat_mape = torch.mean(torch.abs((targets[:, i] - predictions[:, i]) / 
                                           (targets[:, i] + 1e-8))) * 100
            
            # Per-feature SMAPE
            feat_smape = torch.mean(torch.abs(predictions[:, i] - targets[:, i]) / 
                                  ((torch.abs(predictions[:, i]) + torch.abs(targets[:, i])) / 2 + 1e-8)) * 100
            
            # Per-feature R²
            feat_ss_res = torch.sum((targets[:, i] - predictions[:, i]) ** 2)
            feat_ss_tot = torch.sum((targets[:, i] - torch.mean(targets[:, i])) ** 2)
            feat_r2 = 1 - (feat_ss_res / (feat_ss_tot + 1e-8))
            
            metrics['per_feature'][name] = {
                'MAE': feat_mae.item(),
                'MSE': feat_mse.item(),
                'RMSE': feat_rmse.item(),
                'MAPE': feat_mape.item(),
                'SMAPE': feat_smape.item(),
                'R2': feat_r2.item()
            }
    
    return metrics

def get_model_types(features_np):
    """
    Determine model types based on layer types present in the features.
    
    Args:
        features_np: numpy array of shape (num_samples, max_layers, num_features)
    
    Returns:
        list: List of model types ['Dense', 'Conv1D', 'Conv2D'] for each sample
    """
    model_types = []
    for model in features_np:
        valid_layers = [layer for layer in model if not np.all(layer == -1)]
        layer_types = [int(layer[9]) for layer in valid_layers]  # layer_type is at index 9
        if any(lt == 2 for lt in layer_types):
            model_types.append('Conv1D')
        elif any(lt == 3 for lt in layer_types):
            model_types.append('Conv2D')
        else:
            model_types.append('Dense')
    return model_types


def save_metrics_to_file(metrics, output_dir, model_config=None, training_config=None, name_base="test"):
    """
    Save metrics to a text file in a nicely formatted way.
    
    Args:
        metrics (dict): Dictionary containing test metrics from calculate_metrics()
        output_dir (str): Directory to save the metrics file
        model_config (dict, optional): Model configuration parameters
        training_config (dict, optional): Training configuration parameters
    """
    # metrics_file_path = os.path.join(output_dir, 'test_metrics.txt')
    metrics_file_path = os.path.join(output_dir, f'{name_base}_metrics.txt')
    
    with open(metrics_file_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION METRICS\n")
        f.write("="*60 + "\n\n")
        
        # Write model configuration if provided
        if model_config:
            f.write("MODEL CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            for key, value in model_config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        # Write training configuration if provided
        if training_config:
            f.write("TRAINING CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            for key, value in training_config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        # Write overall metrics
        f.write("OVERALL METRICS:\n")
        f.write("-" * 30 + "\n")
        for metric, value in metrics['overall'].items():
            if metric in ['MAPE', 'SMAPE']:
                f.write(f"{metric}: {value:.2f}%\n")
            else:
                f.write(f"{metric}: {value:.6e}\n")
        f.write("\n")
        
        # Write per-feature metrics
        f.write("PER-FEATURE METRICS:\n")
        f.write("-" * 30 + "\n")
        for feature, feature_metrics in metrics['per_feature'].items():
            f.write(f"\n{feature}:\n")
            for metric, value in feature_metrics.items():
                if metric in ['MAPE', 'SMAPE']:
                    f.write(f"  {metric}: {value:.2f}%\n")
                else:
                    f.write(f"  {metric}: {value:.6e}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("End of Metrics Report\n")
        f.write("="*60 + "\n")
    
    print(f"Metrics saved to: {metrics_file_path}")


class MagnitudeWeightedMSELoss(nn.Module):
    """MSE loss weighted by target magnitude to emphasize larger values."""
    def __init__(self, alpha=0.5, epsilon=1.0):
        super().__init__()
        self.alpha = alpha  # Controls the strength of weighting
        self.epsilon = epsilon  # Prevents division by zero
        
    def forward(self, predictions, targets):
        # Calculate weights based on target magnitude
        # Larger targets get higher weights
        weights = torch.pow(torch.abs(targets) + self.epsilon, self.alpha)
        weights = weights / weights.mean()  # Normalize weights
        
        # Calculate weighted MSE
        mse = (predictions - targets) ** 2
        weighted_mse = mse * weights
        
        return weighted_mse.mean()


class RelativeMSELoss(nn.Module):
    """Relative MSE that penalizes percentage errors rather than absolute errors."""
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, predictions, targets):
        # Calculate relative error
        relative_error = (predictions - targets) / (torch.abs(targets) + self.epsilon)
        relative_mse = relative_error ** 2
        
        return relative_mse.mean()


class AsymmetricMSELoss(nn.Module):
    """Asymmetric loss that penalizes underprediction more than overprediction."""
    def __init__(self, underpredict_weight=2.0):
        super().__init__()
        self.underpredict_weight = underpredict_weight
        
    def forward(self, predictions, targets):
        errors = predictions - targets
        
        # Apply different weights for under vs over prediction
        weights = torch.where(errors < 0, 
                            self.underpredict_weight, 
                            torch.ones_like(errors))
        
        weighted_mse = weights * (errors ** 2)
        return weighted_mse.mean()


class QuantileWeightedMSELoss(nn.Module):
    """Weights samples based on their target quantile - higher quantiles get more weight."""
    def __init__(self, quantile_weights=None):
        super().__init__()
        # Default: emphasize top quantiles
        if quantile_weights is None:
            quantile_weights = [0.5, 0.7, 1.0, 1.5, 2.0]  # for quintiles
        self.quantile_weights = torch.tensor(quantile_weights)
        
    def forward(self, predictions, targets, batch):
        # For each feature, determine quantile of each sample
        batch_size = predictions.shape[0]
        num_features = predictions.shape[1]
        
        weights = torch.ones_like(predictions)
        
        for feat_idx in range(num_features):
            # Get quantiles for this feature
            feat_targets = targets[:, feat_idx]
            quantiles = torch.quantile(feat_targets, 
                                     torch.linspace(0, 1, len(self.quantile_weights) + 1).to(targets.device))
            
            # Assign weights based on quantile
            for i in range(len(self.quantile_weights)):
                if i < len(self.quantile_weights) - 1:
                    mask = (feat_targets >= quantiles[i]) & (feat_targets < quantiles[i+1])
                else:
                    mask = feat_targets >= quantiles[i]
                weights[mask, feat_idx] = self.quantile_weights[i]
        
        # Calculate weighted MSE
        mse = (predictions - targets) ** 2
        weighted_mse = mse * weights
        
        return weighted_mse.mean()
