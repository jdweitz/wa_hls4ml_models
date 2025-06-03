# generate_plots.py
# Code to generate plots after training your GNN model

import torch
import numpy as np
from utils.plot import plot_loss, plot_box_plots_symlog, plot_results_simplified
import os


def generate_all_plots(model, dataset, train_loader, test_loader, train_losses, val_losses, 
                      output_dir="results", device=None):
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
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define output feature names based on your model
    # output_features = ['CYCLES', 'II', 'FF', 'LUT', 'BRAM', 'DSP']
    # output_features =['CYCLES', 'FF', 'LUT', 'BRAM', 'DSP', 'II']
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
    
    print(f"Test set size: {test_predictions_denorm.shape[0]} samples")
    print(f"Number of targets: {test_predictions_denorm.shape[1]}")
    
    # 3. Plot box plots with prediction errors
    print("3. Plotting box plots...")
    plot_box_plots_symlog(test_predictions_denorm, test_targets_denorm, output_dir)
    
    # 4. Plot scatter plots (actual vs predicted)
    print("4. Plotting scatter plots...")
    plot_results_simplified(
        name="GNN_Test_Results",
        mpl_plots=True,  # Generate matplotlib plots
        y_test=test_targets_denorm,
        y_pred=test_predictions_denorm,
        output_features=output_features,
        folder_name=output_dir
    )
    
    print(f"All plots saved to {output_dir}/plots/")
    
    # 5. Calculate and display metrics using the calculate_metrics function
    print("5. Calculating metrics...")
    
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