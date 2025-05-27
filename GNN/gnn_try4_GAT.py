import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from Models import FPGA_GNN_GATv2, FPGA_GNN_GATv2_Enhanced
from Dataset2 import create_dataloaders
from Utils import generate_all_plots

# Configuration
FEATURES_PATH = 'May_15_processed/combined_features.npy'
LABELS_PATH = 'May_15_processed/combined_labels.npy'
# STATS_PATH = 'May_15_processed/calculated_TrainingOnly_normalization_stats.npy'
STATS_PATH = 'May_15_processed/calculated_TrainingOnly_norm_May26.npy'
SPLIT_MAPPING_PATH = 'May_15_processed/dataset_split_mapping_May26.npy'  # NEW: Add split mapping path


# Training configuration
BATCH_SIZE = 2048
LEARNING_RATE = 2e-3
NUM_EPOCHS = 750
WEIGHT_DECAY = 5e-6

# GATv2 specific hyperparameters
GNN_HIDDEN_DIM = 64
GNN_NUM_LAYERS = 3
NUM_ATTENTION_HEADS = 3  # New for GATv2
MLP_HIDDEN_DIM = 64
DROPOUT_RATE = 0.3


import os

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training", disable=True):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch)
        loss = criterion(predictions, batch.y.squeeze(1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability with attention mechanisms
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def evaluate(model, data_loader, criterion, device, dataset, denormalize=True):
    """Evaluate the model on given data loader."""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Forward pass
            predictions = model(batch)
            targets = batch.y.squeeze(1)
            
            # Calculate loss on normalized values
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions and targets
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Denormalize if requested
    if denormalize:
        all_predictions = dataset.denormalize_labels(all_predictions)
        all_targets = dataset.denormalize_labels(all_targets)
    
    avg_loss = total_loss / num_batches
    return avg_loss, all_predictions, all_targets

def calculate_metrics(predictions, targets, feature_names=None):
    """Calculate various regression metrics."""
    metrics = {}
    
    # Overall metrics
    mae = torch.mean(torch.abs(predictions - targets))
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    mape = torch.mean(torch.abs((targets - predictions) / (targets + 1e-8))) * 100
    
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

def train_gatv2_gnn(output_dir='results/GATv2_results', use_enhanced_model=False):
    print(f"\nUsing Enhanced GATv2 model: {use_enhanced_model}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    # 1. Create DataLoaders with split mapping for reproducibility
    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader, dataset, node_feature_dim, num_targets = create_dataloaders(
        feature_path=FEATURES_PATH,
        labels_path=LABELS_PATH,
        stats_load_path=None,
        stats_save_path=STATS_PATH,
        split_mapping_path=SPLIT_MAPPING_PATH,  # NEW: Add this parameter
        batch_size=BATCH_SIZE,
        train_val_test_split=(0.7, 0.15, 0.15),
        random_seed=42,
        num_workers=5,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"DataLoaders created successfully")
    print(f"Node feature dimension: {node_feature_dim}")
    print(f"Number of targets: {num_targets}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")


    # 2. Initialize Model
    if use_enhanced_model:
        print("\nUsing Enhanced GATv2 model with edge features and skip connections")
        model = FPGA_GNN_GATv2_Enhanced(
            node_feature_dim=node_feature_dim,
            num_targets=num_targets,
            hidden_dim=GNN_HIDDEN_DIM,
            num_gnn_layers=GNN_NUM_LAYERS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            mlp_hidden_dim=MLP_HIDDEN_DIM,
            dropout_rate=DROPOUT_RATE,
            use_edge_features=True
        ).to(device)
    else:
        print("\nUsing Standard GATv2 model")
        model = FPGA_GNN_GATv2(
            node_feature_dim=node_feature_dim,
            num_targets=num_targets,
            hidden_dim=GNN_HIDDEN_DIM,
            num_gnn_layers=GNN_NUM_LAYERS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            mlp_hidden_dim=MLP_HIDDEN_DIM,
            dropout_rate=DROPOUT_RATE,
            concat_heads=True,
            residual_connections=True
        ).to(device)
    
    print("\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 3. Initialize Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler with warmup
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=warmup_epochs/NUM_EPOCHS,
        anneal_strategy='cos'
    )

    # 4. Training Loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 25
    
    # feature_names = ['CYCLES', 'II', 'FF', 'LUT', 'BRAM', 'DSP']
    feature_names = ['CYCLES', 'FF', 'LUT', 'BRAM', 'DSP', 'II']

    # for epoch in range(NUM_EPOCHS):   
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs", unit="epoch", disable=True):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Step the scheduler after each batch (done in train_epoch)
        # Update learning rate after epoch
        if epoch >= warmup_epochs:
            for batch in train_loader:
                scheduler.step()
                break
        
        # Validation
        val_loss, val_predictions, val_targets = evaluate(
            model, val_loader, criterion, device, dataset, denormalize=False
        )
        val_losses.append(val_loss)
        
        # Early stopping and best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'model_config': {
                    'node_feature_dim': node_feature_dim,
                    'num_targets': num_targets,
                    'hidden_dim': GNN_HIDDEN_DIM,
                    'num_gnn_layers': GNN_NUM_LAYERS,
                    'num_attention_heads': NUM_ATTENTION_HEADS,
                    'mlp_hidden_dim': MLP_HIDDEN_DIM,
                    'dropout_rate': DROPOUT_RATE,
                    'use_enhanced': use_enhanced_model
                }
            }
            torch.save(checkpoint, f'{output_dir}/best_checkpoint.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 1 == 0 or epoch == NUM_EPOCHS - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6e}, Val Loss = {val_loss:.6e}, LR = {current_lr:.6e}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (patience: {early_stopping_patience})")
            break

    # 5. Load best model and evaluate
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation loss: {best_val_loss:.6e}")

    # 6. Final Evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION - GATv2 Model")
    print("="*50)

    # Test set evaluation
    test_loss_norm, test_pred, test_targets = evaluate(
        model, test_loader, criterion, device, dataset, denormalize=True
    )
    test_metrics = calculate_metrics(test_pred, test_targets, feature_names)
    
    print(f"\nTest Set Metrics (denormalized):")
    print(f"Overall Metrics:")
    for metric, value in test_metrics['overall'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nPer-Feature Metrics:")
    for feature, metrics in test_metrics['per_feature'].items():
        print(f"\n{feature}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # 7. Generate all plots
    print("\nGenerating plots...")
    generate_all_plots(
        model=model,
        dataset=dataset,
        train_loader=train_loader,
        test_loader=test_loader,
        train_losses=train_losses,
        val_losses=val_losses,
        output_dir=output_dir,
        device=device
    )

    # 8. Save final model
    final_model_path = f'{output_dir}/final_gatv2_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'node_feature_dim': node_feature_dim,
            'num_targets': num_targets,
            'hidden_dim': GNN_HIDDEN_DIM,
            'num_gnn_layers': GNN_NUM_LAYERS,
            'num_attention_heads': NUM_ATTENTION_HEADS,
            'mlp_hidden_dim': MLP_HIDDEN_DIM,
            'dropout_rate': DROPOUT_RATE,
            'use_enhanced': use_enhanced_model
        },
        'training_config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': epoch + 1,
            'weight_decay': WEIGHT_DECAY
        },
        'best_val_loss': best_val_loss,
        'test_metrics': test_metrics
    }, final_model_path)
    print(f"\nModel saved to: {final_model_path}")

    return model, dataset, train_losses, val_losses, test_metrics


if __name__ == "__main__":
    model, dataset, train_losses, val_losses, test_metrics = train_gatv2_gnn(output_dir= 'results/May_24_day/run2_GAT_enhanced', use_enhanced_model=True)
    # model, dataset, train_losses, val_losses, test_metrics = train_gatv2_gnn(output_dir= 'results/May_22_day/run3_GAT/', use_enhanced_model=False)