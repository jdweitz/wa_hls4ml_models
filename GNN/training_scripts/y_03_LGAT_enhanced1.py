import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Models import FPGA_GNN_GATv2, FPGA_GNN_GATv2_Enhanced
# from Dataset2 import create_dataloaders_from_split_data  # Updated import
from DatasetGNN import create_dataloaders_from_split_data, FPGAGraphDataset
from utils.Utils import generate_all_plots, calculate_metrics, save_metrics_to_file


#!!!!!! CHANGE THIS FOLDER NAME TO YOUR OWN FOLDER NAME !!!!!!!#
SAVE_FOLDERNAME = 'results/y_03_LGAT_enhanced1'
# CHANGE THIS FOLDER NAME TO YOUR OWN FOLDER NAME !!!!!!!#

# Extract the part after the first '/' for folder_base





# image_data_path = '/app/dataset/Full_dataset_processed_split/'
image_data_path = '/dima-pvc/wa_hls4ml_models/dataset/Full_dataset_processed_split/'

TRAIN_FEATURES_PATH = os.path.join(image_data_path, 'train_features.npy')
TRAIN_LABELS_PATH = os.path.join(image_data_path, 'train_labels.npy')
VAL_FEATURES_PATH = os.path.join(image_data_path, 'val_features.npy')
VAL_LABELS_PATH = os.path.join(image_data_path, 'val_labels.npy')
TEST_FEATURES_PATH = os.path.join(image_data_path, 'test_features.npy')
TEST_LABELS_PATH = os.path.join(image_data_path, 'test_labels.npy')


# Training configuration
BATCH_SIZE = 2048
LEARNING_RATE = 3e-3
NUM_EPOCHS = 500
WEIGHT_DECAY = 5e-6

GNN_HIDDEN_DIM = 128
GNN_NUM_LAYERS = 4
NUM_ATTENTION_HEADS = 4
MLP_HIDDEN_DIM = 160
DROPOUT_RATE = 0.3


# Normalization stats path (will be created if doesn't exist)
# Add log transformation parameters
USE_LOG_TRANSFORM = True  # Set to True to enable log transformation
LOG_EPSILON = 1e-6       # Small value to add before log transform
STATS_PATH = './results/normalization_stats_log.npy' if USE_LOG_TRANSFORM else './results/normalization_stats_01.npy'


# Before running training
# if os.path.exists(STATS_PATH):
#     os.remove(STATS_PATH)
#     print(f"Deleted old stats file: {STATS_PATH}")

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

def train_gatv2_gnn(output_dir='results/GATv2_results', use_enhanced_model=False):
    print(f"\nUsing Enhanced GATv2 model: {use_enhanced_model}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    # 1. Create DataLoaders from pre-split data
    print("Creating DataLoaders from pre-split data...")
    # Update the dataloader creation call (around line 140 in your original code)
    train_loader, val_loader, test_loader, node_feature_dim, num_targets = create_dataloaders_from_split_data(
        train_features_path=TRAIN_FEATURES_PATH,
        train_labels_path=TRAIN_LABELS_PATH,
        val_features_path=VAL_FEATURES_PATH,
        val_labels_path=VAL_LABELS_PATH,
        test_features_path=TEST_FEATURES_PATH,
        test_labels_path=TEST_LABELS_PATH,
        stats_load_path=STATS_PATH if os.path.exists(STATS_PATH) else None,
        # stats_load_path= None,
        stats_save_path=STATS_PATH,
        batch_size=BATCH_SIZE,
        num_workers=5,
        pin_memory=True if device.type == 'cuda' else False,
        use_log_transform=USE_LOG_TRANSFORM,  # Add this line
        log_epsilon=LOG_EPSILON                # Add this line
    )
    
    # Load stats for denormalization dataset instance
    loaded_log_shift_val = None # Initialize
    stats_for_denorm_dataset = None # Initialize

    if os.path.exists(STATS_PATH):
        # This load_normalization_stats returns 7 values:
        # fm, fs, lm, ls, use_log, log_eps, log_shift
        loaded_stats_tuple = FPGAGraphDataset.load_normalization_stats(STATS_PATH)
        
        if len(loaded_stats_tuple) == 7:  # Current expected format
            fm, fs, lm, ls, _, _, loaded_log_shift_val = loaded_stats_tuple
            stats_for_denorm_dataset = (fm, fs, lm, ls)
        elif len(loaded_stats_tuple) == 6:  # Older format (if load_normalization_stats was the commented one)
            fm, fs, lm, ls, _, _ = loaded_stats_tuple
            print(f"Warning: Loaded stats from {STATS_PATH} has 6 items, expected 7. Assuming no log transform.")
            stats_for_denorm_dataset = (fm, fs, lm, ls)
            # loaded_log_shift_val might be set to LOG_EPSILON or remain None,
            # depending on desired behavior if stats file is from an older version.
            # The FPGAGraphDataset.__init__ handles log_shift=None by calculating it.
        elif len(loaded_stats_tuple) == 4: # Very old format (only the four stats values)
            stats_for_denorm_dataset = loaded_stats_tuple
            print(f"Warning: Loaded stats from {STATS_PATH} has 4 items, expected 7. Assuming no log transform.")
            # loaded_log_shift_val will remain None
        else:
            # Or, if stats must exist and be in a known format:
            raise ValueError(f"Loaded stats from {STATS_PATH} has an unexpected number of items: {len(loaded_stats_tuple)}")
            # Alternatively, could fall back to recalculating, but the current script structure expects to load them here.

    else:
        # The original code raises FileNotFoundError, which is fine.
        # If stats must exist:
        raise FileNotFoundError(f"Stats file not found at {STATS_PATH}, which is required for the denormalization dataset.")

    # Create a dataset object for denormalization purposes (line 169)
    # Pass the correctly extracted 4-tuple 'stats_for_denorm_dataset'
    # Also, pass the 'loaded_log_shift_val' if USE_LOG_TRANSFORM is True
    dataset = FPGAGraphDataset(
        TRAIN_FEATURES_PATH, TRAIN_LABELS_PATH,
        stats=stats_for_denorm_dataset, # This will now be a 4-tuple
        use_log_transform=USE_LOG_TRANSFORM,
        log_epsilon=LOG_EPSILON,
        log_shift=loaded_log_shift_val if USE_LOG_TRANSFORM else None # Pass the loaded log_shift
    )
    #----------------------------------
    
    # # Load or calculate stats for denormalization
    # if os.path.exists(STATS_PATH):
    #     loaded_stats = FPGAGraphDataset.load_normalization_stats(STATS_PATH)
    #     if len(loaded_stats) == 6:  # New format with log transform info
    #         feature_means, feature_stds, label_means, label_stds, loaded_use_log, loaded_epsilon = loaded_stats
    #         stats = (feature_means, feature_stds, label_means, label_stds)
    #     else:  # Old format
    #         stats = loaded_stats
    # else:
    #     raise FileNotFoundError(f"Stats file not found: {STATS_PATH}")

    # # Create a dataset object for denormalization purposes
    # dataset = FPGAGraphDataset(
    #     TRAIN_FEATURES_PATH, TRAIN_LABELS_PATH, 
    #     stats=stats, 
    #     use_log_transform=USE_LOG_TRANSFORM, 
    #     log_epsilon=LOG_EPSILON
    # )
    
    print(f"DataLoaders created successfully")
    print(f"Node feature dimension: {node_feature_dim}")
    print(f"Number of targets: {num_targets}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Print dataset sizes
    train_size = len(train_loader) * BATCH_SIZE
    val_size = len(val_loader.dataset) if hasattr(val_loader, 'dataset') else len(val_loader) * BATCH_SIZE
    test_size = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else len(test_loader) * BATCH_SIZE
    print(f"Approximate dataset sizes - Train: {train_size:,}, Val: {val_size:,}, Test: {test_size:,}")

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
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=True,
        threshold=1e-4,
        cooldown=3
    )

    # 4. Training Loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 15
    
    # Define feature names
    feature_names = ['CYCLES', 'FF', 'LUT', 'BRAM', 'DSP', 'II'] 

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs", unit="epoch", disable=True):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validation
        val_loss, val_predictions, val_targets = evaluate(
            model, val_loader, criterion, device, dataset, denormalize=False
        )
        val_losses.append(val_loss)
        
        # Step scheduler with validation loss
        scheduler.step(val_loss)
        
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
        # if epoch % 1 == 0 or epoch == NUM_EPOCHS - 1:
        #     current_lr = optimizer.param_groups[0]['lr']
        #     print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6e}, Val Loss = {val_loss:.6e}, LR = {current_lr:.6e}")

        # if epoch % 2 == 0 or epoch == NUM_EPOCHS - 1:
        #     test_loss_norm, test_pred, test_targets = evaluate(
        #         model, test_loader, criterion, device, dataset, denormalize=True
        #     )
        #     test_loss = test_loss_norm.item() if isinstance(test_loss_norm, torch.Tensor) else test_loss_norm
        #     print(f"\n     Epoch {epoch:3d} - Test Loss (denormalized): {test_loss:.6e}")
        #     test_metrics = calculate_metrics(test_pred, test_targets, feature_names)

        #     # Format overall metrics with scientific notation
        #     overall_formatted = {k: f"{v:.4e}" for k, v in test_metrics['overall'].items()}
        #     print(f"     overall metrics: {overall_formatted}")
        if epoch % 1 == 0 or epoch == NUM_EPOCHS - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6e}, Val Loss = {val_loss:.6e}, LR = {current_lr:.6e}")

        # Test reports every 5 epochs
        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            test_loss_norm, test_pred, test_targets = evaluate(
                model, test_loader, criterion, device, dataset, denormalize=True
            )
            test_loss = test_loss_norm.item() if isinstance(test_loss_norm, torch.Tensor) else test_loss_norm
            print(f"    --> TEST Epoch {epoch:3d} - Test Loss (denormalized): {test_loss:.4e}")
            test_metrics = calculate_metrics(test_pred, test_targets, feature_names)
            overall_formatted = {k: f"{v:.3e}" for k, v in test_metrics['overall'].items()}
            print(f"    --> overall metrics: {overall_formatted}")
        
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
    print("FINAL EVALUATION - FPGA GNN Model")
    print("="*50)

    # Test set evaluation
    test_loss_norm, test_pred, test_targets = evaluate(
        model, test_loader, criterion, device, dataset, denormalize=True
    )

    test_metrics = calculate_metrics(test_pred, test_targets, feature_names)
    
    # Display test metrics in organized format
    print(f"\nTest Set Metrics (denormalized):")
    print(f"Overall Metrics:")
    for metric, value in test_metrics['overall'].items():
        if metric in ['MAPE', 'SMAPE']:
            print(f"  {metric}: {value:.2f}%")
        else:
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nPer-Feature Metrics:")
    for feature, metrics in test_metrics['per_feature'].items():
        print(f"\n{feature}:")
        for metric, value in metrics.items():
            if metric in ['MAPE', 'SMAPE']:
                print(f"  {metric}: {value:.2f}%")
            else:
                print(f"  {metric}: {value:.4f}")

    # 7. Save metrics to text file
    print("\nSaving metrics to file...")
    model_config_for_file = {
        'node_feature_dim': node_feature_dim,
        'num_targets': num_targets,
        'hidden_dim': GNN_HIDDEN_DIM,
        'num_gnn_layers': GNN_NUM_LAYERS,
        'num_attention_heads': NUM_ATTENTION_HEADS,
        'mlp_hidden_dim': MLP_HIDDEN_DIM,
        'dropout_rate': DROPOUT_RATE,
        'model_type': 'FPGA_GNN_GATv2_Enhanced' if use_enhanced_model else 'FPGA_GNN_GATv2'
    }
    
    training_config_for_file = {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'final_epochs': epoch + 1,
        'weight_decay': WEIGHT_DECAY,
        'best_val_loss': best_val_loss,
        'dataset_sizes': {
            'train': 440650,  # From your output
            'val': 94537,
            'test': 94430
        }
    }

    folder_base = SAVE_FOLDERNAME.split('/', 1)[1] if '/' in SAVE_FOLDERNAME else SAVE_FOLDERNAME
    
    save_metrics_to_file(
        metrics=test_metrics,
        output_dir=output_dir,
        model_config=model_config_for_file,
        training_config=training_config_for_file,
        name_base= folder_base,
    )

    # 8. Generate all plots
    print("\nGenerating plots...")
    plot_metrics = generate_all_plots(
        model=model,
        dataset=dataset,
        train_loader=train_loader,
        test_loader=test_loader,
        train_losses=train_losses,
        val_losses=val_losses,
        output_dir=output_dir,
        device=device
    )

    # 9. Save final model
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
            'model_type': 'FPGA_GNN_GATv2_Enhanced' if use_enhanced_model else 'FPGA_GNN_GATv2'
        },
        'training_config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': epoch + 1,
            'weight_decay': WEIGHT_DECAY
        },
        'best_val_loss': best_val_loss,
        'test_metrics': test_metrics,
        'plot_metrics': plot_metrics,
        'feature_names': feature_names,
        'dataset_info': {
            'train_size': 440650,
            'val_size': 94537,
            'test_size': 94430,
            'max_layers': 51,  # From your train data
            'num_features': 18
        }
    }, final_model_path)
    print(f"\nModel saved to: {final_model_path}")

    return model, dataset, train_losses, val_losses, test_metrics


if __name__ == "__main__":
    # folder_name = 'results/y_01_GAT_simple_bigger'
    folder_name = SAVE_FOLDERNAME

    # Create results folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Train with the new pre-split data approach
    model, dataset, train_losses, val_losses, test_metrics = train_gatv2_gnn(
        output_dir=folder_name, 
        use_enhanced_model=True
    )
    
    print(f"\nTraining completed! Results saved to: {folder_name}")
