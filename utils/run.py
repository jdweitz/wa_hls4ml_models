# run.py

import torch
# from data import load_and_preprocess_data, get_first_non_padded_layer
# from GNN.Dataset import *
from GNN.Dataset2 import * # updated to load the split data
from model import TransformerRegressor
from train import train_model, test_model, calculate_metrics, calculate_metrics_per_feature
from plot import plot_loss, plot_box_plots_symlog, plot_results_simplified
import numpy as np
import os
import argparse
from datetime import datetime

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["gnn", "transformer"], default="transformer", help="Architecture type")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate and plot, no training.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to saved model checkpoint (for eval-only mode).")
    args = parser.parse_args()    

    timestamp = datetime.now().strftime("%m_%d_%H_%M")

    num_epochs = 2
    learning_rate = 1e-4
    BATCH_SIZE = 1024 

    if args.eval_only:
        outdir = f"results_and_plots/testing_only_{timestamp}_{num_epochs}epochs_{learning_rate}lr_{BATCH_SIZE}bs"
    else:
        outdir = f"results_and_plots/{timestamp}_{num_epochs}epochs_{learning_rate}lr_{BATCH_SIZE}bs"

    # Config
    # data_dir = "../../../../ddemler/dima_stuff/wa_remake/May_15_processed"
    # max_samples = 100000 # Do a small set for testing the process
    # batch_size = 512
    # output_features = ["WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls", "BRAM_18K_hls", "DSP_hls"]
    output_features = ['CYCLES', 'FF', 'LUT', 'BRAM', 'DSP', 'II']
    # outdir = "results_and_plots/5_31_results_2_epochs_TESTING"
    # outdir = "testing_all_output_features"

    # # Data
    # ( train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, feat_np_raw, label_np) = load_and_preprocess_data(
    #     data_dir=data_dir,
    #     max_samples=max_samples,
    #     batch_size=batch_size,
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOCAL
    # Configuration
    FEATURES_PATH = '../dataset/output/May_29_complete/combined_features.npy'
    LABELS_PATH = '../dataset/output/May_29_complete/combined_labels.npy'
    STATS_PATH = '../dataset/output/May_29_complete/calculated_TrainingOnly_norm_May26.npy'
    SPLIT_MAPPING_PATH = '../dataset/output/May_29_complete/dataset_split_mapping_May26.npy'  # NEW: Add split mapping path

    # JOB
    # # image_data_path = '../../app/'
    # image_data_path = '/app/dataset/May_29_complete/'
    # FEATURES_PATH = os.path.join(image_data_path, FEATURES_PATH)
    # LABELS_PATH = os.path.join(image_data_path, LABELS_PATH)
    # STATS_PATH = os.path.join(image_data_path, STATS_PATH)
    # SPLIT_MAPPING_PATH = os.path.join(image_data_path, SPLIT_MAPPING_PATH)

    train_loader, val_loader, test_loader, dataset, node_feature_dim, num_targets = create_dataloaders(
            feature_path=FEATURES_PATH,
            labels_path=LABELS_PATH,
            stats_load_path=STATS_PATH,
            stats_save_path=STATS_PATH,
            split_mapping_path=SPLIT_MAPPING_PATH,  # NEW: Add this parameter
            batch_size=BATCH_SIZE,
            train_val_test_split=(0.7, 0.15, 0.15),
            random_seed=42,
            num_workers=4, # dropped from 5
            pin_memory=True if device.type == 'cuda' else False
        )

    dataset.mode = args.arch

    model = TransformerRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    best_model_dir = os.path.join(outdir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    if args.eval_only:
        # Load model from checkpoint
        if args.model_path is None:
            model_path = os.path.join(best_model_dir, "model.pt")
        else:
            model_path = args.model_path
        print(f"Loading model weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # Training
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=num_epochs, verbose=True, checkpoint_path=best_model_dir
        )
        # Plot Loss
        plot_loss(train_losses, val_losses, outdir=os.path.join(outdir, "plots"))
        # Save best model for convenience
        model_path = os.path.join(best_model_dir, "model.pt")
        torch.save(model.state_dict(), model_path)

    # Test Evaluation
    y_true, y_pred = test_model(model, test_loader, device)
    # Denormalize:
    y_true_denorm = dataset.denormalize_labels(torch.tensor(y_true)).numpy()
    y_pred_denorm = dataset.denormalize_labels(torch.tensor(y_pred)).numpy()

    # Use for metrics and plotting:
    calculate_metrics(y_true_denorm, y_pred_denorm)

    # After obtaining y_true_denorm and y_pred_denorm:
    metrics_per_feature = calculate_metrics_per_feature(y_true_denorm, y_pred_denorm, output_features)

    plot_box_plots_symlog(y_pred_denorm, y_true_denorm, folder_name=outdir)
    plot_results_simplified(
        name="run1",
        mpl_plots=True,
        y_test=y_true_denorm,
        y_pred=y_pred_denorm,
        output_features=output_features,
        folder_name=outdir
    )

if __name__ == "__main__":
    main()