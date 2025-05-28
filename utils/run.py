# run.py

import torch
# from data import load_and_preprocess_data, get_first_non_padded_layer
from GNN.Dataset import *
from model import TransformerRegressor
from train import train_model, test_model, calculate_metrics, calculate_metrics_per_feature
from plot import plot_loss, plot_box_plots_symlog, plot_results_simplified
import numpy as np
import os
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["gnn", "transformer"], default="transformer", help="Architecture type")
    args = parser.parse_args()    

    # Config
    # data_dir = "../../../../ddemler/dima_stuff/wa_remake/May_15_processed"
    # max_samples = 100000 # Do a small set for testing the process
    # batch_size = 512
    num_epochs = 2
    learning_rate = 1e-4
    # output_features = ["WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls", "BRAM_18K_hls", "DSP_hls"]
    output_features = ['CYCLES', 'FF', 'LUT', 'BRAM', 'DSP', 'II']
    outdir = "results_and_plots/TESTING_alldata_5_28_results_all_2_epochs"
    # outdir = "testing_all_output_features"

    # # Data
    # ( train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, feat_np_raw, label_np) = load_and_preprocess_data(
    #     data_dir=data_dir,
    #     max_samples=max_samples,
    #     batch_size=batch_size,
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FEATURES_PATH = '../../../../ddemler/dima_stuff/wa_gnn/May_15_processed/combined_features.npy'
    # LABELS_PATH = '../../../../ddemler/dima_stuff/wa_gnn/May_15_processed/combined_labels.npy'

    FEATURES_PATH = '../dataset/output/May_28_all/combined_features.npy'
    LABELS_PATH = '../dataset/output/May_28_all/combined_labels.npy'

    # STATS_PATH = 'stats/stats.npy'

    #SPLIT_MAPPING_PATH = '../../../../ddemler/dima_stuff/wa_gnn/May_15_processed/dataset_split_mapping_May26.npy'  # NEW: Add split mapping path

    BATCH_SIZE = 1024 # tried 512 and plots seemed fine, but with 2048 got a divide by zero (has to do with other stuff though)

    train_loader, val_loader, test_loader, dataset, node_feature_dim, num_targets = create_dataloaders(
            feature_path=FEATURES_PATH,
            labels_path=LABELS_PATH,
            #stats_load_path=STATS_PATH,
            stats_save_path=None,
            #split_mapping_path=SPLIT_MAPPING_PATH,  # NEW: Add this parameter
            batch_size=BATCH_SIZE,
            train_val_test_split=(0.7, 0.15, 0.15),
            random_seed=42,
            num_workers=5,
            pin_memory=True if device.type == 'cuda' else False
        )

    dataset.mode = args.arch

    model = TransformerRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    best_model_dir = os.path.join(outdir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    # Training
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=num_epochs, verbose=True, checkpoint_path=best_model_dir
    )

    # Plot Loss
    plot_loss(train_losses, val_losses, outdir=os.path.join(outdir, "plots"))

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