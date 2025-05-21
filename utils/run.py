# run.py

import torch
from data import load_and_preprocess_data, get_first_non_padded_layer
from model import TransformerRegressor
from train import train_model, test_model, calculate_metrics
from plot import plot_loss, plot_box_plots_symlog, plot_results
import numpy as np
import os

def main():
    # Config
    data_dir = "../../../../ddemler/dima_stuff/wa_remake/May_15_processed"
    max_samples = 10000 # Do a small set for testing the process
    batch_size = 512
    num_epochs = 10
    learning_rate = 1e-1 # loss is huge right now, can make smaller by normalizing the labels
    output_features = ["WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls", "BRAM_18K_hls", "DSP_hls"]
    outdir = "results"

    # Data
    ( train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, feat_np_raw, label_np) = load_and_preprocess_data(
        data_dir=data_dir,
        max_samples=max_samples,
        batch_size=batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    # Training
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=num_epochs
    )

    # Plot Loss
    plot_loss(train_losses, val_losses, outdir=os.path.join(outdir, "plots"))

    # Test Evaluation
    y_true, y_pred = test_model(model, test_loader, device)
    calculate_metrics(y_true, y_pred)

    # Boxplot
    plot_box_plots_symlog(y_pred, y_true, folder_name=outdir)

    # Prepare for plotting results
    # Get X_raw_test (shape: N_test, 18, 16) and reduce to (N_test, 16)
    test_indices = test_ds.indices
    X_raw_test = feat_np_raw[test_indices]
    X_raw_test_1layer = np.stack([get_first_non_padded_layer(arr) for arr in X_raw_test])

    # Scatterplots
    plot_results(
        name="run1",
        mpl_plots=True,
        y_test=y_true,
        y_pred=y_pred,
        X_raw_test=X_raw_test_1layer,
        output_features=output_features,
        folder_name=outdir
    )

if __name__ == "__main__":
    main()