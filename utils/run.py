# run.py

import torch # type: ignore
# from data import load_and_preprocess_data, get_first_non_padded_layer
# from GNN.Dataset import *
from GNN.Dataset2 import * # updated to load the split data
from model import TransformerRegressor
from train import train_model, test_model, calculate_metrics, calculate_metrics_per_feature
from plot import plot_loss, plot_box_plots_symlog, plot_results_simplified
import numpy as np # type: ignore
import os
import argparse
from datetime import datetime

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["gnn", "transformer"], default="transformer", help="Architecture type")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate and plot, no training.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to saved model checkpoint (for eval-only mode).")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    args = parser.parse_args()    

    timestamp = datetime.now().strftime("%m_%d_%H_%M")

    num_epochs = args.epochs
    learning_rate = args.lr
    BATCH_SIZE = args.batch_size
    USE_LOG_TRANSFORM = True  # Set to True to enable log transformation
    LOG_EPSILON = 1e-6       # Small value to add before log transform

    if args.eval_only:
        outdir = f"new_results_plots/testing_only_{timestamp}_{num_epochs}epochs_{learning_rate}lr_{BATCH_SIZE}bs"
    else:
        outdir = f"new_results_plots/{timestamp}_{num_epochs}epochs_{learning_rate}lr_{BATCH_SIZE}bs"

    # Config
    output_features = ['CYCLES', 'FF', 'LUT', 'BRAM', 'DSP', 'II']
    # outdir = "results_and_plots/5_31_results_2_epochs_TESTING"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # JOB
    # # image_data_path = '../../app/'
    # image_data_path = '/app/dataset/May_29_complete/'
    # FEATURES_PATH = os.path.join(image_data_path, FEATURES_PATH)
    # LABELS_PATH = os.path.join(image_data_path, LABELS_PATH)
    # STATS_PATH = os.path.join(image_data_path, STATS_PATH)
    # SPLIT_MAPPING_PATH = os.path.join(image_data_path, SPLIT_MAPPING_PATH)

    # train_loader, val_loader, test_loader, dataset, node_feature_dim, num_targets = create_dataloaders(
    #         feature_path=FEATURES_PATH,
    #         labels_path=LABELS_PATH,
    #         stats_load_path=STATS_PATH,
    #         stats_save_path=STATS_PATH,
    #         split_mapping_path=SPLIT_MAPPING_PATH,  # NEW: Add this parameter
    #         batch_size=BATCH_SIZE,
    #         train_val_test_split=(0.7, 0.15, 0.15),
    #         random_seed=42,
    #         num_workers=4, # dropped from 5
    #         pin_memory=True if device.type == 'cuda' else False
    #     )

## CHANGE BEFORE PUSH
    # base_dir = "../dataset/output/split_dataset/result/result/"  # UPDATE FOR THE JOB WITH NEW PATH
    base_dir = "/jason-pvc/june_wa-hls4ml/result/" # IN THE PVC

    train_loader, val_loader, test_loader, node_feature_dim, num_targets = create_dataloaders_from_split_data(
            train_features_path=os.path.join(base_dir, "train_features.npy"),
            train_labels_path=os.path.join(base_dir, "train_labels.npy"),
            val_features_path=os.path.join(base_dir, "val_features.npy"),
            val_labels_path=os.path.join(base_dir, "val_labels.npy"),
            test_features_path=os.path.join(base_dir, "test_features.npy"),
            test_labels_path=os.path.join(base_dir, "test_labels.npy"),
            # stats_load_path=None,  # Will calculate from training data
            stats_load_path=os.path.join(base_dir, "lognormalization_stats.npy"),
            # stats_save_path=os.path.join(base_dir, "lognormalization_stats.npy"),  # Save for future use
            batch_size=BATCH_SIZE,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False,
            mode=args.arch, # different than Dataset2.py bc now already split and preprocessed
            use_log_transform=USE_LOG_TRANSFORM,  # Add this line
            log_epsilon=LOG_EPSILON                # Add this line
        )

    # Set mode on datasets!
    for loader in [train_loader, val_loader, test_loader]:
        loader.dataset.mode = args.arch

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
    # For denormalization, use the dataset, not the class
    y_true_denorm = test_loader.dataset.denormalize_labels(torch.tensor(y_true)).numpy()
    y_pred_denorm = test_loader.dataset.denormalize_labels(torch.tensor(y_pred)).numpy()

    # Use for metrics and plotting:
    calculate_metrics(y_true_denorm, y_pred_denorm)

    # After obtaining y_true_denorm and y_pred_denorm:
    metrics_per_feature = calculate_metrics_per_feature(y_true_denorm, y_pred_denorm, output_features)

    # ADDITION
    # Load test input features to get model types for coloring
    test_features_np = np.load(os.path.join(base_dir, "test_features.npy"))  # (num_samples, max_layers, num_features)

    # # Extract strategies for each model (0: latency, 1: resource)
    # def get_strategies(features_np):
    #     strategies = []
    #     for model in features_np:
    #         for layer in model:
    #             if not np.all(layer == -1):
    #                 strategies.append(int(layer[8]))  # 0=latency, 1=resource
    #                 break
    #         else:
    #             strategies.append(None)  # Handle fully padded model if ever
    #     return strategies

    # test_strategies = get_strategies(test_features_np)

    def get_model_types(features_np):
        model_types = []
        for model in features_np:
            valid_layers = [layer for layer in model if not np.all(layer == -1)]
            layer_types = [int(layer[9]) for layer in valid_layers] # model_type is the 9th index
            if any(lt == 2 for lt in layer_types):
                model_types.append('Conv1D')
            elif any(lt == 3 for lt in layer_types):
                model_types.append('Conv2D')
            else:
                model_types.append('Dense')
        return model_types

    test_model_types = get_model_types(test_features_np)
    # END ADDITION

    plot_box_plots_symlog(y_pred_denorm, y_true_denorm, folder_name=outdir)
    plot_results_simplified(
        name="run1",
        mpl_plots=True,
        y_test=y_true_denorm,
        y_pred=y_pred_denorm,
        output_features=output_features,
        folder_name=outdir,
        model_types=test_model_types # , ADDED
        # strategies=test_strategies # ADDED
    )

if __name__ == "__main__":
    main()

# nohup python run.py --arch transformer > new_logs/output1.log 2>&1 &

# # example of inference only:
# python run.py --arch transformer --eval-only --model-path new_results_plots/06_02_17_45_20epochs_0.0001lr_1024bs/best_model/model.pt

# nohup python run.py --arch transformer > 6_3_logs/test1.log 2>&1 &