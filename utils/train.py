# train.py

import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    num_epochs=10,
    verbose=True,
    checkpoint_path: str = None,
):
    best_val = float('inf')
    train_losses = []
    val_losses = []

    # make sure checkpoint dir exists
    if checkpoint_path is not None:
        os.makedirs(checkpoint_path, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for feats, pad_mask, labels in train_loader:
            feats, pad_mask, labels = feats.to(device), pad_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(feats, pad_mask)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * feats.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, pad_mask, labels in val_loader:
                feats, pad_mask, labels = feats.to(device), pad_mask.to(device), labels.to(device)
                preds = model(feats, pad_mask)
                val_loss += loss_fn(preds, labels).item() * feats.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if verbose:
            print(f"Epoch {epoch:2d} — Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        # Checkpoint if validation improved
        if checkpoint_path is not None and val_loss < best_val:
            best_val = val_loss
            ckpt_file = os.path.join(checkpoint_path, "model.pt")
            torch.save(model.state_dict(), ckpt_file)
            if verbose:
                print(f"  ↳ New best model (val={val_loss:.4f}), saved to {ckpt_file}")

    return train_losses, val_losses

def test_model(model, test_loader, device):
    y_true_list = []
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for feats, pad_mask, labels in test_loader:
            feats, pad_mask = feats.to(device), pad_mask.to(device)
            preds = model(feats, pad_mask).cpu()
            y_true_list.append(labels.cpu().numpy())
            y_pred_list.append(preds.numpy())
    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)
    return y_true, y_pred

def calculate_metrics(y_true, y_pred):
    '''Calculate MAE, MSE, RMSE, R^2, and SMAPE'''
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    smape = (100 / len(y_true)) * np.sum(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )
    print(f'Symmetric mean absolute percentage error (SMAPE): {smape:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'R-squared (R2 Score): {r2:.4f}')
    return dict(smape=smape, mae=mae, mse=mse, rmse=rmse, r2=r2)

def calculate_metrics_per_feature(y_true, y_pred, output_features=None):
    n_outputs = y_true.shape[1]
    metrics_dict = {}
    for i in range(n_outputs):
        print(f"\n--- Metrics for {output_features[i] if output_features else f'Feature {i}'} ---")
        metrics = calculate_metrics(y_true[:, i], y_pred[:, i])
        metrics_dict[output_features[i] if output_features else f'Feature_{i}'] = metrics
    return metrics_dict

## To use:
# from train import train_model, test_model, calculate_metrics

# train_losses, val_losses = train_model(
#     model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=10
# )
# y_true, y_pred = test_model(model, test_loader, device)
# metrics = calculate_metrics(y_true, y_pred)