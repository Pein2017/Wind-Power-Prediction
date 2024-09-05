import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from utils.tools import visual


def plot_results(preds, trues, plot_dir, suffix):
    visual(
        trues,
        preds,
        name_base=os.path.join(plot_dir, f"{suffix}_transform"),
        skip_plot=False,
    )


def collect_preds(model, dataloader):
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in dataloader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = model._prepare_batch(batch)
            dec_inp = None
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            all_preds.append(outputs.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)

    preds = preds.reshape(preds.shape[0])
    trues = trues.reshape(trues.shape[0])

    return preds, trues


def save_best_metrics_to_csv(best_metrics, csv_path, exp_settings):
    best_metrics = {
        k: (v.item() if isinstance(v, torch.Tensor) else v)
        for k, v in best_metrics.items()
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Parse experiment settings
    exp_settings_dict = {}
    settings_split = exp_settings.split("-")
    for i in range(0, len(settings_split), 2):
        key = settings_split[i]
        value = settings_split[i + 1]
        exp_settings_dict[key] = value

    # Get the current timestamp
    exp_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Create the dataframe
    df = pd.DataFrame([best_metrics])

    # Reorder columns to have exp_date first and experiment settings next
    exp_settings_df = pd.DataFrame([exp_settings_dict])
    exp_date_df = pd.DataFrame({"exp_date": [exp_date]})
    df = pd.concat([exp_date_df, df, exp_settings_df], axis=1)

    # Save to CSV
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    print(f"best metrics saved to {csv_path} successfully.")


def custom_test(
    trainer,
    model_class,
    data_module,
    exp_settings,
    device,
    best_metrics_dir,
    plot_dir=None,
    plot_scaled=False,
    config=None,
):
    test_loader = data_module.test_dataloader()

    if plot_dir is None:
        plot_dir = os.path.join(best_metrics_dir, "plots")
    else:
        plot_dir = os.path.join(plot_dir, "plots")

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(best_metrics_dir):
        os.makedirs(best_metrics_dir)

    # Load the best checkpoint
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        pl_module = model_class.load_from_checkpoint(best_model_path, config=config)
    else:
        print("No checkpoint found. Using the current model.")

    preds, trues = collect_preds(pl_module, test_loader)

    if preds.size == 0 or trues.size == 0:
        print("Error: preds or trues arrays are empty. Check data collection logic.")
        return

    # Plot the scaled data if plot_scaled is True
    if plot_scaled:
        plot_results(preds, trues, plot_dir, "scaled")

    # Inverse transform the predictions and true values to the original scale
    if data_module.scaler_y:
        # Reshape to 2D array before inverse transforming
        preds_reshaped = preds.reshape(-1, 1)
        trues_reshaped = trues.reshape(-1, 1)

        inversed_preds = data_module.inverse_transform(
            data_module.scaler_y, preds_reshaped, order=data_module.y_transform_order
        ).flatten()  # Flatten back to 1D after inverse transforming
        inversed_trues = data_module.inverse_transform(
            data_module.scaler_y, trues_reshaped, order=data_module.y_transform_order
        ).flatten()  # Flatten back to 1D after inverse transforming

        # Ensure the predictions and true values are non-negative
        inversed_preds = np.maximum(inversed_preds, 0)
        inversed_trues = np.maximum(inversed_trues, 0)

        # Plot the results in the original scale
        plot_results(inversed_preds, inversed_trues, plot_dir, "original")
    else:
        # If no scaler_y, treat preds and trues as original scale
        plot_results(preds, trues, plot_dir, "original")

    # Save the best metrics to a file
    best_metrics = trainer.model.best_metrics
    save_best_metrics_to_csv(
        best_metrics, os.path.join(best_metrics_dir, "metrics.csv"), exp_settings
    )
