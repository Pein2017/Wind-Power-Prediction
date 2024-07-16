import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

plt.switch_backend("agg")


# Utility function to get the scaler
def get_scaler(scaler_type):
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "min_max":
        return MinMaxScaler()
    # Add other scalers here if needed
    else:
        return None


# Utility function to fit and transform the data
def fit_transform(scaler, data):
    if scaler:
        return scaler.fit_transform(data)
    else:
        return data


# Utility function to inverse transform the data
def inverse_transform(scaler, data):
    if scaler:
        return scaler.inverse_transform(data)
    else:
        return data


# Utility function to transform the data using the fitted scaler
def transform(scaler, data):
    if scaler:
        return scaler.transform(data)
    else:
        return data


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: args.learning_rate
            if epoch < 3
            else args.learning_rate * (0.9 ** ((epoch - 3) // 1))
        }
    elif args.lradj == "PEMS":
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == "TST":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if printout:
            print("Updating learning rate to {}".format(lr))


def visual(true, preds=None, name_base="./pic/test", skip_plot=False):
    """
    Results visualization
    skip_plot for greed search
    """
    if skip_plot:
        return

    # Plot together
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(
        true, label="GroundTruth", color="blue", linewidth=1, marker="o", markersize=3
    )
    if preds is not None:
        plt.plot(
            preds,
            label="Prediction",
            color="red",
            linestyle="--",
            linewidth=1,
            marker="s",
            markersize=3,
        )
    plt.legend()

    if preds is not None:
        # Calculate residuals
        residuals = np.array(true) - np.array(preds)
        plt.subplot(2, 1, 2)
        plt.plot(
            residuals,
            label="Residual",
            color="green",
            linestyle="-",
            linewidth=1,
            marker="x",
            markersize=3,
        )
        plt.title("Residual")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{name_base}_together.pdf", bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {name_base}_together.pdf")

    # Plot separately
    plt.figure(figsize=(10, 12))
    plt.subplot(3, 1, 1)
    plt.plot(
        true, label="GroundTruth", color="blue", linewidth=1, marker="o", markersize=3
    )
    plt.title("GroundTruth")
    plt.legend()

    if preds is not None:
        plt.subplot(3, 1, 2)
        plt.plot(
            preds,
            label="Prediction",
            color="red",
            linestyle="--",
            linewidth=1,
            marker="s",
            markersize=3,
        )
        plt.title("Prediction")
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(
            residuals,
            label="Residual",
            color="green",
            linestyle="-",
            linewidth=1,
            marker="x",
            markersize=3,
        )
        plt.title("Residual")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{name_base}_separate.pdf", bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {name_base}_separate.pdf")


class EarlyStopping:
    def __init__(self, early_stop_patience=7, verbose=False, delta=0):
        self.early_stop_patience = early_stop_patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.early_stop_patience}"
            )
            if self.counter >= self.early_stop_patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        # torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


def get_next_version(output_dir):
    """Get the next version number for the logger."""
    version = 0
    if os.path.exists(output_dir):
        existing_versions = [
            int(d.split("_")[-1])
            for d in os.listdir(output_dir)
            if d.startswith("version_") and d.split("_")[-1].isdigit()
        ]
        if existing_versions:
            version = max(existing_versions) + 1
    return version
