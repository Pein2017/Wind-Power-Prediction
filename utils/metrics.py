import numpy as np
import torch
import torch.nn as nn


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAE(pred, true, threshold=1.0):
    mask = true >= threshold
    filtered_pred = pred[mask]
    filtered_true = true[mask]
    return np.mean(np.abs(filtered_pred - filtered_true))


def MAPE(pred, true, threshold=1.0):
    mask = true >= threshold
    filtered_pred = pred[mask]
    filtered_true = true[mask]
    mape = np.abs((filtered_pred - filtered_true) / filtered_true)
    mape = np.where(mape > 5, 0, mape)
    return np.mean(mape)


def MSPE(pred, true, threshold=1.0):
    mask = true >= threshold
    filtered_pred = pred[mask]
    filtered_true = true[mask]
    return np.mean(np.square((filtered_pred - filtered_true) / filtered_true))


def metric(pred, true):
    mask = ~np.isnan(true)

    nan_count = np.isnan(true).sum()
    if nan_count > 0:
        print(f"Number of NaN values in true: {nan_count} during metric calculation")

    pred = pred[mask]
    true = true[mask]

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


factor = (300 + 400 + 900) / 300 / 1000
CAP = 300400.0 * factor


class AccuracyMetricLoss(nn.Module):
    def __init__(self, device: torch.device, cap=CAP):
        super(AccuracyMetricLoss, self).__init__()
        self.cap = cap
        self.device = device

    def forward(self, pred, true):
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, device=self.device, dtype=torch.float32)
        if isinstance(true, np.ndarray):
            true = torch.tensor(true, device=self.device, dtype=torch.float32)

        assert (
            pred.shape == true.shape
        ), "Shape mismatch between prediction and ground truth arrays"

        num_complete_days = true.shape[0] // 96
        if num_complete_days == 0:
            raise ValueError(
                "Input data does not contain a complete day (96 time points)."
            )

        true = true[: num_complete_days * 96]
        pred = pred[: num_complete_days * 96]

        true = true.view(num_complete_days, 96)
        pred = pred.view(num_complete_days, 96)

        score = torch.zeros(num_complete_days, device=self.device, dtype=torch.float32)

        for i_date in range(num_complete_days):
            error_sum = 0.0
            for i_time in range(96):
                if true[i_date, i_time] > 0.2 * self.cap:
                    error_sum += (
                        (true[i_date, i_time] - pred[i_date, i_time])
                        / true[i_date, i_time]
                    ) ** 2
                else:
                    error_sum += (
                        (true[i_date, i_time] - pred[i_date, i_time]) / (0.2 * self.cap)
                    ) ** 2
            score[i_date] = (1 - torch.sqrt(error_sum / 96.0)) * 100.0

        mean_score = torch.mean(score)

        return mean_score
