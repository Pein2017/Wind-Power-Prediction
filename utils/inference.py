import numpy as np
import torch

from utils.metrics import AccuracyMetricLoss, metric


def full_inference(
    model, train_loader, val_loader, test_loader, criterion, custom_scaler, scale_y_flag
):
    model.eval()

    def invert_and_compute_metrics(loader):
        model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for batch in loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = model._prepare_batch(
                    batch
                )
                dec_inp = None
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                all_preds.append(outputs.cpu().numpy())
                all_trues.append(batch_y.cpu().numpy())
        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)

        if scale_y_flag and custom_scaler:
            preds_reshaped = preds.reshape(-1, 1)
            trues_reshaped = trues.reshape(-1, 1)

            inversed_preds = custom_scaler.inverse_transform(preds_reshaped).flatten()
            inversed_trues = custom_scaler.inverse_transform(trues_reshaped).flatten()

            inversed_preds = np.maximum(inversed_preds, 0)
            inversed_trues = np.maximum(inversed_trues, 0)
        else:
            inversed_preds, inversed_trues = preds, trues

        mae, mse, rmse, mape, mspe = metric(inversed_preds, inversed_trues)
        custom_acc = AccuracyMetricLoss(device=model.device)
        custom_acc_value = custom_acc(inversed_preds, inversed_trues)
        return rmse, custom_acc_value

    train_rmse, train_custom_acc = invert_and_compute_metrics(train_loader)
    val_rmse, val_custom_acc = invert_and_compute_metrics(val_loader)
    test_rmse, test_custom_acc = invert_and_compute_metrics(test_loader)

    model.train()

    return (
        train_rmse,
        train_custom_acc,
        val_rmse,
        val_custom_acc,
        test_rmse,
        test_custom_acc,
    )


# def inverse_transform(preds, trues, scaler):
#     preds_reshaped = preds.reshape(-1, 1)
#     trues_reshaped = trues.reshape(-1, 1)

#     non_nan_mask = ~np.isnan(trues_reshaped)

#     preds_inverse = np.empty_like(preds_reshaped)
#     trues_inverse = np.empty_like(trues_reshaped)

#     preds_inverse[non_nan_mask] = scaler.inverse_transform(
#         preds_reshaped[non_nan_mask].reshape(-1, 1)
#     ).flatten()
#     trues_inverse[non_nan_mask] = scaler.inverse_transform(
#         trues_reshaped[non_nan_mask].reshape(-1, 1)
#     ).flatten()

#     preds_inverse[~non_nan_mask] = np.nan
#     trues_inverse[~non_nan_mask] = np.nan

#     inversed_preds = preds_inverse.flatten()
#     inversed_trues = trues_inverse.flatten()

#     return inversed_preds, inversed_trues


"""
deprecated code
"""


# def evaluate_and_log_metrics(model, loss, phase):
#     if phase == "train":
#         model.min_train_loss = min(model.min_train_loss, loss)
#         model.best_train_epoch = (
#             model.current_epoch + 1
#             if model.min_train_loss == loss
#             else model.best_train_epoch
#         )
#     elif phase == "val":
#         model.min_vali_loss = min(model.min_vali_loss, loss)
#         model.best_val_epoch = (
#             model.current_epoch + 1
#             if model.min_vali_loss == loss
#             else model.best_val_epoch
#         )
#     elif phase == "test":
#         model.min_test_loss = min(model.min_test_loss, loss)
#         model.best_test_epoch = (
#             model.current_epoch + 1
#             if model.min_test_loss == loss
#             else model.best_test_epoch
#         )

#     metrics_updated = model.current_epoch + 1 in {
#         model.best_train_epoch,
#         model.best_val_epoch,
#         model.best_test_epoch,
#     }
#     if metrics_updated:
#         train_loader = model.train_dataloader()
#         val_loader = model.val_dataloader()
#         test_loader = model.test_dataloader()
#         (
#             train_rmse,
#             train_custom_acc,
#             val_rmse,
#             val_custom_acc,
#             test_rmse,
#             test_custom_acc,
#         ) = full_inference(
#             model,
#             train_loader,
#             val_loader,
#             test_loader,
#             model.criterion,
#             model.train_set.scaler_y,
#             model.train_set.scale_y_flag,
#         )

#         if phase == "train":
#             model.best_train_rmse = min(model.best_train_rmse, train_rmse)
#             model.best_train_custom_acc = min(
#                 model.best_train_custom_acc, train_custom_acc
#             )
#             model.best_val_rmse_for_train = min(model.best_val_rmse_for_train, val_rmse)
#             model.best_val_custom_acc_for_train = min(
#                 model.best_val_custom_acc_for_train, val_custom_acc
#             )
#             model.best_test_rmse_for_train = min(
#                 model.best_test_rmse_for_train, test_rmse
#             )
#             model.best_test_custom_acc_for_train = min(
#                 model.best_test_custom_acc_for_train, test_custom_acc
#             )
#         elif phase == "val":
#             model.best_train_rmse_for_val = min(
#                 model.best_train_rmse_for_val, train_rmse
#             )
#             model.best_train_custom_acc_for_val = min(
#                 model.best_train_custom_acc_for_val, train_custom_acc
#             )
#             model.best_val_rmse = min(model.best_val_rmse, val_rmse)
#             model.best_val_custom_acc = min(model.best_val_custom_acc, val_custom_acc)
#             model.best_test_rmse_for_val = min(model.best_test_rmse_for_val, test_rmse)
#             model.best_test_custom_acc_for_val = min(
#                 model.best_test_custom_acc_for_val, test_custom_acc
#             )
#         elif phase == "test":
#             model.best_train_rmse_for_test = min(
#                 model.best_train_rmse_for_test, train_rmse
#             )
#             model.best_train_custom_acc_for_test = min(
#                 model.best_train_custom_acc_for_test, train_custom_acc
#             )
#             model.best_val_rmse_for_test = min(model.best_val_rmse_for_test, val_rmse)
#             model.best_val_custom_acc_for_test = min(
#                 model.best_val_custom_acc_for_test, val_custom_acc
#             )
#             model.best_test_rmse = min(model.best_test_rmse, test_rmse)
#             model.best_test_custom_acc = min(
#                 model.best_test_custom_acc, test_custom_acc
#             )

#         print(
#             f"Epoch {model.current_epoch + 1} | train loss: {model.min_train_loss:.5f} | val loss: {model.min_vali_loss:.5f} | test loss: {model.min_test_loss:.5f}"
#         )
