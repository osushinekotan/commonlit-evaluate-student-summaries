import numpy as np
import torch
import torch.nn as nn


class MCRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        # Check that the predictions and targets have the same shape
        assert preds.shape == targets.shape, "Predictions and targets must have the same shape"

        # Calculate column-wise RMSE
        mse = torch.mean((preds - targets) ** 2, dim=0)  # column-wise MSE
        rmse = torch.sqrt(mse)  # column-wise RMSE

        # Calculate the mean of the column-wise RMSE
        mcrmse = torch.mean(rmse)

        return mcrmse


def mcrmse_score(outputs, targets):
    """Compute the Mean Columnwise Root Mean Square score (negative)."""
    columnwise_rmse = np.sqrt(np.mean(np.square(targets - outputs), axis=0))
    return -np.mean(columnwise_rmse)
