import numpy as np


def mcrms_score(outputs, targets):
    """Compute the Mean Columnwise Root Mean Square score (negative)."""
    columnwise_rmse = np.sqrt(np.mean(np.square(targets - outputs), axis=0))
    return -np.mean(columnwise_rmse)
