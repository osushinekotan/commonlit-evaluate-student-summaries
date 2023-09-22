import numpy as np


class MCRMSEScore:
    def __call__(self, outputs, targets):
        """Compute the Mean Columnwise Root Mean Square score (negative)."""
        columnwise_rmse = np.sqrt(np.mean(np.square(targets - outputs), axis=0))
        return -np.mean(columnwise_rmse)
