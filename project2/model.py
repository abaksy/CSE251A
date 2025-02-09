import numpy as np


class LogisticRegression:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        """
        Implement logistic regression using co-ordinate descent to optimize the loss
        """
        assert X_train.shape[-1] == X_test.shape[-1]
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        self.d = X_train.shape[-1]
        
