import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
    ):
        """
        Implement logistic regression using co-ordinate descent to optimize the loss
        """

        self.d = df_train.shape[-1]
        self.w = np.zeros(self.d + 1)
        self.n_train = df_train.shape[0]
        self.X_train = np.hstack(np.ones(self.n_train), df_train.iloc[:, 1:].to_numpy())
        self.y_train = df_train.iloc[:, 0].to_numpy()
        self.X_test = np.hstack(np.ones(self.n_train), df_test.iloc[:, 1:].to_numpy())
        self.y_test = df_test.iloc[:, 0].to_numpy()

    def loss(self):
        """
        Compute cross-entropy loss and return
        """
        p_i = 1/(1 + np.exp(-self.X_train @ self.w))
        cross_entropy = -np.sum(self.y_train * np.log(p_i) + (1 - self.y_train) * np.log(1 - p_i))
        return cross_entropy


        
