import numpy as np
from logging import Logger
from scipy.special import expit
from sklearn.metrics import log_loss


class CDLogisticRegressor:
    """
    Implements logistic regression with co-ordinate descent
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        logger: Logger,
        n_iter: int = 100,
    ):
        """
        Implement logistic regression using co-ordinate descent to optimize the loss
        """
        np.random.seed(42)
        self.d = X_train.shape[-1]  # dimensionality of the model
        self.w = np.random.normal(0, 0.1, self.d)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.alpha = 0.5
        self.logger = logger
        self.n_iter = n_iter

    def loss(self, X, y) -> float:
        """
        Compute cross-entropy loss and return
        """
        p_hat = expit(X @ self.w)
        y_hat = np.where(p_hat >= 0.5, 1, 0)
        return log_loss(y, y_hat)

    def select_feature(self) -> int:
        return 0

    def update_weights(self):
        """
        Update weights using co-ordinate descent
        """
        y_hat = expit(self.X_train @ self.w)
        
        # First order derivative of the loss function
        self.gradients = np.array(
            [np.sum(self.X_train[:, i] @ (y_hat - self.y_train)) for i in range(self.d)]
        )
        
        # Second order derivative of the loss function
        self.hessian = self.X_train.T @ np.diag(y_hat * (1 - y_hat)) @ self.X_train

        idx = self.select_feature()
        self.w[idx] -= (np.linalg.inv(self.hessian) @ self.gradients)[idx]

    def learn(self):
        self.logger.info(f"STARTING LOSS: {self.loss(self.X_train, self.y_train)}")
        for i in range(self.n_iter):
            self.update_weights()
            loss = self.loss(self.X_train, self.y_train)
            if (i + 1) % 10 == 0:
                self.logger.info(f"Loss at iteration {i} : {loss}")

    def predict(self):
        return np.where(expit(self.X_test @ self.w) >= 0.5, 1, 0)

    def accuracy(self):
        y_hat = self.predict()
        return np.mean(y_hat == self.y_test)


class RandomFeatureModel(CDLogisticRegressor):
    """
    Implements logistic regression with random-feature co-ordinate descent
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        logger: Logger,
        n_iter: int = 100,
    ):
        """
        Implement logistic regression using co-ordinate descent to optimize the loss
        """
        super().__init__(X_train, y_train, X_test, y_test, logger, n_iter)

    def select_feature(self):
        return np.random.randint(0, self.d)


class CustomModel(CDLogisticRegressor):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        logger,
        n_iter,
    ):
        super().__init__(X_train, y_train, X_test, y_test, logger, n_iter)

    def select_feature(self):
        grads_abs = np.abs(self.gradients)
        mg_idx = np.argmax(grads_abs)  # Index having the largest gradient
        return mg_idx
