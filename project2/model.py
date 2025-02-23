import numpy as np
from logging import Logger
from scipy.special import expit
from scipy import sparse
from sklearn.metrics import log_loss

def calculate_stable_hessian(X, y_hat, eps=1e-15):
    # Clip probabilities to prevent numerical instability
    y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
    
    # Calculate weights (diagonal terms)
    weights = y_hat_clipped * (1 - y_hat_clipped)
    
    # Ensure weights are positive and not too close to zero
    weights = np.maximum(weights, eps)
    
    # Method 1: Using np.einsum for better numerical precision
    hessian = np.einsum('ji,j,jk->ik', X, weights, X)
    
    # Alternative Method 2: Using sparse diagonal matrix if data is large
    
    W = sparse.diags(weights, format='csr')
    hessian = X.T @ W @ X
    
    return hessian

# Example usage:
"""
# Assuming X_train and y_hat are your training data and predictions:
hessian = calculate_stable_hessian(X_train, y_hat)
"""

class CDLogisticRegressor:
    """
    Implements logistic regression with co-ordinate descent
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        logger: Logger,
        alpha: float = 0.5,
        n_iter: int = 100,
        wt_update: str = "newton",
        seed:int = 42
    ):
        """
        Implement logistic regression using co-ordinate descent to optimize the loss
        """
        np.random.seed(seed)
        self.d = X_train.shape[-1]  # dimensionality of the model
        self.w = np.random.normal(0, 0.1, self.d)
        self.X_train = X_train
        self.y_train = y_train
        self.logger = logger
        self.alpha = alpha
        self.n_iter = n_iter
        self.wt_update = wt_update

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
        self.hessian = calculate_stable_hessian(self.X_train, y_hat) 

        idx = self.select_feature()

        # Newton update of co-ordinate
        if self.wt_update == "newton":
            self.w[idx] -=  (np.linalg.inv(self.hessian) @ self.gradients)[idx]
        elif self.wt_update == "gd":
            # self.logger.info(f"Using alpha= {self.alpha}")
            self.w[idx] -= self.alpha * self.gradients[idx]

    def learn(self):
        losses = [self.loss(self.X_train, self.y_train)]
        # self.logger.info(f"STARTING LOSS: {losses[0]}")
        for i in range(self.n_iter):
            self.update_weights()
            loss = self.loss(self.X_train, self.y_train)
            losses.append(loss)
        return losses


class RandomFeatureModel(CDLogisticRegressor):
    """
    Implements logistic regression with random-feature co-ordinate descent
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        logger: Logger,
        n_iter: int = 100,
        wt_update: str = "newton",
        alpha: float = 0.5,
        seed: int = 42
    ):
        """
        Implement logistic regression using co-ordinate descent to optimize the loss
        """
        super().__init__(X_train, y_train, logger, alpha=alpha, n_iter=n_iter, wt_update=wt_update, seed=seed)

    def select_feature(self):
        return np.random.randint(0, self.d)


class CustomModel(CDLogisticRegressor):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        logger: Logger,
        n_iter: int,
        wt_update: str,
        alpha: float = 0.5,
        seed: int = 42
    ):
        super().__init__(X_train, y_train,logger, alpha=alpha, n_iter=n_iter, wt_update=wt_update, seed=seed)

    def select_feature(self):
        """
        Return the index of the feature having the largest absolute value of gradient
        """
        grads_abs = np.abs(self.gradients)
        return np.argmax(grads_abs)  # Index having the largest gradient
