import numpy as np
from logging import Logger
from scipy.special import expit
from sklearn.metrics import log_loss

class RFLogisticRegression:
    """
    Implements logistic regression with random-feature co-ordinate descent
    """
    def __init__(
        self,
        X: np.ndarray,
        y : np.ndarray,
        logger : Logger
    ):
        """
        Implement logistic regression using co-ordinate descent to optimize the loss
        """
        self.d = X.shape[-1]  #dimensionality of the model
        self.w = np.random.normal(0, 0.1, self.d)
        self.X = X
        self.y = y
        self.alpha = 0.5
        self.logger = logger

    def loss(self):
        """
        Compute cross-entropy loss and return
        """
        eps = 1e-10
        p_hat = expit(self.X @ self.w)
        y_hat = np.where(p_hat >= 0.5, 1, 0)
        return log_loss(self.y, y_hat)

    def update_weights(self):
        """
        Update weights using co-ordinate descent
        """
        y_hat = 1 / (1 + np.exp(-self.X @ self.w))
        gradients = np.array([np.sum(self.X[:, i]@(y_hat - self.y)) for i in range(self.d)])
        # print(gradients)
        # mg_idx = np.argmax(gradients) # Index having the largest gradient
        idx = np.random.randint(0, self.d)
        self.w[idx] -= gradients[idx]*self.alpha
    
    def learn(self):
        for i in range(100):
            self.update_weights()
            loss = self.loss()
            self.logger.info(f"Loss at iteration {i} : {loss}")
