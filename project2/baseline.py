from sklearn.linear_model import LogisticRegression
import sklearn.metrics


class BaselineModel:
    def __init__(self):
        self.model = LogisticRegression(
            random_state=42, penalty=None, max_iter=100, n_jobs=-1
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        # get final training loss
        y_hat = self.model.predict(X_train)
        return sklearn.metrics.log_loss(y_train, y_hat)
        # | || | |_
