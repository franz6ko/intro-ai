import numpy as np
from sklearn.preprocessing import PolynomialFeatures
# see: https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/


class MiniBatchGradientDescent:

    def __init__(self, alpha, n_epochs, n_batches, poly=None):
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.model = None
        self.n_batches = n_batches

        if poly is not None:
            self.poly = PolynomialFeatures(poly)
        else:
            self.poly = None

    def fit(self, x, y):

        if self.poly is not None:
            x = self.poly.fit_transform(x.reshape(-1, 1))

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n_samples = x.shape[0]
        n_features = x.shape[1]

        # 1 - Random w initialization
        w = np.random.random(n_features)

        for epoch in range(self.n_epochs):
            # Shuffle samples and create batches
            batch_size = int(n_samples / self.n_batches)
            idx = np.random.permutation(n_samples)
            x_sh = x[idx]
            y_sh = y[idx]

            for i in range(self.n_batches):
                bx = x_sh[i * batch_size:(i + 1) * batch_size]
                by = y_sh[i * batch_size:(i + 1) * batch_size]
                w = w - self.alpha * (-2 / n_samples) * np.sum((by - bx @ w)[:, np.newaxis] * bx, axis=0)

        self.model = w

    def predict(self, x):

        if self.poly is not None:
            x = self.poly.fit_transform(x.reshape(-1, 1))

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        return x @ self.model

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.predict(x).reshape(1, -1)
