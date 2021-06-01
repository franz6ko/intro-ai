import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class StochasticGradientDescent:

    def __init__(self, alpha, n_epochs, poly=None):
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.model = None

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
            # Shuffle the samples
            idx = np.random.permutation(n_samples)
            x_sh = x[idx]
            y_sh = y[idx]

            for i in range(n_samples):
                # 2 - Prediction
                y_hat = x_sh[i] @ w
                # 3 - Error
                e = y_sh[i] - y_hat
                # 4 - Gradient
                g = -2 * e * x_sh[i] / n_samples
                # 5 - Correction
                w = w - self.alpha * g

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