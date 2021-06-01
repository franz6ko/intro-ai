import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class GradientDescent:

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
        np.seterr(all='warn')

        for epoch in range(self.n_epochs):
            # 2 - Prediction
            y_hat = x @ w
            # 3 - Error
            e = y - y_hat
            # 4 - Gradient
            o = np.zeros((n_samples, n_features))
            for i in range(n_samples):
                o[i, :] = e[i] * x[i, :]
            g = -2 * np.sum(o, axis=0) / n_samples
            # 5 - Correction
            w = w - self.alpha * g

            # 2-5 condensed version
            # for epoch in range(n_epochs):
            #     w = w - self.alpha * (-2 / n_samples) * np.sum((y - x @ w)[:, np.newaxis] * x, axis=0)

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
