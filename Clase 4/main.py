import numpy as np
import matplotlib.pyplot as plt
from PolynomialRegression import PolynomialRegression
from MSE import MSE

# Dataset definition
amt_points = 36
x = np.linspace(0, 360, num=amt_points)
y = np.sin(x * np.pi / 180.)
noise = np.random.normal(0, .1, y.shape)
noisy_y = y + noise
X_train = x
y_train = noisy_y

# Polynomial regression
limit_grade = 10
res = np.zeros((limit_grade + 1, len(x)))
mse = np.zeros(limit_grade + 1)
mse_eval = MSE()

for i in range(limit_grade + 1):
    p = PolynomialRegression(i)
    p.fit(X_train, y_train.reshape(-1, 1))
    res[i, :] = p.predict(X_train).reshape(1, -1)
    mse[i] = mse_eval(y_train, res[i, :])

# Plot results
plt.figure(0, figsize=(8, 8))
plt.title("Test results")
plt.scatter(X_train, y_train, label='Test dataset')
show_items = [1, 2, 3, 10]
for i in show_items:
    plt.plot(X_train, res[i, :], label='Grade ' + str(i))
plt.legend()

# Plot MSE
plt.figure(1, figsize=(8, 8))
plt.title("Test MSE results")
plt.plot(np.arange(limit_grade + 1), mse)
plt.show()

