import numpy as np
import matplotlib.pyplot as plt
from PolynomialRegression import PolynomialRegression
from MSE import MSE
from sklearn.preprocessing import PolynomialFeatures
from GradientDescent import GradientDescent
from StochasticGradientDescent import StochasticGradientDescent
from MiniBatchGradientDescent import MiniBatchGradientDescent

# Dataset definition
amt_points = 36
x = np.linspace(0, 360, num=amt_points)
y = np.sin(x * np.pi / 180.)
# y = x**5

noise = np.random.normal(0, .1, y.shape)
noisy_y = y + noise

x_train = x
y_train = noisy_y

# Polynomial regression
limit_grade = 15
res = np.zeros((limit_grade + 1, len(x)))
mse = np.zeros(limit_grade + 1)
mse_eval = MSE()

# n_epochs = 100
# res_g = np.zeros((limit_grade + 1, len(x)))
# res_s = np.zeros((limit_grade + 1, len(x)))
# res_m = np.zeros((limit_grade + 1, len(x)))
# mse_g = np.zeros(limit_grade + 1)
# mse_s = np.zeros(limit_grade + 1)
# mse_m = np.zeros(limit_grade + 1)

for i in range(limit_grade + 1):
    p = PolynomialRegression(i)
    res[i, :] = p.fit_transform(x_train, y_train)
    mse[i] = mse_eval(y_train, res[i, :])

    # g = GradientDescent(1e-50, n_epochs, i)
    # res_g[i, :] = g.fit_transform(x_train, y_train)
    # mse_g[i] = mse_eval(y_train, res_g[i, :])

    # s = StochasticGradientDescent(0.001, n_epochs, i)
    # res_s[i, :] = s.fit_transform(x_train, y_train)
    # mse_s[i] = mse_eval(y_train, res_s[i, :])
    #
    # m = MiniBatchGradientDescent(1e-10, n_epochs, 4, i)
    # res_m[i, :] = m.fit_transform(x_train, y_train)
    # mse_s[i] = mse_eval(y_train, res_m[i, :])

# Plot results
plt.figure(0, figsize=(8, 8))
plt.title("Test results")
plt.scatter(x_train, y_train, label='Test dataset')
show_items = [1, 2, 3, 5, 10]
for i in show_items:
    plt.plot(x_train, res[i, :], label='Grade ' + str(i))
plt.legend()

# Plot MSE
plt.figure(1, figsize=(8, 8))
plt.title("Test MSE results")
plt.plot(np.arange(limit_grade + 1), mse, label='Analytic')
plt.legend()
plt.show()
