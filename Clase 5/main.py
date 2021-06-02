import numpy as np
import matplotlib.pyplot as plt
from PolynomialRegression import PolynomialRegression
from MSE import MSE
from GradientDescent import GradientDescent
from StochasticGradientDescent import StochasticGradientDescent
from MiniBatchGradientDescent import MiniBatchGradientDescent

# Dataset definition
amt_points = 36
x = np.linspace(0, 360, num=amt_points)
# y = x**5
# y = np.sin(x * np.pi / 180.)
y = np.sin(x * np.pi / 180.) + np.cos(x * 2 * np.pi / 180.) + np.cos(x * 3 * np.pi / 180.)
# y = np.exp(x * np.pi / 180.)

noise = np.random.normal(0, .1, y.shape)
noisy_y = y + noise

x_train = (x - x.mean()) / x.std()   # Normalization
y_train = noisy_y

# Polynomial regression
p_grade = 10
until_p_grade = False  # True: include all polynomial grades until p_grade | False: Only calculates p_grade
limit_grade = p_grade if until_p_grade else 0

res = np.zeros((4, limit_grade + 1, len(x)))
mse = np.zeros((4, limit_grade + 1))
mse_eval = MSE()

for i in range(limit_grade + 1):

    grade = i if until_p_grade else p_grade

    # Analytic Solution
    pr = PolynomialRegression(grade)

    # Numerical Solutions
    n_epochs = 2000
    learning_rate = 0.2
    reg_factor = 0.0001

    gr = GradientDescent(learning_rate, n_epochs, grade, reg_factor)
    st = StochasticGradientDescent(learning_rate, n_epochs, grade, reg_factor)
    bt = MiniBatchGradientDescent(learning_rate, n_epochs, 10, grade, reg_factor)

    res[0, i, :] = pr.fit_transform(x_train, y_train)
    res[1, i, :] = gr.fit_transform(x_train, y_train)
    res[2, i, :] = st.fit_transform(x_train, y_train)
    res[3, i, :] = bt.fit_transform(x_train, y_train)

    mse[0, i] = mse_eval(y_train, res[0, i, :])
    mse[1, i] = mse_eval(y_train, res[1, i, :])
    mse[2, i] = mse_eval(y_train, res[2, i, :])
    mse[3, i] = mse_eval(y_train, res[3, i, :])

# Bias correction
bias = (res - y_train).mean(axis=2)
res -= bias[:, :, np.newaxis]
mse -= bias[:, :]**2

# Plot results
fig, axs = plt.subplots(2, 4)
axs[0, 0].scatter(x_train, y_train, label='Test dataset')
axs[0, 1].scatter(x_train, y_train, label='Test dataset')
axs[0, 2].scatter(x_train, y_train, label='Test dataset')
axs[0, 3].scatter(x_train, y_train, label='Test dataset')

show_items = [0, 1, 2, 3, 5, 10, 15]
for i in show_items:
    if i > limit_grade:
        break
    axs[0, 0].plot(x_train, res[0, i, :], label='Grade ' + str(i))
    axs[0, 1].plot(x_train, res[1, i, :], label='Grade ' + str(i))
    axs[0, 2].plot(x_train, res[2, i, :], label='Grade ' + str(i))
    axs[0, 3].plot(x_train, res[3, i, :], label='Grade ' + str(i))
axs[0, 3].legend()
axs[0, 0].title.set_text('Polynomial Regression')
axs[0, 1].title.set_text('Gradient Descent')
axs[0, 2].title.set_text('Stochastic Gradient Descent')
axs[0, 3].title.set_text('Mini Batch Gradient Descent')

# Plot MSE
axs[1, 0].scatter(np.arange(limit_grade + 1), mse[0, :], label='MSE')
axs[1, 1].scatter(np.arange(limit_grade + 1), mse[1, :], label='MSE')
axs[1, 2].scatter(np.arange(limit_grade + 1), mse[2, :], label='MSE')
axs[1, 3].scatter(np.arange(limit_grade + 1), mse[3, :], label='MSE')
axs[1, 3].legend()
plt.show()
