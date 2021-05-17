from Dataset import Dataset
from LinearRegression import LinearRegression
from LinearRegressionB import LinearRegressionB
from MSE import MSE
import matplotlib.pyplot as plt

# Import dataset and split in training and test sets
ds = Dataset('data/income')
training_x, training_y, test_x, test_y = Dataset.split_dataset(ds.data['income'], ds.data['happiness'], 0.8)

# Create and fit a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(training_x, training_y)

# Create and fit a Linear Regression model
lrb_model = LinearRegressionB()
lrb_model.fit(training_x, training_y)

# Predict test set
lr_results = lr_model.predict(test_x)
lrb_results = lrb_model.predict(test_x)

# Evaluate MSE error
mse = MSE()
lr_mse = mse(test_y, lr_results)
lrb_mse = mse(test_y, lrb_results)
print("Linear regression MSE: " + str(round(lr_mse, 4)))
print("Linear regression B MSE: " + str(round(lrb_mse, 4)))

# Plot results
plt.figure(2, figsize=(8, 8))
plt.title("Test results")
plt.scatter(test_x, test_y, label='Test dataset')
plt.plot(test_x, lr_results, label='Linear Regression')
plt.plot(test_x, lrb_results, label='Linear Regression B')
plt.legend()
plt.show()


