# Import external libraries
import numpy as np

class LinearRegression(object):

	def fit(self, X, Y, regularization = None, alpha = None, learning_rate = 0.05, num_iterations = 1000, print_cost = False):

		# Setup variables
		self.regularization = regularization
		self.alpha = alpha
		self.learning_rate = learning_rate

		# Initialize the parameters
		self.initialize_parameters(X)

		costs = []

		for i in range(num_iterations):
			# Predict Y
			y_pred = self.predict(X)

			# Compute the cost
			cost = self.compute_cost(Y, y_pred)
			costs.append(cost)

			# Print the cost every 100 iterations
			if print_cost and i % 100 == 0:
				print(cost)

			# Compute the gradients
			grads = self.compute_gradients(X, Y, y_pred)

			# Update the parameters of the model using the gradients
			self.update_parameters(grads)

		# Plot the evolution of the cost
		if print_cost:
			plt.plot(range(0, num_iterations), costs)
			plt.show()

	def initialize_parameters(self, X):
		self.W = np.random.randn(X.shape[1], 1) * 0.01
		self.b = 0

	def predict(self, X):
		y_pred = np.dot(X, self.W) + self.b

		return y_pred

	def compute_cost(self, Y, y_pred):
		m = Y.shape[0]

		cost = (1 / (2 * m)) * np.sum(np.square(y_pred - Y))

		if self.regularization == "L2":
			cost = cost + (self.alpha / (2 * m)) * np.sum(self.W ** 2)

		return cost

	def compute_gradients(self, X, Y, y_pred):
		m = Y.shape[0]

		grads = {}
		grads["W"] = (1 / m) * np.dot(X.T, (y_pred - Y))
		grads["b"] = (1 / m) * np.sum(y_pred - Y)

		if self.regularization == "L2":
			grads["W"] = grads["W"] + (self.alpha / m) * self.W

		return grads

	def update_parameters(self, grads):
		self.W = self.W - self.learning_rate * grads["W"]
		self.b = self.b - self.learning_rate * grads["b"]


### TEST 1 : Linear Regression

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d

# X = np.linspace(0, 5, 100) 
# Y = 1.5 * X + 2
# noise = np.random.normal(0, 1, 100) 
# Y = Y + noise

# X = X.reshape(X.shape[0], 1)
# Y = Y.reshape(Y.shape[0], 1)

# # plt.scatter(X, Y)
# # plt.show()

# lr = LinearRegression()
# lr.fit(X, Y, learning_rate = 0.01, num_iterations = 200, print_cost = True)
# predictions = lr.predict(X)

# lr = LinearRegression()
# lr.fit(X, Y, regularization = "L2", alpha = 0.01, learning_rate = 0.01, num_iterations = 200, print_cost = True)
# predictions = lr.predict(X)

# plt.scatter(X, Y)
# plt.plot(X, X * lr.W + lr.b)
# plt.show()

### TEST 2 : Multivariate Linear Regression

# X1 = np.linspace(0, 5, 100)
# X2 = np.linspace(1, 10, 100)
# noise = np.random.normal(0, 5, 100) 
# X1 = X1 + noise
# noise = np.random.normal(0, 5, 100) 
# X2 = X2 + noise
# Y = (1.5 * X1) + (3.2 * X2) + 2
# noise = np.random.normal(0, 5, 100) 
# Y = Y + noise

# X1 = X1.reshape(X1.shape[0], 1)
# X2 = X2.reshape(X2.shape[0], 1)
# X = np.concatenate((X1, X2), axis = 1)
# Y = Y.reshape(Y.shape[0], 1)

# lr = LinearRegression()
# lr.fit(X, Y, learning_rate = 0.01, num_iterations = 200, print_cost = False)
# predictions = lr.predict(X)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# ax.scatter(X1, X2, Y, c = 'blue', marker = 'o', alpha = 1)
# plt.show()

### TEST 3 : Regularization

from scipy.io import loadmat
data = loadmat('data/ex5data1.mat')

X = data['X']
Y = data['y']

X_val = data["Xval"]
Y_val = data["yval"]

# yval = data['yval']
# Xval = np.c_[np.ones_like(data['Xval']), data['Xval']]

# print('X_train:', X_train.shape)
# print('y_train:', y_train.shape)
# print('Xval:', Xval.shape)
# print('yval:', yval.shape)

# plt.scatter(X[:,1], Y, s=50, c='r', marker='x', linewidths=1)
# plt.xlabel('Change in water level (x)')
# plt.ylabel('Water flowing out of the dam (y)')
# plt.ylim(ymin=0);

np.random.seed(42)

# lr = LinearRegression()
# lr.fit(X, Y, learning_rate = 0.002, num_iterations = 1000, print_cost = False)
# predictions = lr.predict(X)
# val_predictions = lr.predict(X_val)

# fig = plt.figure(figsize = (12, 10))
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224)

# ax1.plot(X, predictions)
# ax1.scatter(X, Y, s = 50, c = 'r', marker = 'x', linewidths = 1)
# ax2.plot(X_val, val_predictions)
# ax2.scatter(X_val, Y_val, s = 50, c = 'r', marker = 'x', linewidths = 1)

# lr = LinearRegression()
# lr.fit(X, Y, regularization = "L2", alpha = 0.1, learning_rate = 0.002, num_iterations = 1000, print_cost = False)
# predictions = lr.predict(X)
# val_predictions = lr.predict(X_val)

# ax3.plot(X, predictions)
# ax3.scatter(X, Y, s = 50, c = 'r', marker = 'x', linewidths = 1)
# ax4.plot(X_val, val_predictions)
# ax4.scatter(X_val, Y_val, s = 50, c = 'r', marker = 'x', linewidths = 1)
# plt.show()

# Learning curve

m = Y.shape[0]
train_error = np.zeros((m, 1))
val_error = np.zeros((m, 1))

for i in range(m):
	lr = LinearRegression()
	predictions = lr.fit(X[:i+1], Y[:i+1], regularization = "L2", alpha = 0.1, learning_rate = 0.01, num_iterations = 100, print_cost = False)
	predictions = lr.predict(X[:i+1])
	train_error[i] = lr.compute_cost(Y[:i+1], predictions)
	predictions = lr.predict(X_val[:i+1])
	val_error[i] = lr.compute_cost(Y_val[:i+1], predictions)

plt.plot(np.arange(1,13), train_error, label='Training error')
plt.plot(np.arange(1,13), val_error, label='Validation error')
plt.title('Learning curve for linear regression')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend();
plt.show()