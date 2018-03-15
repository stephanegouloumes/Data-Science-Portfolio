import numpy as np

class LogisticRegression(object):

	def fit(self, X, Y, learning_rate = 0.01, num_iterations = 1000, print_cost = False):

		# Setup variables
		self.learning_rate = learning_rate

		# Randomly initialize parameters
		self.initialize_parameters(X)

		costs = []

		for i in range(num_iterations):
			# Make prediction
			A = self.compute_A(X)

			# Compute the cost
			cost = self.compute_cost(Y, A)
			costs.append(cost)
			
			# Print the cost every 100 iterations
			if print_cost and i % 100 == 0:
				print(cost)

			# Compute the gradients
			grads = self.compute_gradients(X, Y, A)

			# Update the parameters
			self.update_parameters(grads)

		# Plot the evolution of the cost
		if print_cost:
			plt.plot(range(0, num_iterations), costs)
			plt.show()

	def initialize_parameters(self, X):
		self.W = np.random.randn(X.shape[1], 1) * 0.01
		self.b = 0

	def sigmoid(self, Z):
		A = 1 / (1 + np.exp(-Z))

		return A

	def compute_A(self, X):
		Z = np.dot(X, self.W) + self.b
		A = self.sigmoid(Z)

		return A

	def compute_cost(self, Y, A):
		m = Y.shape[0]

		cost = (-1 / m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A)))
		# cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

		return cost

	def compute_gradients(self, X, Y, A):
		m = Y.shape[0]
		grads = {}

		# grads["W"] = (1 / m) * np.sum(np.multiply((A - Y), X))
		grads["W"] = (1 / m) * np.dot(X.T, (A - Y))
		grads["b"] = (1 / m) * np.sum(A - Y)

		return grads

	def update_parameters(self, grads):
		self.W = self.W - self.learning_rate * grads["W"]
		self.b = self.b - self.learning_rate * grads["b"]

	def predict(self, X, Y):
		y_pred = self.sigmoid(np.dot(X, self.W) + self.b)
		y_pred = y_pred > 0.5
		accuracy = np.mean(y_pred == Y)

		print("Accuracy : " + str(accuracy))

		return y_pred


### TESTING 1

import pandas as pd
import matplotlib.pyplot as plt
import os

path = os.getcwd() + '\data\ex2data1.txt'
data = pd.read_csv(path, header = None, names = ['Exam 1', 'Exam 2', 'Admitted'])
# print(data.head())

# positive = data[data['Admitted'].isin([1])]
# negative = data[data['Admitted'].isin([0])]

# fig, ax = plt.subplots(figsize = (12,8))
# ax.scatter(positive['Exam 1'], positive['Exam 2'], s = 50, c = 'b', marker = 'o', label = 'Admitted')
# ax.scatter(negative['Exam 1'], negative['Exam 2'], s = 50, c = 'r', marker = 'x', label = 'Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
# plt.show()

cols = data.shape[1]  
X = data.iloc[:, 0:cols-1]  
Y = data.iloc[:, cols-1:cols]
X = np.array(X.values)  
Y = np.array(Y.values)

lr = LogisticRegression()
lr.fit(X, Y, learning_rate = 0.0014, num_iterations = 10000, print_cost = True)
lr.predict(X, Y)