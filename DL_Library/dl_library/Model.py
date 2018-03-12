#===============================
#	Neural Network (version 4)
#
#	Model :
#		* Class that creates the neural network, handle forward, backpropagation and makes predictions
#
#===============================

# Import classes from library
from .Dense import Dense
from .Activation import Activation

# Import external libraries
import numpy as np

class Model(object):


	def __init__(self):
		
		# Layers of the model
		self.layers = []

		# Last activation output
		self.AL = None

	def add_layer(self, layer):
		self.layers.append(layer)

	def forward(self, X):
		A_prev = X
		for layer in self.layers:
			# print(A_prev.shape)
			A_prev = layer.forward(A_prev)
		self.AL = A_prev
		# print(self.AL.shape)

	def compute_cost(self, Y):
		m = Y.shape[1]

		if self.layers[-1].activation.activation_type == "sigmoid":
			cost = (-1 / m) * np.sum(np.multiply(Y, np.log(self.AL)) + np.multiply(1 - Y, np.log(1 - self.AL)))
			cost = np.squeeze(cost)
		elif self.layers[-1].activation.activation_type == "softmax":
			cost = -np.sum(Y * np.log(self.AL))

		return cost

	def backward(self, Y):
		m = self.AL.shape[1]

		# Loss
		if self.layers[-1].activation.activation_type == "sigmoid":
			dA = - (np.divide(Y, self.AL) - np.divide(1 - Y, 1 - self.AL))
		elif self.layers[-1].activation.activation_type == "softmax":
			dA = self.AL - Y

		L = len(self.layers)
		for l in reversed(range(L)):
			dA = self.layers[l].backward(dA)

	def update_parameters(self, learning_rate):
		for layer in self.layers:
			if type(layer) is Dense:
				layer.W = layer.W - learning_rate * layer.dW
				layer.b = layer.b - learning_rate * layer.db

	def fit(self, X, Y, learning_rate = 0.01, num_iterations = 100, print_cost = False):
		for i in range(num_iterations):
			self.forward(X)
			cost = self.compute_cost(Y)
			self.backward(Y)
			self.update_parameters(learning_rate)

			if print_cost and i % 10 == 0:
				accuracy = Y == np.round(self.AL)
				accuracy = np.mean(accuracy)
				print("Iteration (" + str(i + 1) + ") : " + str(cost) + ", accuracy = " + str(accuracy))

	def predict(self, X, Y, X_test = None, Y_test = None):
		self.forward(X)
		train_cost = self.compute_cost(Y)

		train_accuracy = Y == np.round(self.AL)
		train_accuracy = np.mean(train_accuracy)

		print()
		print("==========")
		print("Prediction")
		print("==========")
		print("Cost :", train_cost)
		print("Training accuracy :", train_accuracy)

		if X_test is not None:
			self.forward(X_test)
			test_cost = self.compute_cost(Y_test)
			test_accuracy = Y_test == np.round(self.AL)
			test_accuracy = np.mean(test_accuracy)
			print("Test accuracy :", test_accuracy)

	def print_model(self):
		print("======")
		print("Layers")
		print("======")
		print("")
		for layer in self.layers:
			if type(layer) is Dense:
				print(type(layer).__name__ + "(" + str(layer.size) + ")")
			if type(layer) is Activation:
				print(type(layer).__name__ + "(\"" + layer.activation + "\")")