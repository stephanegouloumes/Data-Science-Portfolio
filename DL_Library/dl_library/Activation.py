#===============================
#	Neural Network (version 4)
#
#	Activation :
#		* Class that handles multiple activation functions (Sigmoid, ReLu, Softmax)
#
#===============================

# Import external libraries
import numpy as np

class Activation(object):

	def __init__(self, activation_type):
		
		# Type of the activation function
		self.activation_type = activation_type

		# Caches
		self.A = None
		self.Z = None

	def forward(self, Z):
		if self.activation_type == "relu":
			return self.relu_forward(Z)
		elif self.activation_type == "sigmoid":
			return self.sigmoid_forward(Z)
		elif self.activation_type == "softmax":
			return self.softmax_forward(Z)

	def backward(self, dA):
		if self.activation_type == "relu":
			return self.relu_backward(dA)
		elif self.activation_type == "sigmoid":
			return self.sigmoid_backward(dA)
		elif self.activation_type == "softmax":
			return self.softmax_backward(dA)

	def relu_forward(self, Z):
		A = np.maximum(0, Z)

		self.A = A
		self.Z = Z

		return A

	def relu_backward(self, dA):
		Z = self.Z

		dZ = np.array(dA, copy = True)
		dZ[Z <= 0] = 0

		return dZ

	def sigmoid_forward(self, Z):
		A = 1 / (1 + np.exp(-Z))

		self.A = A
		self.Z = Z

		return A

	def sigmoid_backward(self, dA):
		Z = self.Z

		s = 1 / (1 + np.exp(-Z))
		dZ = dA * s * (1 - s)

		return dZ

	def softmax_forward(self, Z):
		t = np.exp(Z - np.max(Z))
		A = t / np.sum(t, axis = 0)

		return A

	def softmax_backward(self, dA):

		return dA

	def info(self):
		print("I'm a " + self.activation_type + " activation function !")