#===============================
#	Neural Network (version 4)
#
#	Dense :
#		* Dense Layer
#
#===============================

# Import external libraries
import numpy as np

class Dense(object):

	# dimensions : takes two values (n_inputs, n_hidden)
	# 	* n_inputs : size fo layer l - 1
	#	* n_hidden : size of layer l
	def __init__(self, dimensions, activation, fully_connected = False):

		# Size of the layer
		self.size = dimensions[1]

		# Fully Connected layer
		self.fully_connected = fully_connected # If true, input_values will be flattened

		# Activation function of the layer
		self.activation = activation

		# Cache
		self.A_prev = None # Activation output of the previous layer
		self.Z = None
		self.A = None

		# Gradients
		self.dW = None
		self.db = None
		self.dA = None

		# Randomly initialize the parameters of each neuron
		self.initialize_parameters(dimensions)

	def initialize_parameters(self, dimensions):
		n_inputs, n_hidden = dimensions

		self.W = np.random.randn(n_hidden, n_inputs) * 0.01
		self.b = np.zeros((n_hidden, 1))

	def forward(self, A):
		self.A_prev = A

		# Flatten A to use it in the fully connected layer
		if self.fully_connected:
			A = A.reshape(A.shape[0], -1)
			A = A.T

		Z = np.dot(self.W, A) + self.b

		self.Z = Z
		self.A = self.activation.forward(Z)

		return self.A

	def backward(self, dA):
	
		if self.fully_connected:
			m = self.A_prev.shape[1]

			dZ = self.activation.backward(dA)

			A = self.A_prev.reshape(self.A_prev.shape[0], -1)
			A = A.T

			self.dW = (1 / m) * np.dot(dZ, A.T)
			self.db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
			self.dA = np.dot(self.W.T, dZ)
			self.dA = self.dA.reshape(self.A_prev.shape)

		else:
			m = self.A_prev.shape[1]

			dZ = self.activation.backward(dA)

			self.dW = (1 / m) * np.dot(dZ, self.A_prev.T)
			self.db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
			self.dA = np.dot(self.W.T, dZ)

		return self.dA

	def info(self):
		print(self.W.shape, self.b.shape)