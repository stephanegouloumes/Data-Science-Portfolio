#===============================
#	Neural Network (version 4)
#
#	Conv2D :
#		* Convulational Layer
#
#===============================

# Import external libraries
import numpy as np

class Conv2D(object):

	def __init__(self, dimensions, hyper_parameters, activation):

		# Size of the layer
		self.size = dimensions

		# Hyperparameters
		self.stride = hyper_parameters["stride"]
		self.pad = hyper_parameters["pad"]

		# Activation function
		self.activation = activation

		# Parameters
		self.W = None
		self.b = None

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
		(f, f, n_C_prev, n_C) = dimensions

		self.W = np.random.randn(f, f, n_C_prev, n_C) * 0.01
		self.b = np.random.randn(1, 1, 1, n_C) * 0.01

	def zero_pad(self, X, pad):
		# X is of shape (m, n_H, n_W, n_C)
		X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values = 0)

		return X_pad

	def conv_single_step(self, a_slice_prev, W, b):
		s = np.multiply(a_slice_prev, W) + b
		Z = np.sum(s)

		return Z

	def conv(self, A_prev):
		(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
		(f, f, n_C_prev, n_C) = self.W.shape

		stride = self.stride
		pad = self.pad

		n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
		n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

		Z = np.zeros((m, n_W, n_H, n_C))

		A_prev_pad = self.zero_pad(A_prev, pad)

		for i in range(m):
			a_prev_pad = A_prev_pad[i]
			for h in range(n_H):
				for w in range(n_W):
					for c in range(n_C):
						vert_start = h * stride
						vert_end = vert_start + f
						horiz_start = w * stride
						horiz_end = horiz_start + f

						a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

						Z[i, h, w, c] = self.conv_single_step(a_slice_prev, self.W[..., c], self.b[..., c])

		self.A_prev = A_prev
		self.Z = Z

		return Z

	def forward(self, A_prev):
		Z = self.conv(A_prev)
		self.A = self.activation.forward(Z)

		return self.A

	def backward(self, dA):
		(m, n_H_prev, n_W_prev, n_C_prev) = self.A_prev.shape
		(f, f, n_C_prev, n_C) = self.W.shape

		dZ = self.activation.backward(dA)
		(m, n_H, n_W, n_C) = dZ.shape

		dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
		dW = np.zeros((f, f, n_C_prev, n_C))
		db = np.zeros((1, 1, 1, n_C))

		pad = self.pad
		stride = self.stride

		A_prev_pad = self.zero_pad(self.A_prev, pad)
		dA_prev_pad = self.zero_pad(dA_prev, pad)

		for i in range(m):
			a_prev_pad = A_prev_pad[i]
			da_prev_pad = dA_prev_pad[i]

			for h in range(n_H):
				for w in range(n_W):
					for c in range(n_C):
						# Maybe delete the stride because we are going backward, so we're not passing values
						vert_start = h * stride
						vert_end = vert_start + f
						horiz_start = w * stride
						horiz_end = horiz_start + f

						a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

						da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.W[:, :, :, c] * dZ[i, h, w, c]
						dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
						db[:, :, :, c] += dZ[i, h, w, c]

			dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

		return dA_prev